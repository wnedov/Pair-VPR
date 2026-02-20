import argparse
import torch
import os
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import time

# PairVPR Imports
from pairvpr.models.pairvpr import PairVPRNet
from pairvpr.configs import pairvpr_speed
from pairvpr.eval.wildcross_loader import WildCrossDataset

def get_cfg_from_args_eval(args):
    default_cfg = OmegaConf.create(pairvpr_speed)
    if args.config:
        cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(default_cfg, cfg)
    else:
        cfg = default_cfg
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_seq", required=True, help="Path to Database Sequence")
    parser.add_argument("--query_seq", required=True, help="Path to Query Sequence")
    parser.add_argument("--trained_ckpt", default="trained_models/pairvpr-vitB.pth")
    parser.add_argument("--config", default=None, help="Path to custom config yaml")
    parser.add_argument("--output_csv", default="results.csv", help="Path to save the output CSV")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=50, help="Number of candidates to rerank")
    parser.add_argument("--subsample_step", type=int, default=5, help="Subsample step (default: 5)")
    
    args = parser.parse_args()

    # 1. Setup Model
    cfg = get_cfg_from_args_eval(args)
    cfg.eval.refinetopcands = args.top_k 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Loading PairVPR from {args.trained_ckpt} on {device} ---")
    model = PairVPRNet(cfg)
    ckpt = torch.load(args.trained_ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    # 2. Setup Dataset
    print(f"--- Setting up Dataset with Subsample Step: {args.subsample_step} ---")
    dataset = WildCrossDataset(
        db_route_dir=args.db_seq, 
        query_route_dir=args.query_seq,
        input_size=(322, 322), 
        subsample_step=args.subsample_step 
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 3. Extract Global Features
    print("--- Extracting Global Features ---")
    global_feats = []
    with torch.no_grad():
        for batch_imgs, _ in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)
            _, descriptors = model(batch_imgs, None, mode='global')
            global_feats.append(descriptors.detach().cpu())

    global_feats = torch.cat(global_feats, dim=0)
    db_feats = global_feats[:dataset.num_references]
    q_feats = global_feats[dataset.num_references:]

    # 4. Global Retrieval (FAISS)
    print(f"--- Running Global Retrieval (Top-{args.top_k}) ---")
    faiss_index = faiss.IndexFlatL2(db_feats.shape[1])
    faiss_index.add(db_feats.numpy())
    _, predictions = faiss_index.search(q_feats.numpy(), args.top_k)

    # 5. Pairwise Reranking
    print("--- Starting Pairwise Reranking ---")
    results_log = []
    skipped_count = 0
    
    with torch.no_grad():
        for q_idx in tqdm(range(len(q_feats)), desc="Reranking"):
            
            # --- FILTERING: Check Ground Truth Existence ---
            # dataset.ground_truth is a list of lists of valid DB indices
            gt_matches = dataset.ground_truth[q_idx]
            if len(gt_matches) == 0:
                skipped_count += 1
                continue 
            
            candidates = predictions[q_idx]
            
            # Load Images
            abs_q_idx = dataset.num_references + q_idx
            q_img_tensor, _ = dataset[abs_q_idx]
            q_img_tensor = q_img_tensor.to(device).unsqueeze(0)
            
            db_imgs_list = [dataset[db_idx][0] for db_idx in candidates]
            db_imgs_tensor = torch.stack(db_imgs_list).to(device)

            # Rerank
            q_feat_map, _ = model(q_img_tensor, None, mode='global')
            db_feat_maps, _ = model(db_imgs_tensor, None, mode='global')
            
            q_feat_maps_rep = q_feat_map.repeat(len(candidates), 1, 1)
            scores_a = model(q_feat_maps_rep, db_feat_maps, "pairvpr")
            scores_b = model(db_feat_maps, q_feat_maps_rep, "pairvpr")
            scores = (scores_a + scores_b).cpu().squeeze(1)
            
            # Sort Results
            sorted_indices = scores.argsort(descending=True)
            
            # --- Top-1 Logic ---
            best_local_idx = sorted_indices[0].item()
            best_db_idx = candidates[best_local_idx]
            
            # --- Top-5 Logic (Added) ---
            k_r5 = min(5, len(candidates))
            top5_local_indices = sorted_indices[:k_r5].cpu().numpy()
            top5_db_indices = candidates[top5_local_indices]
            
            # Check if any of the top 5 are in the ground truth list
            is_success_r5 = np.any(np.isin(top5_db_indices, gt_matches))

            # --- Result Calculation (Top-1) ---
            # Poses are stored as (x,y,z) to match official WildCross evaluation.
            # Inter-sequence CSVs/logs report x,y only.
            gt_x, gt_y = dataset.q_poses[q_idx][:2]
            pred_x, pred_y = dataset.db_poses[best_db_idx][:2]
            
            error_m = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            is_success = error_m <= dataset.pos_thresh
            
            results_log.append({
                # Match WildCross-Replication inter CSV shape:
                # query_idx,x,y,pred_x,pred_y,error_m,success (+ success_r5 kept)
                'query_idx': q_idx,
                'x': gt_x, 'y': gt_y,
                'pred_x': pred_x, 'pred_y': pred_y,
                'error_m': error_m,
                'success': is_success,
                'success_r5': is_success_r5,
            })

    print(f"Skipped {skipped_count} queries (no ground truth match).")

    # 6. Save CSV
    df_out = pd.DataFrame(results_log)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    
    if len(df_out) > 0:
        print(f"Recall@1: {df_out['success'].mean() * 100:.2f}%")
        print(f"Recall@5: {df_out['success_r5'].mean() * 100:.2f}%")
    
    print(f"Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()