import argparse
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

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
    parser.add_argument("--seq_dir", required=True, help="Path to the Sequence (e.g. data/V-01)")
    parser.add_argument("--trained_ckpt", default="trained_models/pairvpr-vitB.pth")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_csv", default="intra_results.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=20, help="Candidates to rerank")
    
    # Official constraints
    parser.add_argument("--time_thresh", type=float, default=600.0)
    parser.add_argument("--pos_thresh", type=float, default=25.0)
    parser.add_argument("--subsample_step", type=int, default=5, help="Subsample step (default: 5)")
    
    args = parser.parse_args()

    # 1. Setup Model
    cfg = get_cfg_from_args_eval(args)
    cfg.eval.refinetopcands = args.top_k
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Loading PairVPR from {args.trained_ckpt} ---")
    model = PairVPRNet(cfg)
    ckpt = torch.load(args.trained_ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    # 2. Setup Dataset
    print(f"--- Loading Sequence Data (Subsample={args.subsample_step}) ---")
    dataset = WildCrossDataset(
        db_route_dir=args.seq_dir, 
        query_route_dir=args.seq_dir, 
        input_size=(322, 322),
        subsample_step=args.subsample_step
    )
    
    # Dataset is doubled [DB|Query], take first half
    total_len = len(dataset) // 2
    timestamps = dataset.db_timestamps
    # Use 3D poses (x,y,z) for distances to match official WildCross evaluation.
    poses = torch.tensor(dataset.db_poses, dtype=torch.float32)
    
    # 3. Extract Global Features
    print("--- Extracting Global Features ---")
    subset = torch.utils.data.Subset(dataset, list(range(total_len)))
    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    global_feats = []
    with torch.no_grad():
        for batch_imgs, _ in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)
            _, descriptors = model(batch_imgs, None, mode='global')
            global_feats.append(descriptors.detach().cpu())
    global_feats = torch.cat(global_feats, dim=0)
    
    # 4. Pre-calculate Distances
    print("--- Pre-calculating Distance Matrices ---")
    position_dists = torch.cdist(poses, poses) 

    # 5. Intra-Sequence Evaluation Loop
    results_log = []
    time_start = timestamps[0]
    
    print(f"--- Running Intra-Sequence Eval (Time Thresh: {args.time_thresh}s) ---")
    
    count_skipped = 0
    
    for q_idx in tqdm(range(total_len), desc="Processing Frames"):
        q_timestamp = timestamps[q_idx]
        gt_x, gt_y = poses[q_idx][:2].tolist()

        # A. Time Threshold Check
        if (q_timestamp - time_start - args.time_thresh) < 0:
            count_skipped += 1
            continue

        # B. Identify Valid Database Indices
        cutoff_mask = timestamps < (q_timestamp - args.time_thresh)
        valid_indices = np.where(cutoff_mask)[0]
        
        if len(valid_indices) == 0: 
            count_skipped += 1
            continue

        # C. Revisit Check
        valid_pos_dists = position_dists[q_idx, valid_indices]
        if torch.min(valid_pos_dists) > args.pos_thresh:
            count_skipped += 1
            continue

        # D. Global Retrieval
        q_feat = global_feats[q_idx].unsqueeze(0)
        db_feats_valid = global_feats[valid_indices]
        
        dists_global = torch.cdist(q_feat, db_feats_valid)
        k = min(args.top_k, len(valid_indices))
        _, topk_local_indices = torch.topk(dists_global, k, largest=False)
        candidate_indices = valid_indices[topk_local_indices[0].numpy()]

        # E. PairVPR Reranking
        q_img, _ = dataset[q_idx]
        q_img_tensor = q_img.to(device).unsqueeze(0)
        
        db_imgs_list = [dataset[db_idx][0] for db_idx in candidate_indices]
        db_imgs_tensor = torch.stack(db_imgs_list).to(device)
        
        with torch.no_grad():
            q_dense, _ = model(q_img_tensor, None, mode='global')
            db_dense, _ = model(db_imgs_tensor, None, mode='global')
            
            q_dense_rep = q_dense.repeat(len(candidate_indices), 1, 1)
            scores_a = model(q_dense_rep, db_dense, "pairvpr")
            scores_b = model(db_dense, q_dense_rep, "pairvpr")
            scores = (scores_a + scores_b).cpu().squeeze(1)
            
            sorted_indices = scores.argsort(descending=True)
            
            # Top-1
            best_cand_idx = sorted_indices[0].item()
            pred_idx = candidate_indices[best_cand_idx]

            # Top-5 Logic (Added)
            k_r5 = min(5, len(candidate_indices))
            top5_local_indices = sorted_indices[:k_r5].cpu().numpy()
            top5_preds = candidate_indices[top5_local_indices]
            
            # Check if any top-5 are within 25m
            dists_top5 = position_dists[q_idx, top5_preds]
            is_success_r5 = torch.any(dists_top5 <= args.pos_thresh).item()

        # F. Calculate Result
        pred_x, pred_y = poses[pred_idx][:2].tolist()
        error_m = position_dists[q_idx, pred_idx].item()
        is_success = error_m <= args.pos_thresh

        results_log.append({
            'query_idx': q_idx,
            'timestamp': float(q_timestamp),
            'x': gt_x, 'y': gt_y,
            'pred_idx': int(pred_idx),
            'pred_x': pred_x, 'pred_y': pred_y,
            'error_m': float(error_m),
            'success': bool(is_success),
            'success_r5': bool(is_success_r5)
        })

    # Save
    df = pd.DataFrame(results_log)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    
    if len(df) > 0:
        print(f"Recall@1: {df['success'].mean() * 100:.2f}%")
        print(f"Recall@5: {df['success_r5'].mean() * 100:.2f}%")
    
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()