import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from natsort import natsorted

class WildCrossDataset(Dataset):
    def __init__(self, db_route_dir, query_route_dir, input_size=(322, 322), pos_thresh=25.0, subsample_step=1):
        super().__init__()
        self.input_size = input_size
        self.subsample_step = subsample_step
        self.pos_thresh = pos_thresh
        
        print(f"\n--- Initializing WildCross Dataset (Step={subsample_step}) ---")
        
        # Load Database
        self.db_imgs, self.db_poses, self.db_timestamps = self._load_route(db_route_dir, "Database")
        # Load Query
        self.q_imgs, self.q_poses, self.q_timestamps = self._load_route(query_route_dir, "Query")
        
        if len(self.db_imgs) == 0:
            raise ValueError(f"CRITICAL: No images found. Check paths!")

        self.num_references = len(self.db_imgs)
        self.num_queries = len(self.q_imgs)
        
        # Combined list for the DataLoader (DB first, then Query)
        self.all_images = self.db_imgs + self.q_imgs
        
        print(f"Summary: {self.num_references} DB images, {self.num_queries} Query images.")

        print("Calculating Ground Truth matrix...")
        self.ground_truth = self._calc_ground_truth(self.db_poses, self.q_poses, pos_thresh)

        self.transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_route(self, route_dir, label):
        if not os.path.exists(route_dir):
            raise FileNotFoundError(f"{label} path not found: {route_dir}")

        # 1. Find Images
        img_dir = os.path.join(route_dir, 'images_shrunk')
        if not os.path.exists(img_dir):
            img_dir = os.path.join(route_dir, 'images')
        
        # 2. List Files
        all_files = natsorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])

        # 3. Load CSV
        pose_path = os.path.join(route_dir, 'camera_poses.csv')
        try:
            # We try to read as strings first to prevent pandas from messing up precision
            df = pd.read_csv(pose_path, dtype={'%time': str, 'timestamp': str})
        except:
             df = pd.read_csv(pose_path) # Fallback
             
        df.columns = df.columns.str.strip()
        
        # Standardize Time Column
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)
        if time_col:
            df = df.sort_values(time_col)
            # FIX: Force convert to float here so we can do math later
            timestamps = df[time_col].values.astype(float)
        else:
            timestamps = np.zeros(len(df)) 

        # Standardize Position Columns
        #
        # WildCross official evaluation uses 3D (x,y,z) distances for:
        # - ground-truth positives
        # - intra-sequence revisit checks / error distances
        #
        # We therefore load poses as (x,y,z) when available, and fall back
        # to z=0 if only 2D is present.
        if {'x', 'y', 'z'}.issubset(df.columns):
            poses = df[['x', 'y', 'z']].values.astype(float)
        elif {'p_x', 'p_y', 'p_z'}.issubset(df.columns):
            poses = df[['p_x', 'p_y', 'p_z']].values.astype(float)
        elif {'x', 'y'}.issubset(df.columns):
            xy = df[['x', 'y']].values.astype(float)
            poses = np.concatenate([xy, np.zeros((len(xy), 1), dtype=float)], axis=1)
        elif {'p_x', 'p_y'}.issubset(df.columns):
            xy = df[['p_x', 'p_y']].values.astype(float)
            poses = np.concatenate([xy, np.zeros((len(xy), 1), dtype=float)], axis=1)
        else:
            raise ValueError(f"Could not find x/y(/z) or p_x/p_y(/p_z) columns in {pose_path}")

        # 4. Subsample
        min_len = min(len(all_files), len(poses))
        all_files = all_files[:min_len]
        poses = poses[:min_len]
        timestamps = timestamps[:min_len]

        if self.subsample_step > 1:
            print(f"[{label}] Subsampling by {self.subsample_step}...")
            all_files = all_files[::self.subsample_step]
            poses = poses[::self.subsample_step]
            timestamps = timestamps[::self.subsample_step]

        print(f"[{label}] Final count: {len(all_files)}")
        return all_files, poses, timestamps

    def _calc_ground_truth(self, db_poses, q_poses, threshold):
        nbrs = NearestNeighbors(radius=threshold, metric='euclidean')
        nbrs.fit(db_poses)
        _, indices = nbrs.radius_neighbors(q_poses)
        return [arr.tolist() for arr in indices]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        path = self.all_images[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, idx