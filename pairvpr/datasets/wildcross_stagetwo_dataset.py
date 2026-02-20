import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from natsort import natsorted
from torch.utils.data import Dataset
import torchvision.transforms as T


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN_STD = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

DEFAULT_WILDCROSS_ROUTES = [
    "K-01",
    "K-02",
    "K-03",
    "K-04",
    "V-01",
    "V-02",
    "V-03",
    "V-04",
]


def _resolve_wildcross_path(cfg, dsetroot: str) -> str:
    location = "WildCross-Replication/data"
    if hasattr(cfg, "dataset_locations") and hasattr(cfg.dataset_locations, "wildcross"):
        location = cfg.dataset_locations.wildcross
    return os.path.join(dsetroot, location)


def load_train_dataset(cfg, dsetroot: str, routes: Optional[List[str]] = None):
    mean_dataset = IMAGENET_MEAN_STD["mean"]
    std_dataset = IMAGENET_MEAN_STD["std"]

    image_size = (cfg.augmentation.img_res, cfg.augmentation.img_res)
    train_transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_dataset, std=std_dataset),
        ]
    )

    base_path = _resolve_wildcross_path(cfg, dsetroot)
    wildcross_cfg = getattr(cfg, "wildcross", None)

    img_per_place = 4
    min_img_per_place = 4
    place_cell_size_m = 10.0
    random_sample_from_each_place = True

    if wildcross_cfg is not None:
        img_per_place = int(getattr(wildcross_cfg, "img_per_place", img_per_place))
        min_img_per_place = int(getattr(wildcross_cfg, "min_img_per_place", min_img_per_place))
        place_cell_size_m = float(getattr(wildcross_cfg, "place_cell_size_m", place_cell_size_m))
        random_sample_from_each_place = bool(
            getattr(wildcross_cfg, "random_sample_from_each_place", random_sample_from_each_place)
        )

    routes = routes or DEFAULT_WILDCROSS_ROUTES

    return WildCrossPlacesDataset(
        base_path=base_path,
        routes=routes,
        img_per_place=img_per_place,
        min_img_per_place=min_img_per_place,
        place_cell_size_m=place_cell_size_m,
        random_sample_from_each_place=random_sample_from_each_place,
        transform=train_transform,
    )


class WildCrossPlacesDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        routes: List[str],
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        place_cell_size_m: float = 10.0,
        random_sample_from_each_place: bool = True,
        transform=None,
    ):
        super().__init__()
        self.base_path = base_path
        self.routes = routes
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.place_cell_size_m = place_cell_size_m
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"WildCross data path does not exist: {self.base_path}")
        if self.img_per_place > self.min_img_per_place:
            raise ValueError("img_per_place should be <= min_img_per_place")
        if self.place_cell_size_m <= 0:
            raise ValueError("place_cell_size_m should be > 0")

        self.records = self._load_records()
        self.place_to_records = self._build_places(self.records)
        self.place_ids = sorted(self.place_to_records.keys())

        if len(self.place_ids) == 0:
            raise RuntimeError(
                "No WildCross places satisfy min_img_per_place. "
                "Try lowering wildcross.min_img_per_place or increasing place_cell_size_m."
            )

    def _load_records(self):
        records = []
        for route in self.routes:
            route_dir = os.path.join(self.base_path, route)
            if not os.path.isdir(route_dir):
                raise FileNotFoundError(f"WildCross route directory not found: {route_dir}")

            image_dir = os.path.join(route_dir, "images_shrunk")
            if not os.path.isdir(image_dir):
                image_dir = os.path.join(route_dir, "images")
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(f"No images_shrunk/ or images/ found for route: {route_dir}")

            pose_csv = os.path.join(route_dir, "camera_poses.csv")
            if not os.path.exists(pose_csv):
                raise FileNotFoundError(f"camera_poses.csv not found for route: {route_dir}")

            all_imgs = natsorted(
                [
                    os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            if len(all_imgs) == 0:
                raise RuntimeError(f"No images found in {image_dir}")

            df = pd.read_csv(pose_csv)
            df.columns = df.columns.str.strip()
            poses = self._extract_xyz(df, pose_csv)

            min_len = min(len(all_imgs), len(poses))
            all_imgs = all_imgs[:min_len]
            poses = poses[:min_len]

            for img_path, (x, y, z) in zip(all_imgs, poses):
                records.append(
                    {
                        "route": route,
                        "img_path": img_path,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                    }
                )

        return records

    @staticmethod
    def _extract_xyz(df: pd.DataFrame, pose_csv: str):
        if {"x", "y", "z"}.issubset(df.columns):
            return df[["x", "y", "z"]].values.astype(float)
        if {"p_x", "p_y", "p_z"}.issubset(df.columns):
            return df[["p_x", "p_y", "p_z"]].values.astype(float)
        if {"x", "y"}.issubset(df.columns):
            xy = df[["x", "y"]].values.astype(float)
            return np.concatenate([xy, np.zeros((len(xy), 1), dtype=float)], axis=1)
        if {"p_x", "p_y"}.issubset(df.columns):
            xy = df[["p_x", "p_y"]].values.astype(float)
            return np.concatenate([xy, np.zeros((len(xy), 1), dtype=float)], axis=1)
        raise ValueError(f"Could not parse x/y(/z) columns in {pose_csv}")

    def _build_places(self, records):
        places = {}
        for rec in records:
            cell_x = int(np.floor(rec["x"] / self.place_cell_size_m))
            cell_y = int(np.floor(rec["y"] / self.place_cell_size_m))
            key = (cell_x, cell_y)
            if key not in places:
                places[key] = []
            places[key].append(rec)

        # Keep only places with enough views
        place_to_records = {}
        for new_id, key in enumerate(sorted(places.keys())):
            if len(places[key]) >= self.min_img_per_place:
                place_to_records[new_id] = places[key]
        return place_to_records

    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Image {path} could not be loaded")
            return Image.new("RGB", (224, 224))

    def __getitem__(self, index):
        place_id = self.place_ids[index]
        place_records = self.place_to_records[place_id]

        if self.random_sample_from_each_place:
            sampled = random.sample(place_records, self.img_per_place)
        else:
            sampled = place_records[: self.img_per_place]

        imgs = []
        for rec in sampled:
            img = self.image_loader(rec["img_path"])
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        label = torch.tensor(place_id, dtype=torch.long).repeat(self.img_per_place)
        return torch.stack(imgs), label

    def __len__(self):
        return len(self.place_ids)
