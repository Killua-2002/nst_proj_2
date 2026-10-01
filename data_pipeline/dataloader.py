"""
dataloader.py
Module quản lý dữ liệu PyTorch Dataset cho dự án NST.
"""
import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 256

def get_image_paths(dataset_dir: Path):
    rows = []
    img_dir = dataset_dir / "images"
    if not img_dir.exists(): return rows
    for p in sorted(img_dir.glob("*.png")):
        n = p.name
        rows.append((
            str(p),
            str(dataset_dir/"priors_skeleton"/n),
            str(dataset_dir/"priors_distance"/n),
            str(dataset_dir/"mask_1"/n),
            str(dataset_dir/"mask_2"/n),
            str(dataset_dir/"mask_3"/n),
            str(dataset_dir/"overlap_12"/n),
            str(dataset_dir/"overlap_23"/n),
            str(dataset_dir/"overlap_13"/n),
            str(dataset_dir/"overlap_123"/n)
        ))
    return rows

class NSTDataset(Dataset):
    def __init__(self, rows, is_train=False, drop_priors=False):
        self.rows = rows
        self.is_train = is_train
        self.drop_priors = drop_priors

    def __len__(self):
        return len(self.rows)

    def _read_img(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Loi doc anh: {path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0

    def _read_mask(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Loi doc mask: {path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return (img > 127).astype(np.float32)

    def __getitem__(self, idx):
        paths = self.rows[idx]

        # Doc input
        img  = self._read_img(paths[0])
        skel = self._read_img(paths[1])
        dist = self._read_img(paths[2])

        # Doc label (7 masks)
        m1   = self._read_mask(paths[3])
        m2   = self._read_mask(paths[4])
        m3   = self._read_mask(paths[5])
        o12  = self._read_mask(paths[6])
        o23  = self._read_mask(paths[7])
        o13  = self._read_mask(paths[8])
        o123 = self._read_mask(paths[9])

        # Canonical Ordering: sort masks by area descending
        # Solves Permutation Ambiguity (symmetry problem)
        areas    = [m1.sum(), m2.sum(), m3.sum()]
        sort_idx = np.argsort(areas)[::-1]

        orig_masks    = [m1, m2, m3]
        orig_overlaps = {
            (0, 1): o12, (1, 0): o12,
            (1, 2): o23, (2, 1): o23,
            (0, 2): o13, (2, 0): o13,
        }

        m1  = orig_masks[sort_idx[0]]
        m2  = orig_masks[sort_idx[1]]
        m3  = orig_masks[sort_idx[2]]
        o12 = orig_overlaps[(sort_idx[0], sort_idx[1])]
        o23 = orig_overlaps[(sort_idx[1], sort_idx[2])]
        o13 = orig_overlaps[(sort_idx[0], sort_idx[2])]

        # Stack (C, H, W)
        x = np.stack([img, skel, dist], axis=0)
        y = np.stack([m1, m2, m3, o12, o23, o13, o123], axis=0)

        if self.drop_priors:
            x[1:] = 0.0

        # Data augmentation — train split only
        if self.is_train:
            if random.random() > 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=2).copy()
            if random.random() > 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()

        return torch.from_numpy(x), torch.from_numpy(y)


def make_ds(dataset_dir: Path, batch_size: int, split: str = 'train',
            drop_priors: bool = False):
    """
    Create a DataLoader with deterministic 80 / 10 / 10 split.

    split='train' -> first 80%  (5,600 samples for 7,000-image dataset)
    split='val'   -> next  10%  (  700 samples)  — used for early stopping
    split='test'  -> last  10%  (  700 samples)  — held-out, used only in evaluate.py

    Augmentation (random H/V flip) is applied only for split='train'.
    """
    rows = get_image_paths(dataset_dir)
    if not rows:
        raise ValueError(
            f"No images found in {dataset_dir / 'images'}. "
            "Make sure dataset.zip is extracted correctly."
        )

    n         = len(rows)
    train_end = int(n * 0.8)   # 80 %
    val_end   = int(n * 0.9)   # 90 %  (val = [80%, 90%))

    if split == 'train':
        rows        = rows[:train_end]
        is_train_aug = True
    elif split == 'val':
        rows        = rows[train_end:val_end]
        is_train_aug = False
    elif split == 'test':
        rows        = rows[val_end:]
        is_train_aug = False
    else:
        raise ValueError(
            f"split must be 'train', 'val', or 'test'. Got: '{split}'"
        )

    if not rows:
        raise ValueError(
            f"Dataset too small to create split '{split}'. Total images: {n}"
        )

    dataset = NSTDataset(rows, is_train=is_train_aug, drop_priors=drop_priors)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == 'train'),
        num_workers = 0,
        pin_memory  = True,
    )
    return loader, len(rows)
