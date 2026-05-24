"""
dataset.py — CelebRetrieval dataset with domain-gap-aware augmentations.
"""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Augmentation factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Domain-gap-aware augmentation pipeline for training.

    Design choices:
    - ColorJitter: the synthetic gallery may have shifted color distributions.
    - RandomGrayscale: forces identity-based rather than color-based matching.
    - GaussianBlur: synthetic images sometimes exhibit softness / generation blur.
    - Scale range (0.65–1.0): aggressive crops force focus on face regions.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.10),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        # CLIP normalization statistics
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Clean centre-crop pipeline for validation / inference."""
    return T.Compose([
        T.Resize(int(image_size * 1.143)),   # 256 for image_size=224
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class CelebRetrievalDataset(Dataset):
    """
    Loads the training set which is organised as:

        data_root/
            identity_1/
                img_a.jpg
                img_b.jpg
            identity_2/
                ...

    Each subfolder name is treated as a class label.
    """

    def __init__(self, data_root: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []   # (path, label)
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        data_root = Path(data_root)
        identity_folders = sorted([
            d for d in data_root.iterdir()
            if d.is_dir()
        ])

        for idx, folder in enumerate(identity_folders):
            label_name = folder.name
            self.classes.append(label_name)
            self.class_to_idx[label_name] = idx
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in VALID_EXTS:
                    self.samples.append((str(img_path), idx))

        print(f"[Dataset] {len(self.classes)} identities, "
              f"{len(self.samples)} images loaded from {data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FolderImageDataset(Dataset):
    """
    Flat folder dataset for query / gallery inference — no labels needed.
    Returns (tensor, filename).
    """

    def __init__(self, folder: str, transform=None):
        self.folder = Path(folder)
        self.transform = transform
        self.filenames: list[str] = sorted([
            f.name for f in self.folder.iterdir()
            if f.suffix.lower() in VALID_EXTS
        ])
        print(f"[Dataset] {len(self.filenames)} images loaded from {folder}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        fname = self.filenames[index]
        image = Image.open(self.folder / fname).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, fname