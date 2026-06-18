"""
train.py — Fine-tune CLIP ViT-L/14 with ArcFace loss on the training set.

Usage
-----
python train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 64
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

from dataset import CelebRetrievalDataset, get_train_transforms, get_val_transforms
from model import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr_schedule(optimizer, epoch: int, total_epochs: int,
                        base_lr_backbone: float, base_lr_head: float,
                        warmup_epochs: int = 5) -> None:
    """
    In-place cosine annealing with linear warmup.
    The first param group is the backbone, the second is the ArcFace head.
    """
    if epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        import math
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    optimizer.param_groups[0]["lr"] = base_lr_backbone * factor
    optimizer.param_groups[1]["lr"] = base_lr_head    * factor


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Compute top-1 accuracy on a validation split using cosine similarity
    between embeddings and per-class prototype centroids."""
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        with autocast(device_type=device.type):
            emb = model.encode(images)
        all_embeddings.append(emb.float().cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)   # (N, D)
    all_labels     = torch.cat(all_labels, dim=0)       # (N,)

    # Build per-class centroids from the validation set itself
    num_classes = all_labels.max().item() + 1
    centroids   = torch.zeros(num_classes, all_embeddings.size(1))
    counts      = torch.zeros(num_classes)
    for emb, lbl in zip(all_embeddings, all_labels):
        centroids[lbl] += emb
        counts[lbl]    += 1
    # Only keep classes that appear in the val set
    valid_mask = counts > 0
    centroids[valid_mask] /= counts[valid_mask].unsqueeze(1)
    centroids = torch.nn.functional.normalize(centroids[valid_mask], dim=1)
    valid_classes = valid_mask.nonzero(as_tuple=True)[0]

    # Nearest-centroid classification
    sims = all_embeddings @ centroids.T    # (N, C_valid)
    preds_in_valid = sims.argmax(dim=1)    # index into valid_classes
    preds = valid_classes[preds_in_valid]  # original class indices

    acc = (preds == all_labels).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"[Train] Using device: {device}")

    # -- Datasets -----------------------------------------------------------
    full_dataset = CelebRetrievalDataset(
        args.data_dir,
        transform=get_train_transforms(image_size=224),
    )
    num_classes = len(full_dataset.classes)
    print(f"[Train] Number of identity classes: {num_classes}")

    # 90/10 train-val split (deterministic seed)
    n_val   = max(1, int(0.10 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Use clean transforms for the val split
    val_set.dataset = CelebRetrievalDataset(
        args.data_dir,
        transform=get_val_transforms(image_size=224),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # -- Model --------------------------------------------------------------
    model = build_model(
        num_classes=num_classes,
        unfreeze_blocks=args.unfreeze_blocks,
        device=str(device),
    )

    # -- Optimiser (two param groups: backbone vs. head) --------------------
    backbone_params = [p for n, p in model.named_parameters()
                       if "arcface" not in n and p.requires_grad]
    head_params     = list(model.arcface.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler()   # mixed-precision scaler

    # -- Checkpointing setup ------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    # -- Training loop ------------------------------------------------------
    print(f"\n[Train] Starting training for {args.epochs} epochs\n")
    for epoch in range(args.epochs):
        cosine_lr_schedule(
            optimizer, epoch, args.epochs,
            args.lr_backbone, args.lr_head, args.warmup_epochs
        )

        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        t0         = time.time()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                loss, embeddings = model(images, labels)

            scaler.scale(loss).backward()
            # Gradient clipping to stabilise transformer fine-tuning
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total += labels.size(0)

            if (step + 1) % 20 == 0:
                avg_loss = total_loss / (step + 1)
                elapsed  = time.time() - t0
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{step+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}  "
                      f"LR backbone: {optimizer.param_groups[0]['lr']:.2e}  "
                      f"Elapsed: {elapsed:.0f}s")

        # -- Validation -----------------------------------------------------
        val_acc = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch+1}] "
              f"Avg Loss: {total_loss/len(train_loader):.4f}  "
              f"Val Acc (centroid-NN): {val_acc:.4f}\n")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_to_idx": full_dataset.class_to_idx,
            }, ckpt_path)
            print(f"  ✓ New best model saved ({val_acc:.4f}) → {ckpt_path}\n")

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, out_dir / f"ckpt_epoch{epoch+1:03d}.pth")

    print(f"\n[Train] Done. Best val acc: {best_val_acc:.4f}")
    print(f"[Train] Best model: {out_dir / 'best_model.pth'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune CLIP ViT-L/14 with ArcFace")
    p.add_argument("--data_dir",       type=str, required=True,
                   help="Root directory of the training set (identity subfolders).")
    p.add_argument("--output_dir",     type=str, default="./checkpoints")
    p.add_argument("--epochs",         type=int, default=50)
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--lr_backbone",    type=float, default=1e-5)
    p.add_argument("--lr_head",        type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--warmup_epochs",  type=int, default=5)
    p.add_argument("--unfreeze_blocks",type=int, default=6,
                   help="Number of final ViT transformer blocks to unfreeze.")
    p.add_argument("--num_workers",    type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())