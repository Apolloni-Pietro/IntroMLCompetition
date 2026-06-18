"""
rerank_tune.py — Grid-search tuning for k-reciprocal re-ranking.

Runs a grid search over (k1, k2, lam) on the validation split and prints
the best parameter combination according to top-1 retrieval accuracy.

Usage:
  python rerank_tune.py --checkpoint ./checkpoints/best_model.pth \
        --data_dir /path/to/train_dataset --batch_size 256
"""

import argparse
import itertools
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from dataset import CelebRetrievalDataset, get_val_transforms
from model import CLIPArcFaceModel
from rerank import rerank_topk


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> CLIPArcFaceModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes")
    model = CLIPArcFaceModel(num_classes=num_classes, unfreeze_blocks=6)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, dataset_subset: Subset, device: torch.device,
                       batch_size: int = 256):
    loader = DataLoader(dataset_subset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    all_embeddings = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        emb = model.encode(images)
        all_embeddings.append(emb.float().cpu())
        all_labels.append(labels)
    if len(all_embeddings) == 0:
        return torch.empty(0, model.EMBED_DIM), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def build_query_gallery(dataset: CelebRetrievalDataset, val_indices: list[int]):
    # Group indices by label within the validation subset
    per_label = defaultdict(list)
    for idx in val_indices:
        _, lbl = dataset.samples[idx]
        per_label[lbl].append(idx)

    query_idxs = []
    gallery_idxs = []
    for lbl, idxs in per_label.items():
        if len(idxs) < 2:
            # Skip identities with only one image in val set
            continue
        # deterministically pick the first as query, rest as gallery
        sorted_idxs = sorted(idxs)
        query_idxs.append(sorted_idxs[0])
        gallery_idxs.extend(sorted_idxs[1:])

    return query_idxs, gallery_idxs


def evaluate_rerank_grid(model, dataset_root, batch_size, device,
                         k1_grid, k2_grid, lam_grid, top_k=10):
    # Load full dataset (for index -> label mapping)
    index_dataset = CelebRetrievalDataset(dataset_root, transform=None)

    # 90/10 train-val split deterministic
    n_val = max(1, int(0.10 * len(index_dataset)))
    n_train = len(index_dataset) - n_val
    train_split, val_split = random_split(
        index_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_indices = val_split.indices

    # Build query/gallery indices from val split
    query_idxs, gallery_idxs = build_query_gallery(index_dataset, val_indices)
    if len(query_idxs) == 0 or len(gallery_idxs) == 0:
        raise RuntimeError("Not enough images per-identity in validation split to evaluate retrieval.")

    val_transform = get_val_transforms(image_size=224)
    eval_dataset = CelebRetrievalDataset(dataset_root, transform=val_transform)

    query_subset = Subset(eval_dataset, query_idxs)
    gallery_subset = Subset(eval_dataset, gallery_idxs)

    print(f"[Tune] Queries: {len(query_subset)}  Gallery: {len(gallery_subset)}")

    q_emb, q_labels = extract_embeddings(model, query_subset, device, batch_size)
    g_emb, g_labels = extract_embeddings(model, gallery_subset, device, batch_size)

    # baseline (no rerank) top-1
    sim = torch.mm(q_emb, g_emb.T)
    _, top_idx = torch.topk(sim, k=1, dim=1)
    top_idx = top_idx[:, 0]
    baseline_acc = (g_labels[top_idx] == q_labels).float().mean().item()
    print(f"[Tune] Baseline (no rerank) top-1: {baseline_acc:.4f}")

    best = None
    results = []
    for k1, k2, lam in itertools.product(k1_grid, k2_grid, lam_grid):
        # rerank_topk expects torch tensors
        topk_idx = rerank_topk(q_emb, g_emb, top_k=1, k1=k1, k2=k2, lam=lam)
        preds = torch.tensor([g_labels[idx[0]] for idx in topk_idx], dtype=torch.long)
        acc = (preds == q_labels).float().mean().item()
        results.append(((k1, k2, lam), acc))
        print(f"[Grid] k1={k1:2d} k2={k2:2d} lam={lam:.2f}  top-1: {acc:.4f}")
        if best is None or acc > best[1]:
            best = ((k1, k2, lam), acc)

    print(f"\n[Result] Best params: k1={best[0][0]}, k2={best[0][1]}, lam={best[0][2]}  top-1={best[1]:.4f}")
    return best, baseline_acc, results


def parse_args():
    p = argparse.ArgumentParser(description="Tune k-reciprocal re-ranking on validation split")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--k1", nargs="+", type=int, default=[15, 20, 25])
    p.add_argument("--k2", nargs="+", type=int, default=[4, 6, 8])
    p.add_argument("--lam", nargs="+", type=float, default=[0.2, 0.3, 0.4])
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"[Tune] Using device: {device}")
    model = load_model(args.checkpoint, device)
    best, baseline, all_results = evaluate_rerank_grid(
        model, args.data_dir, args.batch_size, device,
        k1_grid=args.k1, k2_grid=args.k2, lam_grid=args.lam,
    )


if __name__ == "__main__":
    main()
