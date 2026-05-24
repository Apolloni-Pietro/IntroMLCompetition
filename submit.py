"""
submit.py — End-to-end inference and submission script.

Usage
-----
# With re-ranking (recommended):
python submit.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir   /path/to/test_data \
    --groupname  "Team AAA" \
    --url        http://videosim.disi.unitn.it:3001/retrieval/ \
    --rerank

# Without re-ranking (faster, slightly lower accuracy):
python submit.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir   /path/to/test_data \
    --groupname  "Team AAA" \
    --url        http://videosim.disi.unitn.it:3001/retrieval//
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import requests

from dataset import FolderImageDataset, get_val_transforms
from model import CLIPArcFaceModel
from rerank import rerank_topk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> CLIPArcFaceModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt["num_classes"]
    model = CLIPArcFaceModel(num_classes=num_classes, unfreeze_blocks=6)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[Load] Checkpoint loaded from {checkpoint_path}  "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_acc {ckpt.get('val_acc', 0):.4f})")
    return model


@torch.no_grad()
def extract_embeddings(
    model: CLIPArcFaceModel,
    folder: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """
    Extract L2-normalised embeddings for all images in `folder`.
    Returns (embeddings [N, D], filenames [N]).
    """
    dataset = FolderImageDataset(folder, transform=get_val_transforms(224))
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=6, pin_memory=True,
    )

    all_embeddings = []
    all_filenames  = []

    for images, fnames in tqdm(loader, desc=f"Embedding {os.path.basename(folder)}"):
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type):
            emb = model.encode(images)
        all_embeddings.append(emb.float().cpu())
        all_filenames.extend(fnames)

    return torch.cat(all_embeddings, dim=0), all_filenames


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit(results: dict, groupname: str, url: str) -> None:
    payload = json.dumps({"groupname": groupname, "images": results})
    print(f"\n[Submit] Sending results to {url} as group '{groupname}' ...")
    response = requests.post(url, payload)
    try:
        result = json.loads(response.text)
        print(f"[Submit] Server response: {result}")
        if "accuracy" in result:
            print(f"\n  ✓ Accuracy: {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"[Submit] ERROR: {response.text}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    device = get_device()
    print(f"[Submit] Using device: {device}\n")

    # Load fine-tuned model
    model = load_model(args.checkpoint, device)

    # Paths
    query_folder   = os.path.join(args.data_dir, "query")
    gallery_folder = os.path.join(args.data_dir, "gallery")

    # Extract embeddings
    print()
    query_embeddings,   query_filenames   = extract_embeddings(
        model, query_folder,   args.batch_size, device)
    gallery_embeddings, gallery_filenames = extract_embeddings(
        model, gallery_folder, args.batch_size, device)

    print(f"\n[Embed] Query:   {query_embeddings.shape}")
    print(f"[Embed] Gallery: {gallery_embeddings.shape}")

    # --- Retrieval ---
    top_k = 10

    if args.rerank:
        print("\n[Rerank] Running k-Reciprocal re-ranking "
              f"(k1={args.k1}, k2={args.k2}, lambda={args.lam}) ...")
        top_k_indices = rerank_topk(
            query_embeddings, gallery_embeddings,
            top_k=top_k, k1=args.k1, k2=args.k2, lam=args.lam,
        )
    else:
        print("\n[Retrieval] Computing cosine similarity ...")
        sim = torch.mm(query_embeddings, gallery_embeddings.T)  # (Nq, Ng)
        _, top_k_indices_t = torch.topk(sim, k=top_k, dim=1)
        top_k_indices = top_k_indices_t.numpy()

    # Build results dict  {query_filename: [gallery_filename, ...]}
    results = {}
    for i, qfname in enumerate(query_filenames):
        results[qfname] = [gallery_filenames[idx] for idx in top_k_indices[i]]

    # --- Sanity check ---
    assert all(len(v) == top_k for v in results.values()), \
        "Each query must have exactly 10 gallery matches!"
    print(f"\n[Results] {len(results)} queries, each with {top_k} matches.")

    # --- Print sample ---
    sample_query = query_filenames[0]
    print(f"\n  Sample — Query: {sample_query}")
    for rank, gfname in enumerate(results[sample_query], 1):
        print(f"    #{rank}: {gfname}")

    # --- Submit ---
    submit(results, args.groupname, args.url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inference + submission pipeline")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to best_model.pth from train.py.")
    p.add_argument("--data_dir",    type=str, required=True,
                   help="Test data root (must contain query/ and gallery/ sub-dirs).")
    p.add_argument("--groupname",   type=str, default="CLIP-ArcFace")
    p.add_argument("--url",         type=str, default="http://localhost:3001/retrieval/")
    p.add_argument("--batch_size",  type=int, default=256)
    # Re-ranking options
    p.add_argument("--rerank",      action="store_true",
                   help="Enable k-Reciprocal re-ranking (recommended).")
    p.add_argument("--k1",          type=int,   default=20)
    p.add_argument("--k2",          type=int,   default=6)
    p.add_argument("--lam",         type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())