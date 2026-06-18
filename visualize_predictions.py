"""
Visualize model retrieval results.

For each randomly sampled query image, retrieves the top-5 most similar
gallery images and displays them in a grid with success/failure indicators.
"""

import os
import random
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from model import CLIPArcFaceModel
from dataset import CelebRetrievalDataset, get_val_transforms


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint['num_classes']
    model = CLIPArcFaceModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_embeddings.append(model.encode(imgs.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def tensor_to_rgb(t):
    img = (t * _CLIP_STD + _CLIP_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def add_border(ax, color, lw=6):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize(model, dataset, num_queries, output_dir, device, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting gallery embeddings...")
    embeddings, labels = extract_embeddings(model, dataset, device)

    # Random query indices (unique)
    rng = random.Random(seed)
    query_indices = rng.sample(range(len(dataset)), k=min(num_queries, len(dataset)))

    N_COLS = 6  # 1 query + 5 retrieved
    N_ROWS = len(query_indices)
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.8, N_ROWS * 3.2), squeeze=False)

    total_retrieved = 0
    total_correct   = 0

    for row, q_idx in enumerate(query_indices):
        q_tensor, q_label = dataset[q_idx]
        q_embed  = embeddings[q_idx]          # already L2-normalised
        q_name   = dataset.classes[q_label]

        # Cosine similarity against full gallery; mask out the query itself
        sims = embeddings @ q_embed            # (N,)
        sims[q_idx] = -1.0                    # exclude self

        top5 = torch.topk(sims, k=5).indices.tolist()

        # --- Query column ---
        ax = axes[row][0]
        ax.imshow(tensor_to_rgb(q_tensor))
        ax.set_title(f"QUERY\n{q_name}", fontsize=9, fontweight='bold', pad=4)
        ax.axis('off')
        # Neutral grey border on query
        add_border(ax, color='#888888', lw=4)

        # --- Retrieved columns ---
        for col, g_idx in enumerate(top5, start=1):
            g_tensor, g_label = dataset[g_idx]
            g_name   = dataset.classes[g_label]
            sim_val  = sims[g_idx].item()
            success  = (g_label == q_label)

            total_retrieved += 1
            total_correct   += int(success)

            mark  = '✓' if success else '✗'
            color = '#2ecc40' if success else '#e74c3c'   # green / red

            ax = axes[row][col]
            ax.imshow(tensor_to_rgb(g_tensor))
            ax.set_title(f"Match {col} {mark}\nSim: {sim_val:.2f}",
                         fontsize=9, color=color, fontweight='bold', pad=4)
            ax.axis('off')
            add_border(ax, color=color, lw=6)

    overall = total_correct / total_retrieved * 100 if total_retrieved else 0
    fig.suptitle(
        f"Top-5 Retrieval  —  {total_correct}/{total_retrieved} correct  ({overall:.1f}%)",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")
    print(f"Top-5 correct: {total_correct}/{total_retrieved} ({overall:.1f}%)")

    stats = {'top5_correct': total_correct, 'top5_total': total_retrieved,
             'top5_accuracy': overall}
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model retrieval predictions')
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--data_dir',     type=str, required=True)
    parser.add_argument('--num_queries',  type=int, default=5,
                        help='Number of query rows to show')
    parser.add_argument('--output_dir',   type=str, default='./visualizations')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--device',       type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    dataset = CelebRetrievalDataset(args.data_dir, transform=get_val_transforms(224))
    model   = load_checkpoint(args.checkpoint, device=args.device)
    visualize(model, dataset, args.num_queries, args.output_dir, args.device, args.seed)
