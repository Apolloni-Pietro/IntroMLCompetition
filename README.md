# Team AAA - Celebrity Retrieval Across Domains

Cross-domain image retrieval competition for the Introduction to Machine Learning course at UniTN. Given natural photo **queries**, retrieve the matching **synthetic gallery images** of the same celebrity identity.

Scoring: top-1 (×6) + top-5 (×3) + top-10 (×1), max 1000 pts.

---

## Approach

**OpenCLIP ViT-L/14** fine-tuned with **ArcFace** metric learning loss, followed by **k-Reciprocal re-ranking** at inference.

```
OpenCLIP ViT-L/14 visual encoder  (last 6 blocks unfrozen)
        ↓
  L2-normalised 768-dim embedding
        ↓  (training only)
  ArcFace head  (s=64, m=0.5)
```

At inference, only the encoder is used. Embeddings from 6 TTA views are averaged and re-normalised before retrieval.

---

## Data Layout

**Training** (`CelebRetrievalDataset`):

```
train_data/
├── identity_name_1/
│   ├── img_a.jpg
│   └── img_b.jpg
└── identity_name_2/
    └── ...
```

**Test** (`FolderImageDataset`):

```
test_data/
├── query/    ← real photos
└── gallery/  ← synthetic images
```

---

## Usage

### 1. Install dependencies

```bash
pip install open_clip_torch timm tqdm Pillow requests
```

### 2. Train

```bash
python train.py \
    --data_dir /path/to/train \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 64
```

Key training flags:
| Flag | Default | Description |
|---|---|---|
| `--lr_backbone` | 1e-5 | LR for unfrozen ViT blocks |
| `--lr_head` | 1e-4 | LR for ArcFace head |
| `--unfreeze_blocks` | 6 | Number of final transformer blocks to unfreeze |
| `--warmup_epochs` | 5 | Linear LR warmup length |

Saves `checkpoints/best_model.pth` (best val centroid-NN accuracy) and periodic `ckpt_epochXXX.pth` every 10 epochs.

### 3. (Optional) Tune re-ranking hyperparameters

```bash
python rerank_tune.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir /path/to/train \
    --batch_size 256 \
    --k1 15 20 25 \
    --k2 4 6 8 \
    --lam 0.2 0.3 0.4
```

Runs a grid search over `(k1, k2, lam)` on the validation split and prints the best combination.

### 4. Submit

```bash
python submit.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir /path/to/test_data \
    --groupname "Team AAA" \
    --url http://videosim.disi.unitn.it:3001/retrieval/ \
    --rerank \
    --k1 20 --k2 6 --lam 0.3
```

Drop `--rerank` for faster (slightly lower accuracy) cosine-similarity-only retrieval.

---

## File Overview

| File             | Purpose                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `model.py`       | `CLIPArcFaceModel` + `ArcFaceLoss`                                                            |
| `dataset.py`     | `CelebRetrievalDataset` (train) and `FolderImageDataset` (inference) + augmentation pipelines |
| `train.py`       | Training loop with FP16, cosine LR schedule, checkpointing                                    |
| `rerank.py`      | k-Reciprocal encoding re-ranking (Zhong et al., CVPR 2017)                                    |
| `rerank_tune.py` | Grid-search tuning of re-ranking hyperparameters on val split                                 |
| `submit.py`      | TTA embedding extraction + retrieval + HTTP submission                                        |
