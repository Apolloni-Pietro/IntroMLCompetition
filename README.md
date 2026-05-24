# Celebrity Retrieval Across Domains — Action Plan

## 1. Problem Analysis

### Core Challenge

Cross-domain image retrieval: **natural photo queries → synthetic gallery images**.
This is harder than standard face recognition because:

- Domain gap: real textures/lighting vs synthetic generation artifacts
- No fixed identity set at test time (retrieval, not classification)
- Evaluation rewards ranking quality (top-1, top-5, top-10)

### Why ResNet50 (baseline) Is Weak Here

- ImageNet features capture object-level semantics, not face identity
- No fine-tuning → no adaptation to celebrity identities
- No domain-gap handling
- Expected accuracy: ~20–35% top-1

---

## 2. Proposed Solution: CLIP ViT-B/16 + ArcFace Fine-Tuning

### Model Choice: OpenCLIP ViT-B/16

**Why CLIP over pure CNN or pure face models?**

| Property                 | ResNet50 | ArcFace (MS1M) | CLIP ViT-B/16 (ours)   |
| ------------------------ | -------- | -------------- | ---------------------- |
| Pre-training data        | ImageNet | 5M face images | 400M image–text pairs  |
| Domain robustness        | Low      | Medium         | **High**               |
| Handles synthetic images | No       | Partly         | **Yes**                |
| Fine-tuning potential    | Medium   | Good           | **Excellent**          |
| Embedding quality        | General  | Face-specific  | **General + identity** |

CLIP was trained on internet-scale data that includes AI-generated content, artistic portraits,
and illustrations — exactly the domain of the synthetic gallery. Its ViT architecture also has
stronger global attention than CNNs, which is critical for identity matching when style/texture
differ.

### Fine-Tuning Strategy

**Objective**: Add ArcFace loss on top of frozen-then-gradually-unfrozen CLIP visual encoder.

```
CLIP ViT-B/16 Visual Encoder (partially unfrozen)
        ↓
  L2-normalized 512-dim embedding
        ↓
  ArcFace Classification Head (512 → N_identities)
        ↓
  ArcFace loss (s=64, m=0.5) + Cross-Entropy
```

**What we unfreeze** (to stay within time budget):

- Final 6 transformer blocks (out of 12)
- Final LayerNorm
- Projection layer

This gives ~40M trainable parameters vs. 86M total — fast training but deep adaptation.

---

## 3. Training Recipe

### Data Augmentation (domain-gap-aware)

```
Training transforms:
  - RandomResizedCrop(224, scale=(0.65, 1.0))
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
  - RandomGrayscale(p=0.1)        # bridges color domain gap
  - RandomApply(GaussianBlur)     # simulates generation blur
  - Normalize(CLIP mean/std)
```

### Hyperparameters

| Parameter       | Value                         | Rationale                       |
| --------------- | ----------------------------- | ------------------------------- |
| Backbone LR     | 1e-5                          | Small to preserve CLIP features |
| Head LR         | 1e-4                          | Larger for ArcFace head         |
| Batch size      | 128                           | Fills V100 16GB comfortably     |
| Epochs          | 50                            | ~1.5h on V100 with 5K images    |
| Scheduler       | Cosine with warmup (5 epochs) | Standard for transformers       |
| ArcFace s       | 64.0                          | Standard scale                  |
| ArcFace m       | 0.50                          | Standard margin                 |
| Weight decay    | 1e-4                          | Regularization                  |
| Mixed precision | FP16 (torch.cuda.amp)         | Speed + memory                  |

### Estimated Training Time on V100 16GB

- ~5,000 images / batch 128 = ~39 steps/epoch
- 50 epochs × 39 steps = ~1,950 steps
- ~1.5–2 hours total ✅ (well within 5h budget)

This leaves room to:

- Train ViT-L/14 variant (~3h additional)
- Experiment with re-ranking hyperparameters

---

## 4. Inference Pipeline

```
Query images              Gallery images
     ↓                          ↓
  CLIP encoder            CLIP encoder
     ↓                          ↓
  L2 normalize            L2 normalize
     ↓                          ↓
         Cosine similarity matrix
                  ↓
         k-Reciprocal Re-Ranking
                  ↓
         Top-10 ranked gallery list
                  ↓
              Submit
```

### Re-Ranking: k-Reciprocal Encoding (Zhong et al., 2017)

This post-processing step is crucial and often gives **+5–10% absolute** improvement
on retrieval benchmarks with zero additional training.

The key idea: two images are more likely to be the same person if they appear in each
other's k-nearest-neighbor lists (mutual nearest neighbors).

**Expected gain**: top-1 accuracy +6–12% over raw cosine similarity.

---

## 5. Expected Performance

| Method                             | Est. Top-1  | Est. Top-5  | Est. Top-10 |
| ---------------------------------- | ----------- | ----------- | ----------- |
| ResNet50 (baseline)                | ~25%        | ~45%        | ~55%        |
| CLIP ViT-B/16 (zero-shot)          | ~45%        | ~65%        | ~72%        |
| **CLIP ViT-B/16 + ArcFace (ours)** | ~70–78%     | ~87–92%     | ~93–96%     |
| + Re-ranking                       | **~76–84%** | **~90–95%** | **~95–98%** |

Score estimate (out of 1000):

- Top-1 ~80% → 480 pts, Top-5 ~92% → 276 pts, Top-10 ~96% → 96 pts
- **Total ≈ 852/1000**

---

## 6. File Structure

```
competition/
├── ACTION_PLAN.md          ← this file
├── dataset.py              ← Dataset class + augmentations
├── model.py                ← CLIP + ArcFace architecture
├── train.py                ← Training loop
├── inference.py            ← Embedding extraction + retrieval
├── rerank.py               ← k-Reciprocal re-ranking
└── submit.py               ← Full inference + submission pipeline
```

---

## 7. Step-by-Step Execution Guide

```bash
# 1. Install dependencies
pip install open_clip_torch timm tqdm Pillow requests

# 2. Train the model
python train.py \
  --data_dir /path/to/train \
  --output_dir ./checkpoints \
  --epochs 50 \
  --batch_size 128

# 3. Run inference and submit
python submit.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data_dir /path/to/test_data \
  --groupname "YourTeamName" \
  --url http://localhost:3001/retrieval/ \
  --rerank  # enable k-reciprocal re-ranking
```

---

## 8. Possible Further Improvements (if budget allows)

1. **ViT-L/14 backbone**: larger model, +3–5% accuracy, ~3× slower training
2. **Curriculum hard-negative mining**: mine hardest negatives within batch for ArcFace
3. **Test-Time Augmentation (TTA)**: average embeddings from multiple augmented views
4. **Domain Adversarial Training**: add a domain discriminator loss to explicitly bridge real/synthetic gap
5. **Ensemble**: average embeddings from CLIP ViT-B/16 + a fine-tuned InsightFace model
6. **Query Expansion (DBA/AQE)**: expand query embedding with weighted average of top-k neighbors
