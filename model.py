"""
model.py — CLIP ViT-B/16 visual encoder fine-tuned with an ArcFace head.

Architecture overview
---------------------
  OpenCLIP ViT-B/16 visual encoder (partially unfrozen)
      └─ 512-dim L2-normalised embedding
           └─ ArcFaceLoss head  (used only during training)

At inference time only the encoder is used; the ArcFace head is discarded.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ---------------------------------------------------------------------------
# ArcFace loss
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace / InsightFace).

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
    Face Recognition", CVPR 2019.

    Args:
        in_features:   Dimensionality of the input embedding (512 for ViT-B/16).
        num_classes:   Number of identity classes in the training set.
        s:             Feature scale (default 64). Controls the magnitude of
                       logits fed to cross-entropy.
        m:             Angular margin in radians (default 0.50 ≈ 28.6°).
    """

    def __init__(self, in_features: int, num_classes: int,
                 s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants for the margin calculation
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)   # threshold for numerical stability
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) — L2-normalised feature vectors.
            labels:     (B,)   — integer class indices.
        Returns:
            Scalar cross-entropy loss.
        """
        # cosine similarity between embeddings and class prototypes
        cosine = F.linear(F.normalize(embeddings, dim=1),
                          F.normalize(self.weight, dim=1))   # (B, C)

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-6))
        # cos(θ + m) = cos θ cos m − sin θ sin m
        phi = cosine * self.cos_m - sine * self.sin_m
        # Numerical stability: if cos θ < cos(π - m), use the linear approximation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Replace ground-truth logit with margin-penalised version
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s

        return F.cross_entropy(output, labels)


# ---------------------------------------------------------------------------
# Main model wrapper
# ---------------------------------------------------------------------------

class CLIPArcFaceModel(nn.Module):
    """
    CLIP ViT-B/16 visual encoder wrapped for metric learning.

    Unfreezes the last `unfreeze_blocks` transformer blocks plus the final
    LayerNorm and projection layer, keeping earlier blocks frozen to preserve
    general visual representations while adapting to celebrity identity.
    """

    CLIP_MODEL_NAME = "ViT-L-14"
    CLIP_PRETRAINED  = "openai"          # OpenAI's CLIP weights
    EMBED_DIM        = 768               # ViT-L-14 output dimension

    def __init__(self, num_classes: int, unfreeze_blocks: int = 6):
        super().__init__()

        # Load CLIP visual backbone
        clip_model, _, _ = open_clip.create_model_and_transforms(
            self.CLIP_MODEL_NAME, pretrained=self.CLIP_PRETRAINED
        )
        self.encoder = clip_model.visual   # keep only the visual tower

        # Freeze everything first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Selectively unfreeze the last N transformer blocks
        self._unfreeze_last_blocks(unfreeze_blocks)

        # ArcFace classification head
        self.arcface = ArcFaceLoss(
            in_features=self.EMBED_DIM,
            num_classes=num_classes,
            s=64.0,
            m=0.50,
        )

    # ------------------------------------------------------------------
    # Selective unfreezing
    # ------------------------------------------------------------------

    def _unfreeze_last_blocks(self, n: int) -> None:
        """
        Unfreeze the last `n` transformer blocks and the final LayerNorm /
        projection. Works for OpenCLIP's ViT visual encoder.
        """
        # The transformer blocks live at encoder.transformer.resblocks
        blocks = list(self.encoder.transformer.resblocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze final ln_post and proj
        for name in ("ln_post", "proj"):
            module = getattr(self.encoder, name, None)
            if module is None:
                continue
            if isinstance(module, nn.Parameter):
                module.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Model] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised embeddings.  Used at both train and inference time.
        """
        feats = self.encoder(x)            # raw features from CLIP visual encoder
        return F.normalize(feats, dim=1)   # L2 normalise → unit hypersphere

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """
        Training forward pass.  Returns (loss, embeddings).
        If labels is None, returns embeddings only (inference mode).
        """
        embeddings = self.encode(x)
        if labels is None:
            return embeddings
        loss = self.arcface(embeddings, labels)
        return loss, embeddings


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def build_model(num_classes: int,
                unfreeze_blocks: int = 6,
                device: str = "cuda") -> CLIPArcFaceModel:
    model = CLIPArcFaceModel(num_classes=num_classes,
                             unfreeze_blocks=unfreeze_blocks)
    return model.to(device)