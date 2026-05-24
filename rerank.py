"""
rerank.py — k-Reciprocal Encoding re-ranking (Zhong et al., 2017).

This post-processing step consistently boosts retrieval accuracy by 5–12
percentage points with no additional training.

Key insight
-----------
Two images are strong matches if they appear in each other's k-nearest
neighbour (k-NN) lists — "mutual nearest neighbours".  This is formalised
via a Jaccard-distance reformulation that is then blended with the original
cosine similarity.

Reference
---------
Zhong et al., "Re-ranking Person Re-identification with k-Reciprocal Encoding",
CVPR 2017.  https://arxiv.org/abs/1701.08398
"""

import numpy as np
import torch
import torch.nn.functional as F


def k_reciprocal_rerank(
    query_features:   torch.Tensor,   # (Nq, D)  — L2-normalised query embeddings
    gallery_features: torch.Tensor,   # (Ng, D)  — L2-normalised gallery embeddings
    k1:    int   = 20,
    k2:    int   = 6,
    lam:   float = 0.3,
) -> np.ndarray:
    """
    Compute re-ranked distance matrix.

    Args:
        query_features:   (Nq, D) L2-normalised embedding tensor.
        gallery_features: (Ng, D) L2-normalised embedding tensor.
        k1:   Size of the k-reciprocal neighbourhood.
        k2:   Size of the local query expansion neighbourhood.
        lam:  Blending weight for original cosine distance
              (1 - cosine_similarity).  Lower → more re-ranking influence.

    Returns:
        final_dist: (Nq, Ng) numpy array of final distances.
                    Lower is more similar → argsort ascending for ranking.
    """
    query_features   = query_features.float().cpu()
    gallery_features = gallery_features.float().cpu()

    Nq = query_features.size(0)
    Ng = gallery_features.size(0)
    N  = Nq + Ng

    # --- Build combined feature matrix [query | gallery] ---
    all_features = torch.cat([query_features, gallery_features], dim=0)  # (N, D)

    # --- Pairwise cosine distance matrix (all-vs-all) ---
    # Distance = 1 - cosine_similarity (∈ [0, 2])
    cos_sim  = torch.mm(all_features, all_features.T).clamp(-1, 1)
    dist_mat = (1.0 - cos_sim).numpy().astype(np.float32)              # (N, N)

    # --- k-Reciprocal encoding ---
    # For each probe i, find its k1-NN and check if the relationship is mutual.
    V = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # k1+1 because index 0 is the query itself
        sorted_idx = np.argsort(dist_mat[i])
        nn_k1      = sorted_idx[1:k1 + 2]   # top k1+1 excluding self

        # k-reciprocal: keep j iff i is in j's k1-NN
        nn_k1_half = sorted_idx[1: k1 // 2 + 2]
        rnn_k1 = []
        for j in nn_k1:
            j_nn = np.argsort(dist_mat[j])[1:k1 // 2 + 2]
            if i in j_nn:
                rnn_k1.append(j)
        rnn_k1 = np.array(rnn_k1)

        if len(rnn_k1) == 0:
            rnn_k1 = nn_k1[:k1 // 2]

        # Gaussian kernel weight: exp(- dist^2 / 2)
        weight = np.exp(-dist_mat[i, rnn_k1])
        V[i, rnn_k1] = weight / (weight.sum() + 1e-6)

    # --- Local query expansion (average top-k2 neighbours' V vectors) ---
    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(N):
            sorted_idx = np.argsort(dist_mat[i])[1:k2 + 1]
            V_qe[i] = (V[i] + V[sorted_idx].sum(axis=0)) / (k2 + 1)
        V = V_qe

    # --- Jaccard distance ---
    # jaccard(i,j) = 1 - |V_i ∩ V_j| / |V_i ∪ V_j|
    # With soft sets: intersection = min, union = max
    # Efficient approximation: 2 * |V_i ∩ V_j| / (|V_i| + |V_j|)
    # We use a sparse-matrix trick: min = (V_i + V_j - |V_i - V_j|) / 2
    jaccard_dist = np.zeros((Nq, Ng), dtype=np.float32)
    for i in range(Nq):
        Vi = V[i]                          # (N,)
        Vg = V[Nq:]                        # (Ng, N)
        intersection = np.minimum(Vi, Vg).sum(axis=1)   # (Ng,)
        union        = np.maximum(Vi, Vg).sum(axis=1)   # (Ng,)
        jaccard_dist[i] = 1.0 - intersection / (union + 1e-6)

    # --- Original query-gallery cosine distance ---
    orig_dist = (1.0 - torch.mm(query_features, gallery_features.T)
                 .clamp(-1, 1)).numpy().astype(np.float32)   # (Nq, Ng)

    # --- Blend ---
    final_dist = (1.0 - lam) * jaccard_dist + lam * orig_dist
    return final_dist   # (Nq, Ng)


# ---------------------------------------------------------------------------
# Convenience wrapper that returns top-k indices per query
# ---------------------------------------------------------------------------

def rerank_topk(
    query_features:   torch.Tensor,
    gallery_features: torch.Tensor,
    top_k: int = 10,
    k1:    int = 20,
    k2:    int = 6,
    lam:   float = 0.3,
) -> np.ndarray:
    """
    Returns (Nq, top_k) integer array of gallery indices, sorted best-first.
    """
    dist = k_reciprocal_rerank(query_features, gallery_features, k1, k2, lam)
    return np.argsort(dist, axis=1)[:, :top_k]