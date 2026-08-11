"""Cross-scene fingerprint clustering.

Inputs: a list of v2 diagnosis-report JSONs (each contains a 13-D
``normalized_sensitivities_S_hat`` flat vector under ``s_named_flat``).

Outputs (under output_dir):
    - fingerprint_summary.json
    - figs/fingerprint_pca.png       (2D PCA scatter)
    - figs/fingerprint_dendrogram.png  (scipy linkage)
    - figs/fingerprint_cluster_heatmap.png  (per-cluster mean S_hat)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .semantics import SEMANTIC_VAR_NAMES, TOPO_DIMS


def _load_vec(report: Dict[str, Any]) -> Optional[np.ndarray]:
    flat = report.get("s_named_flat")
    if flat is not None:
        return np.array([float(flat[n]) for n in SEMANTIC_VAR_NAMES], dtype=np.float64)
    return None


def cluster_fingerprints(reports: Sequence[Dict[str, Any]],
                         k: int = 3,
                         output_dir: Optional[Path] = None) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vecs, scenes = [], []
    for r in reports:
        v = _load_vec(r)
        if v is None: continue
        vecs.append(v)
        scenes.append(r.get("metadata", {}).get("scene_id", "?"))
    if len(vecs) < 2:
        return {"error": "need >= 2 valid reports", "n": len(vecs)}

    M = np.stack(vecs, axis=0)        # (N, 13)
    # standardise each dim
    mu = M.mean(0); sd = M.std(0) + 1e-9
    Z = (M - mu) / sd

    # KMeans
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        k = max(2, min(k, len(vecs) - 1))
        km = KMeans(n_clusters=k, n_init=8, random_state=0)
        labels = km.fit_predict(Z)
        pca = PCA(n_components=2).fit_transform(Z)
        centers = km.cluster_centers_
    except Exception as e:
        labels = np.zeros(len(vecs), dtype=int)
        pca = np.column_stack([Z.mean(1), Z.std(1)])
        centers = np.zeros((1, M.shape[1]))

    summary: Dict[str, Any] = {
        "n_scenes": len(vecs), "k": int(k),
        "scenes": scenes,
        "labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
        "feature_names": list(SEMANTIC_VAR_NAMES),
        "feature_matrix": M.tolist(),
        "pca": pca.tolist() if hasattr(pca, "tolist") else list(pca),
    }

    if output_dir is not None:
        out_dir = Path(output_dir); fig_dir = out_dir / "figs"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # --- PCA scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("tab10")
        for i, (xy, s, lab) in enumerate(zip(pca, scenes, labels)):
            ax.scatter(xy[0], xy[1], color=cmap(int(lab) % 10), s=70)
            ax.annotate(s, (xy[0], xy[1]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title("Cross-scene fingerprint (PCA-2D, KMeans)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "fingerprint_pca.png", dpi=110)
        plt.close(fig)

        # --- dendrogram
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            link = linkage(Z, method="ward")
            fig, ax = plt.subplots(figsize=(max(6, len(scenes) * 0.5), 4))
            dendrogram(link, labels=scenes, ax=ax, leaf_rotation=30)
            ax.set_title("Hierarchical clustering of scene fingerprints (Ward)")
            fig.tight_layout(); fig.savefig(fig_dir / "fingerprint_dendrogram.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            summary["dendrogram_err"] = str(e)

        # --- per-cluster mean heatmap
        try:
            uniq = sorted(set(labels.tolist()) if hasattr(labels, "tolist") else set(labels))
            mat = np.stack([M[np.asarray(labels) == u].mean(0) for u in uniq], axis=0)
            fig, ax = plt.subplots(figsize=(10, 1.0 + 0.6 * len(uniq)))
            im = ax.imshow(mat, aspect="auto", cmap="magma")
            ax.set_yticks(range(len(uniq))); ax.set_yticklabels([f"cluster {u}" for u in uniq])
            ax.set_xticks(range(len(SEMANTIC_VAR_NAMES)))
            ax.set_xticklabels(SEMANTIC_VAR_NAMES, rotation=40, ha="right", fontsize=8)
            ax.set_title("Per-cluster mean sensitivity (S_hat)")
            fig.colorbar(im, ax=ax, fraction=0.04)
            fig.tight_layout()
            fig.savefig(fig_dir / "fingerprint_cluster_heatmap.png", dpi=110)
            plt.close(fig)
        except Exception as e:
            summary["heatmap_err"] = str(e)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "fingerprint_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    return summary
