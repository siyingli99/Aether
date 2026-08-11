#!/usr/bin/env python3
"""Function: Cluster AETHER-Diag reports by normalized sensitivity vectors.
Input/Output: Reads report JSON files matched by a recursive glob and writes cluster summaries and plots.
Example:
    python scripts/cluster_diagnoses.py \
        --glob 'outputs/diagnosis/**/report.json' --k 3 \
        --out_dir outputs/diagnosis_cluster
"""
import argparse, json, os, glob, sys
import numpy as np

DIM_KEYS = [
    "delta_s_lon","delta_d_lat","delta_yaw","scale","delta_v_lon","omega_yaw",
    "brightness","contrast","saturation","hue_rotate","opacity_fade","sh_jitter","fog",
]

def load(reports_glob):
    rows, meta = [], []
    for p in sorted(glob.glob(reports_glob, recursive=True)):
        try:
            d = json.load(open(p))
            f = d["s_named_flat"]
            rows.append([float(f[k]) for k in DIM_KEYS])
            meta.append({
                "report":   p,
                "scene":    d["metadata"]["scene_id"],
                "ad":       d["metadata"]["tested_model"],
                "G_topo":   d["macro_pathology_metrics"]["G_topo"],
                "G_photo":  d["macro_pathology_metrics"]["G_photo"],
                "r_vt":     d["macro_pathology_metrics"]["r_vt_ratio"],
                "concl":    d["macro_pathology_metrics"]["diagnosis_conclusion"],
                "conf":     d["macro_pathology_metrics"]["confidence"],
                "target":   d["metadata"]["target_object_id"],
            })
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
    return np.array(rows), meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="aether_results/diagnosis/**/report.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out_dir", default="aether_results/diagnosis_cluster")
    args = ap.parse_args()

    X, meta = load(args.glob)
    if len(X) < 2:
        print(f"need >=2 reports, got {len(X)}"); sys.exit(1)
    print(f"loaded {len(X)} reports, dim={X.shape[1]}")

    # row-normalize so we cluster by sensitivity *shape* not magnitude
    Xn = X / (X.sum(1, keepdims=True) + 1e-9)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    k = min(args.k, len(X)-1)
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xn)
    labels = km.labels_
    sil = silhouette_score(Xn, labels) if k>1 and len(set(labels))>1 else float("nan")
    pca = PCA(n_components=2).fit_transform(Xn)

    os.makedirs(args.out_dir, exist_ok=True)
    csv = os.path.join(args.out_dir, "cluster_assignments.csv")
    with open(csv, "w") as f:
        f.write("scene,ad,cluster,G_topo,G_photo,r_vt,confidence,conclusion,target,pca1,pca2\n")
        for m, c, p in zip(meta, labels, pca):
            f.write(f"{m['scene']},{m['ad']},{c},{m['G_topo']:.4f},{m['G_photo']:.4f},{m['r_vt']:.4f},{m['conf']:.4f},{m['concl']},{m['target']},{p[0]:.4f},{p[1]:.4f}\n")

    # cluster centroid sensitivity profile
    cent = np.zeros((k, len(DIM_KEYS)))
    for c in range(k):
        cent[c] = Xn[labels==c].mean(0) if (labels==c).any() else 0
    profile_csv = os.path.join(args.out_dir, "cluster_profiles.csv")
    with open(profile_csv, "w") as f:
        f.write("cluster," + ",".join(DIM_KEYS) + ",size\n")
        for c in range(k):
            f.write(f"{c}," + ",".join(f"{v:.4f}" for v in cent[c]) + f",{int((labels==c).sum())}\n")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7,6))
    cmap = plt.get_cmap("tab10")
    for i,(m,p) in enumerate(zip(meta, pca)):
        ax.scatter(p[0], p[1], color=cmap(labels[i]%10), s=80, edgecolor="k")
        tag = f"{m['scene'].replace('nuscenes_','').replace('_aether','')}/{m['ad']}"
        ax.annotate(tag, (p[0], p[1]), fontsize=7, xytext=(4,4), textcoords="offset points")
    ax.set_title(f"AETHER-Diag clustering  k={k}  silhouette={sil:.3f}")
    ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir,"cluster_scatter.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(DIM_KEYS))
    w = 0.8/k
    for c in range(k):
        ax.bar(x + c*w, cent[c], w, label=f"cluster {c}  (n={int((labels==c).sum())})", color=cmap(c%10))
    ax.set_xticks(x + w*(k-1)/2); ax.set_xticklabels(DIM_KEYS, rotation=45, ha="right")
    ax.set_ylabel("normalised sensitivity (row-sum=1)")
    ax.set_title("Cluster centroid sensitivity profile")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(args.out_dir,"cluster_profiles.png"), dpi=140); plt.close(fig)

    print(f"silhouette={sil:.3f}")
    print(f"-> {csv}")
    print(f"-> {profile_csv}")
    print(f"-> {args.out_dir}/cluster_scatter.png")
    print(f"-> {args.out_dir}/cluster_profiles.png")

if __name__ == "__main__":
    main()
