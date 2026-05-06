"""
scripts/cache_threshold_viz.py

reads scripts/cache_viz_state.pkl and produces two plots side by side. One is a scatter plot of the query embeddings in the PCA space, 
the other is a heatmap of the thresholds for each region.
"""

import pathlib
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

VIZ_PATH = pathlib.Path("scripts/cache_viz_state.pkl")
OUT_PATH = pathlib.Path("scripts/cache_threshold_viz.png")


def project(embeddings, components, mean):
    """Apply the cache's saved PCA to the test queries from the benchmark"""
    return (embeddings - mean) @ components.T

def region_key(reduced_row, bucket_boundaries, d_reduced, n_buckets):
    """figure out which region the query belongs to"""
    parts = []
    for dim in range(d_reduced):
        bounds = bucket_boundaries[dim]
        b = int(np.searchsorted(bounds, reduced_row[dim]))
        if b >= n_buckets:
            b = n_buckets - 1
        parts.append(str(b))
    return ",".join(parts)

def pick_k(thresholds, preferred=10):
    """pick the k value that is closest to the preferred value"""
    ks = sorted({k for (k, _) in thresholds.keys()})
    if not ks:
        return None
    return preferred if preferred in ks else ks[-1]

def main():
    with open(VIZ_PATH, "rb") as f:
        state = pickle.load(f)

    embs = np.asarray(state["query_embeddings"], dtype="float32")
    topics = np.asarray(state["topics"])
    thresholds = state["thresholds"]
    components = state["pca_components"]
    mean = state["pca_mean"]
    boundaries = state["bucket_boundaries"]
    d_reduced = state["d_reduced"]
    n_buckets = state["n_buckets"]
    reduced = project(embs, components, mean)
    coords2d = reduced[:, :2]

    target_k = pick_k(thresholds, preferred=10)
    per_query_thresh = np.full(len(embs), np.nan)
    if target_k is not None:
        for i, row in enumerate(reduced):
            r = region_key(row, boundaries, d_reduced, n_buckets)
            v = thresholds.get((target_k, r))
            if v is not None:
                per_query_thresh[i] = v

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left plot
    ax = axes[0]
    palette = {"ACID": "tab:blue", "SERIAL": "tab:orange"}
    for topic, color in palette.items():
        m = topics == topic
        if not m.any():
            continue
        ax.scatter(
            coords2d[m, 0], coords2d[m, 1],
            c=color, label=topic, s=80, alpha=0.8, edgecolor="black",
        )
    ax.set_title("Query embeddings in cache PCA space (PC1 vs PC2)\ncolored by topic")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.2)

    # right plot
    ax = axes[1]
    valid = ~np.isnan(per_query_thresh)
    if valid.any():
        sc = ax.scatter(
            coords2d[valid, 0], coords2d[valid, 1],
            c=per_query_thresh[valid], cmap="viridis",
            s=80, edgecolor="black",
        )
        plt.colorbar(sc, ax=ax, label=f"learned L2 threshold (k={target_k})")
        if (~valid).any():
            ax.scatter(
                coords2d[~valid, 0], coords2d[~valid, 1],
                facecolors="none", edgecolor="red", s=80,
                label="no threshold for region",
            )
            ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "no thresholds learned", ha="center",
                va="center", transform=ax.transAxes)

    # draw the boundaries of the principle components
    for x in boundaries[0]:
        ax.axvline(float(x), color="grey", linestyle="--", alpha=0.5)
    if d_reduced >= 2:
        for y in boundaries[1]:
            ax.axhline(float(y), color="grey", linestyle="--", alpha=0.5)

    ax.set_title(f"Same projection, colored by region threshold\n"
                 f"(d_reduced={d_reduced}, n_buckets={n_buckets})")
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # number of regions with the target k value
    n_regions = len({r for (k, r) in thresholds.keys() if k == target_k})
    fig.suptitle(f"Cache thresholds — {len(embs)} queries, {n_regions} active regions @ k={target_k}", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130)

if __name__ == "__main__":
    main()
