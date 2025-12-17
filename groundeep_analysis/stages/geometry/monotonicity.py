import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

plt.switch_backend("Agg")


def compute_class_centroids(Z: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    means = []
    for c in classes:
        idx = labels == c
        if idx.sum() == 0:
            continue
        means.append(Z[idx].mean(axis=0))
    class_means = np.vstack(means) if means else np.empty((0, Z.shape[1]))
    return class_means, classes


def pairwise_centroid_distances(class_means: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if class_means.size == 0:
        return np.empty((0, 0))
    return pairwise_distances(class_means, metric=metric)


def _pairs_deltaN_and_distances(D: np.ndarray, classes: np.ndarray):
    xs, ys, ci, cj = [], [], [], []
    C = len(classes)
    for i in range(C):
        for j in range(i + 1, C):
            xs.append(abs(int(classes[i]) - int(classes[j])))
            ys.append(D[i, j])
            ci.append(int(classes[i]))
            cj.append(int(classes[j]))
    return (
        np.array(xs, dtype=float),
        np.array(ys, dtype=float),
        np.array(ci, dtype=int),
        np.array(cj, dtype=int),
    )


def plot_distance_vs_deltaN(D: np.ndarray, classes: np.ndarray, out_path: Path) -> Dict[str, float | List[Dict[str, float]]]:
    if D.size == 0 or len(classes) < 2:
        return {"spearman_r": np.nan, "p": np.nan, "pairs": 0, "outliers": []}

    x, y, c_i, c_j = _pairs_deltaN_and_distances(D, classes)

    uniq = np.unique(x).astype(int)
    medians, lo, hi = [], [], []
    rng = np.random.default_rng(12345)
    median_map = {}
    for d in uniq:
        vals = y[x == d]
        if len(vals) == 0:
            medians.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        m = np.median(vals)
        medians.append(m)
        median_map[int(d)] = m
        if len(vals) > 1:
            boots = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(200)]
            lo.append(np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5))
        else:
            lo.append(vals[0])
            hi.append(vals[0])

    rho, p = spearmanr(x, y)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="gray", alpha=0.25, s=16, label="pairs")
    plt.plot(uniq, medians, marker="o", color="tab:blue", label="median")
    for xd, l, h in zip(uniq, lo, hi):
        if np.isnan(l) or np.isnan(h):
            continue
        plt.vlines(xd, l, h, colors="tab:blue", alpha=0.8)
    plt.xlabel("ΔN (|i-j|)")
    plt.ylabel("Centroid distance")
    plt.title(f"ΔN vs distance — Spearman ρ={rho:.3f}, p={p:.2e}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    residuals = y - np.array([median_map[int(d)] for d in x])
    outlier_idx = np.argsort(residuals)[::-1]
    outliers: List[Dict[str, float]] = []
    for idx in outlier_idx[: min(10, len(outlier_idx))]:
        if residuals[idx] <= 0:
            break
        outliers.append(
            {
                "deltaN": float(x[idx]),
                "distance": float(y[idx]),
                "residual": float(residuals[idx]),
                "class_i": float(c_i[idx]),
                "class_j": float(c_j[idx]),
            }
        )

    return {
        "spearman_r": float(rho),
        "p": float(p),
        "pairs": int(len(x)),
        "outliers": outliers,
    }


def plot_violin_by_deltaN(D: np.ndarray, classes: np.ndarray, out_path: Path):
    if D.size == 0 or len(classes) < 2:
        return
    x, y, _, _ = _pairs_deltaN_and_distances(D, classes)
    df = pd.DataFrame({"deltaN": x.astype(int), "distance": y})
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="deltaN", y="distance", inner="box", cut=0)
    plt.title("Distribution of distances by ΔN")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_centroid_heatmap(D: np.ndarray, classes: np.ndarray, out_path: Path):
    if D.size == 0:
        return
    plt.figure(figsize=(7, 6))
    sns.heatmap(D, cmap="viridis", square=True, xticklabels=classes.astype(int), yticklabels=classes.astype(int))
    plt.xlabel("Numerosity")
    plt.ylabel("Numerosity")
    plt.title("Centroid distance (class×class)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ordinal_trajectory_1d(class_means: np.ndarray, classes: np.ndarray, out_path: Path) -> Dict[str, float]:
    if class_means.size == 0:
        return {"spearman_r_1d": np.nan, "p": np.nan}
    pca = PCA(n_components=1, random_state=42).fit(class_means)
    s = pca.transform(class_means).reshape(-1)
    rho_raw, _ = spearmanr(classes, s)
    if not np.isnan(rho_raw) and rho_raw < 0:
        s *= -1
    order = np.argsort(classes)
    s_ord = s[order]
    c_ord = classes[order]
    rho, p = spearmanr(c_ord, s_ord)
    plt.figure(figsize=(7, 4))
    plt.plot(c_ord, s_ord, marker="o")
    plt.xlabel("Class (numerosity)")
    plt.ylabel("PCA-1 score")
    plt.title(f"Ordinal 1D trajectory — Spearman ρ={rho:.3f}, p={p:.2e}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return {"spearman_r_1d": float(rho), "p": float(p)}


def plot_ordinal_trajectory_2d(class_means: np.ndarray, classes: np.ndarray, out_path: Path):
    if class_means.size == 0:
        return
    pca = PCA(n_components=2, random_state=42).fit(class_means)
    s2 = pca.transform(class_means)
    rho_pc1, _ = spearmanr(classes, s2[:, 0])
    if not np.isnan(rho_pc1) and rho_pc1 < 0:
        s2[:, 0] *= -1
    rho_pc2, _ = spearmanr(classes, s2[:, 1])
    if not np.isnan(rho_pc2) and rho_pc2 < 0:
        s2[:, 1] *= -1
    order = np.argsort(classes)
    c_ord = classes[order]
    s2_ord = s2[order]
    plt.figure(figsize=(7, 6))
    plt.scatter(s2[:, 0], s2[:, 1], c=classes, cmap="viridis", s=40)
    for (x, y), c in zip(s2, classes):
        plt.text(x, y, str(int(c)), fontsize=8, ha="center", va="center")
    plt.plot(s2_ord[:, 0], s2_ord[:, 1], color="gray", alpha=0.7)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Ordinal 2D trajectory (class means)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_deltaN_stats_csv(D: np.ndarray, classes: np.ndarray, out_csv: Path):
    if D.size == 0:
        return
    x, y, c_i, c_j = _pairs_deltaN_and_distances(D, classes)
    df = pd.DataFrame(
        {
            "deltaN": x.astype(int),
            "distance": y,
            "class_i": c_i,
            "class_j": c_j,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def plot_outlier_pairs(
    dataset: Any,
    centroids: np.ndarray,
    classes: np.ndarray,
    outliers: List[Dict[str, float]],
    out_path: Path,
):
    if not outliers:
        return
    df = pd.DataFrame(outliers)
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(df)), df["residual"], color="crimson")
    plt.xticks(range(len(df)), [f"{int(row['class_i'])}-{int(row['class_j'])}" for _, row in df.iterrows()], rotation=45, ha="right")
    plt.ylabel("Residual")
    plt.title("Top residual pairs")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_outlier_pairs_csv(outliers: List[Dict[str, float]], out_csv: Path):
    if not outliers:
        return
    df = pd.DataFrame(outliers)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
