from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            if self.iterable is not None:
                return iter(self.iterable)
            return iter(range(self.total or 0))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *args, **kwargs):
            pass

    def tqdm(iterable=None, *args, **kwargs):
        return _DummyTqdm(iterable=iterable, total=kwargs.get("total"))

plt.switch_backend("Agg")


@dataclass
class PCAComponentReport:
    component: int
    variance_ratio: float
    pearson_corr: float
    pearson_pval: float


@dataclass
class PCASummary:
    total_variance_ratio: float
    mean_corr: float
    component_reports: List[PCAComponentReport]


def _create_colormap(labels: np.ndarray) -> ListedColormap:
    labels = np.asarray(labels).astype(int)
    uniq = np.unique(labels)
    n = len(uniq)
    cmap = plt.cm.get_cmap("viridis", n)
    return ListedColormap([cmap(i) for i in range(n)])


def _scatter_with_colorbar(ax, x, y, labels, title: str):
    cmap = _create_colormap(labels)
    sc = ax.scatter(x, y, c=labels, cmap=cmap, s=10, alpha=0.7)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc, cax=cax)


def _plot_hist(ax, data: np.ndarray, title: str, xlabel: str):
    ax.hist(data, bins=30, color="steelblue", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def _plot_heatmap(ax, matrix: np.ndarray, title: str, labels: Iterable[str] | None = None):
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)


def _compute_pca(embeddings: np.ndarray, n_components: int, standardize: bool = True):
    X = embeddings.astype(np.float64, copy=False)
    if standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X)
    return pca, scores


def _compute_tsne(embeddings: np.ndarray, n_components: int = 2, *, perplexity: float = 30.0, n_iter: int = 1000, random_state: int = 42):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def _plot_pairwise_distances(ax, embeddings: np.ndarray, labels: np.ndarray, title: str):
    dist_matrix = pairwise_distances(embeddings, metric="euclidean")
    order = np.argsort(labels)
    dist_sorted = dist_matrix[order][:, order]
    labels_sorted = labels[order]
    im = ax.imshow(dist_sorted, cmap="magma", aspect="auto")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Samples (sorted by class)")
    ax.set_ylabel("Samples (sorted by class)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02,
        0.95,
        f"Classes: {len(np.unique(labels))}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
    )


def _plot_component_correlations(ax, scores: np.ndarray, labels: np.ndarray):
    corrs = []
    pvals = []
    for i in range(scores.shape[1]):
        corr, pval = pearsonr(scores[:, i], labels)
        corrs.append(corr)
        pvals.append(pval)
    ax.bar(range(1, scores.shape[1] + 1), corrs, color="teal")
    ax.set_xlabel("Component")
    ax.set_ylabel("Pearson r")
    ax.set_title("Component vs label correlation")
    for i, p in enumerate(pvals, start=1):
        ax.text(i, corrs[i - 1], f"p={p:.2e}", ha="center", va="bottom", fontsize=8, rotation=90)


def _plot_variance_ratio(ax, explained_variance_ratio: np.ndarray):
    ax.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker="o", color="indianred")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Explained variance per component")


def _plot_2d_projection(ax, coords: np.ndarray, labels: np.ndarray, title: str):
    _scatter_with_colorbar(ax, coords[:, 0], coords[:, 1], labels, title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _plot_3d_projection(fig: Figure, coords: np.ndarray, labels: np.ndarray, title: str):
    ax = fig.add_subplot(111, projection="3d")
    cmap = _create_colormap(labels)
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap=cmap, s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    fig.colorbar(sc, ax=ax, shrink=0.6)


def _save_fig(fig: Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _summarize_components(scores: np.ndarray, labels: np.ndarray, explained_variance_ratio: np.ndarray) -> PCASummary:
    component_reports = []
    corrs = []
    for i in range(scores.shape[1]):
        corr, pval = pearsonr(scores[:, i], labels)
        component_reports.append(
            PCAComponentReport(
                component=i + 1,
                variance_ratio=float(explained_variance_ratio[i]),
                pearson_corr=float(corr),
                pearson_pval=float(pval),
            )
        )
        corrs.append(corr)

    return PCASummary(
        total_variance_ratio=float(explained_variance_ratio.sum()),
        mean_corr=float(np.mean(corrs)),
        component_reports=component_reports,
    )


def _save_summary_json(summary: PCASummary, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_variance_ratio": summary.total_variance_ratio,
        "mean_corr": summary.mean_corr,
        "components": [asdict(comp) for comp in summary.component_reports],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    perplexity: float,
    n_iter: int,
    random_state: int,
    max_samples: int | None = None,
    progress_desc: str = "t-SNE",
):
    X = embeddings
    y = labels
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(X.shape[0], size=max_samples, replace=False))
        X = X[idx]
        y = y[idx]
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    with tqdm(total=n_iter, desc=progress_desc, leave=False) as pbar:
        coords = tsne.fit_transform(X)
        pbar.update(n_iter)
    return coords, y


def _plot_label_histogram(ax, labels: np.ndarray):
    labels = np.asarray(labels).astype(int)
    uniq, counts = np.unique(labels, return_counts=True)
    ax.bar(uniq, counts, color="steelblue", alpha=0.85)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Label histogram")


def generate_pca_decomposition_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    out_dir: Path,
    palette: str = "viridis",
    n_components: int = 10,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
    tsne_max_samples: int | None = 5000,
    random_state: int = 42,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 5))
    _plot_3d_projection(fig, embeddings[:, :3], labels, "Raw space (first three dims)")
    _save_fig(fig, out_dir / "raw_space_projection.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _plot_label_histogram(axes[0, 0], labels)
    _plot_pairwise_distances(axes[0, 1], embeddings, labels, "Pairwise distances (sorted)")
    _plot_hist(axes[1, 0], np.linalg.norm(embeddings, axis=1), "Embedding norms", "Norm")
    _plot_hist(axes[1, 1], embeddings[:, 0], "First dimension distribution", "Value")
    _save_fig(fig, out_dir / "dataset_overview.png")

    pca, scores = _compute_pca(embeddings, n_components=n_components)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _plot_variance_ratio(axes[0, 0], pca.explained_variance_ratio_)
    _plot_component_correlations(axes[0, 1], scores, labels)
    _plot_2d_projection(axes[1, 0], scores[:, :2], labels, "PCA 2D projection")
    _plot_2d_projection(axes[1, 1], scores[:, [0, 2]], labels, "PCA (PC1 vs PC3)")
    _save_fig(fig, out_dir / "pca_projection.png")

    fig = plt.figure(figsize=(6, 5))
    _plot_3d_projection(fig, scores[:, :3], labels, "PCA 3D projection")
    _save_fig(fig, out_dir / "pca_projection_3d.png")

    coords_tsne, labels_tsne = _run_tsne(
        embeddings,
        labels,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        random_state=random_state,
        max_samples=tsne_max_samples,
        progress_desc="t-SNE embeddings",
    )

    fig = plt.figure(figsize=(6, 5))
    _plot_2d_projection(fig.add_subplot(111), coords_tsne, labels_tsne, "t-SNE projection")
    _save_fig(fig, out_dir / "tsne_projection.png")

    summary = _summarize_components(scores, labels, pca.explained_variance_ratio_)
    _save_summary_json(summary, out_dir / "pca_summary.json")

    explained_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "variance_ratio": pca.explained_variance_ratio_,
        }
    )
    explained_df.to_csv(out_dir / "explained_variance.csv", index=False)

    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    scores_df["label"] = labels
    scores_df.to_csv(out_dir / "pca_scores.csv", index=False)

    return {
        "scores_path": str(out_dir / "pca_scores.csv"),
        "explained_variance_path": str(out_dir / "explained_variance.csv"),
        "summary_json": str(out_dir / "pca_summary.json"),
        "tsne_coords": coords_tsne.tolist(),
    }
