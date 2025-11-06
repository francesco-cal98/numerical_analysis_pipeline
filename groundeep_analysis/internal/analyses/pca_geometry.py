from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

__all__ = [
    "run_pca_geometry",
]

EPS = 1e-12


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _scatter_matrices(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    n, d = X.shape
    mu = X.mean(axis=0)
    Sw = np.zeros((d, d), dtype=np.float64)
    Sb = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        idx = np.where(y == c)[0]
        Xc = X[idx]
        pc = len(idx) / n
        muc = Xc.mean(axis=0)
        Xc0 = Xc - muc
        if len(idx) > 1:
            Sw += pc * (Xc0.T @ Xc0) / (len(idx) - 1)
        dmu = (muc - mu)[:, None]
        Sb += pc * (dmu @ dmu.T)
    return Sw, Sb


def _eig_sorted(A: np.ndarray):
    w, V = np.linalg.eigh(A)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def _cos2(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (norm(a) + EPS)
    b_norm = b / (norm(b) + EPS)
    return float((a_norm @ b_norm) ** 2)


def _class_centroids(X: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])
    return centroids, classes


def _pca_on(X: np.ndarray, n: int = 3):
    pca = PCA(n_components=min(n, X.shape[1]), random_state=0)
    scores = pca.fit_transform(X)
    return pca, scores


def _balanced_indices(y: np.ndarray, per_class: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx_all = []
    for c in np.unique(y):
        Ic = np.where(y == c)[0]
        if len(Ic) <= per_class:
            idx_all.append(Ic)
        else:
            idx_all.append(rng.choice(Ic, per_class, replace=False))
    return np.concatenate(idx_all)


def _curvature_3pt(x: np.ndarray) -> np.ndarray:
    curv = []
    for i in range(1, len(x) - 1):
        curv.append(norm(x[i + 1] - 2 * x[i] + x[i - 1]))
    return np.array(curv)


@dataclass
class VarianceStats:
    trace_within: float
    anisotropy_within: float
    participation_within: float
    trace_between: float
    anisotropy_between: float
    participation_between: float


@dataclass
class AngleStats:
    rho_pc1: float
    rho_pc2: float
    angle_pc1_deg: float
    angle_pc2_deg: float


@dataclass
class BalancedPCAStats:
    rho_pc1: float
    rho_pc2: float


@dataclass
class CurvatureStats:
    mean: float
    low_mean: float
    high_mean: float


@dataclass
class PCAGeometryReport:
    variance: VarianceStats
    angles: AngleStats
    balanced: BalancedPCAStats
    curvature: CurvatureStats


def _variance_report(X: np.ndarray, y: np.ndarray) -> Tuple[VarianceStats, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    Sw, Sb = _scatter_matrices(X, y)
    wW, VW = _eig_sorted(Sw)
    wB, VB = _eig_sorted(Sb)
    traceW = float(wW.sum())
    traceB = float(wB.sum())
    anisW = float(wW[0] / (traceW + EPS))
    anisB = float(wB[0] / (traceB + EPS))
    prW = float((traceW**2) / (np.dot(wW, wW) + EPS))
    prB = float((traceB**2) / (np.dot(wB, wB) + EPS))
    stats = VarianceStats(
        trace_within=traceW,
        anisotropy_within=anisW,
        participation_within=prW,
        trace_between=traceB,
        anisotropy_between=anisB,
        participation_between=prB,
    )
    return stats, (wW, VW), (wB, VB)


def _angle_report(X: np.ndarray, y: np.ndarray) -> AngleStats:
    P, Z = _pca_on(X, n=3)
    rho1 = spearmanr(y, Z[:, 0]).statistic
    rho2 = spearmanr(y, Z[:, 1]).statistic

    centroids, _ = _class_centroids(X, y)
    Pc, _ = _pca_on(centroids, n=2)
    number_axis = Pc.components_[0]

    ang_p1 = math.degrees(math.acos(math.sqrt(_cos2(number_axis, P.components_[0]))))
    ang_p2 = math.degrees(math.acos(math.sqrt(_cos2(number_axis, P.components_[1]))))
    return AngleStats(
        rho_pc1=rho1,
        rho_pc2=rho2,
        angle_pc1_deg=ang_p1,
        angle_pc2_deg=ang_p2,
    )


def _balanced_report(X: np.ndarray, y: np.ndarray, per_class: int) -> BalancedPCAStats:
    idx = _balanced_indices(y, per_class)
    P, Z = _pca_on(X[idx], n=2)
    rho1 = spearmanr(y[idx], Z[:, 0]).statistic
    rho2 = spearmanr(y[idx], Z[:, 1]).statistic
    return BalancedPCAStats(rho_pc1=rho1, rho_pc2=rho2)


def _curvature_report(name: str, X: np.ndarray, y: np.ndarray, outdir: Path, run_isomap: bool) -> CurvatureStats:
    centroids, classes = _class_centroids(X, y)
    P3, Z3 = _pca_on(centroids, n=3)
    order = np.argsort(classes)
    traj = Z3[order]
    curv = _curvature_3pt(traj)
    c_mean = float(curv.mean())
    c_low = float(curv[:10].mean()) if curv.size >= 10 else float(curv.mean())
    c_high = float(curv[-11:].mean()) if curv.size >= 11 else float(curv.mean())

    plt.figure(figsize=(5, 4))
    plt.plot(range(2, len(traj)), curv, marker="o", lw=1.5)
    plt.axhline(c_mean, ls="--", lw=1)
    plt.title(f"{name}: PCA-3D curvature along centroids")
    plt.xlabel("Class index")
    plt.ylabel(r"$||x_{i+1} - 2x_i + x_{i-1}||$")
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"curvature_centroids_{name}.png", dpi=200)
    plt.close()

    if run_isomap:
        try:
            iso = Isomap(n_components=2)
            iso_traj = iso.fit_transform(traj)
            plt.figure(figsize=(5, 4))
            plt.plot(range(len(iso_traj)), iso_traj[:, 0], marker="o", lw=1.5)
            plt.title(f"{name}: Isomap dimension 0")
            plt.tight_layout()
            plt.savefig(outdir / f"isomap_dim0_{name}.png", dpi=200)
            plt.close()
        except Exception:
            pass

    return CurvatureStats(mean=c_mean, low_mean=c_low, high_mean=c_high)


def _save_variance_plots(name: str, eig: np.ndarray, outdir: Path):
    eig = np.asarray(eig, dtype=float)
    eig_ratio = eig / (eig.sum() + EPS)

    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(eig_ratio) + 1), eig_ratio, marker="o")
    plt.title(f"{name}: eigenvalue ratios")
    plt.xlabel("Component")
    plt.ylabel("Ratio")
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"eigenvalue_ratios_{name}.png", dpi=200)
    plt.close()


def _save_angle_plot(name: str, angles: AngleStats, outdir: Path):
    plt.figure(figsize=(5, 4))
    plt.bar(["PC1", "PC2"], [angles.angle_pc1_deg, angles.angle_pc2_deg], color=["tab:blue", "tab:orange"])
    plt.ylabel("Angle (deg)")
    plt.title(f"{name}: angle with number axis")
    plt.ylim(0, 90)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"angles_{name}.png", dpi=200)
    plt.close()


def _save_balanced_plot(name: str, balanced: BalancedPCAStats, outdir: Path):
    plt.figure(figsize=(5, 4))
    plt.bar(["PC1", "PC2"], [balanced.rho_pc1, balanced.rho_pc2], color=["tab:red", "tab:purple"])
    plt.ylabel("Spearman ρ")
    plt.title(f"{name}: balanced correlations")
    plt.ylim(-1, 1)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"balanced_corr_{name}.png", dpi=200)
    plt.close()


def _serialize_report(report: PCAGeometryReport) -> Dict[str, object]:
    return {
        "variance": asdict(report.variance),
        "angles": asdict(report.angles),
        "balanced": asdict(report.balanced),
        "curvature": asdict(report.curvature),
    }


def run_pca_geometry(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    outdir: Path,
    per_class: int = 200,
    components: Iterable[int] = (2, 3),
    zscore: bool = True,
    run_isomap: bool = False,
) -> PCAGeometryReport:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same first dimension.")

    X = embeddings.astype(np.float64, copy=False)
    if zscore:
        X = _zscore_cols(X)
    y = labels.astype(int)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variance_stats, (wW, _), (wB, _) = _variance_report(X, y)
    _save_variance_plots(name, wW, outdir / "within_class")
    _save_variance_plots(name, wB, outdir / "between_class")

    angle_stats = _angle_report(X, y)
    _save_angle_plot(name, angle_stats, outdir)

    balanced_stats = _balanced_report(X, y, per_class=per_class)
    _save_balanced_plot(name, balanced_stats, outdir)

    curvature_stats = _curvature_report(name, X, y, outdir, run_isomap)

    for d in components:
        if d <= 0 or d > X.shape[1]:
            continue
        pca = PCA(n_components=d, random_state=0)
        scores = pca.fit_transform(X)
        plt.figure(figsize=(6, 5))
        if d == 2:
            sc = plt.scatter(scores[:, 0], scores[:, 1], c=y, cmap="viridis", s=10, alpha=0.6)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        elif d == 3:
            ax = plt.axes(projection="3d")
            sc = ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=y, cmap="viridis", s=10, alpha=0.6)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
        else:
            continue
        plt.title(f"{name}: PCA projection (dim={d})")
        plt.colorbar(sc, label="Class")
        plt.tight_layout()
        plt.savefig(outdir / f"projection_{name}_dim{d}.png", dpi=200)
        plt.close()

    report = PCAGeometryReport(
        variance=variance_stats,
        angles=angle_stats,
        balanced=balanced_stats,
        curvature=curvature_stats,
    )

    with open(outdir / f"report_{name}.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_report(report), f, indent=2)

    return report
