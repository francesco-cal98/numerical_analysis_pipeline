from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from .ccnl_readout_dbn import forwardDBN
from .CLs import (
    Elastic_net_regression,
    Lasso_regression,
    Linear_regression,
    Logistic_regression_multiclass,
    Poisson_regression,
    Ridge_regression,
    SGD_regression,
)

plt.switch_backend("Agg")

CLASSIFIER_REGISTRY = {
    "SGD_regression": SGD_regression,
    "Ridge_regression": Ridge_regression,
    "Linear_regression": Linear_regression,
    "Logistic_regression_multiclass": Logistic_regression_multiclass,
    "Lasso_regression": Lasso_regression,
    "Elastic_net_regression": Elastic_net_regression,
    "Poisson_regression": Poisson_regression,
}


def _to_tensor(array, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.tensor(array, device=device, dtype=dtype)


def load_estimation_dataset(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    if not isinstance(dataset, dict):
        raise ValueError(f"[Estimation] File {path} must contain a dict.")

    data = dataset.get("data")
    labels = dataset.get("labels")
    idxs = dataset.get("idxs")

    if data is None or labels is None:
        raise ValueError(f"[Estimation] Dataset {path} missing 'data'/'labels'.")

    data_tensor = _to_tensor(data, device=device, dtype=torch.float32)
    labels_tensor = _to_tensor(labels, device=device, dtype=torch.float32)
    idxs_tensor = _to_tensor(idxs, device=device, dtype=torch.float32) if idxs is not None else None

    return {"data": data_tensor, "labels": labels_tensor, "idxs": idxs_tensor}


def _flatten_tensor(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().view(-1, tensor.shape[-1]).numpy()


def _flatten_labels(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().view(-1).numpy()


def _run_classifier(name: str, Xtrain: torch.Tensor, Xtest: torch.Tensor, Ytrain: torch.Tensor, Ytest: torch.Tensor):
    classifier = CLASSIFIER_REGISTRY.get(name)
    if classifier is None:
        raise ValueError(f"[Estimation] Unknown classifier '{name}'.")
    return classifier(Xtrain, Xtest, Ytrain, Ytest)


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classifier_name: str,
    model_label: str,
    out_path: Path,
    max_display_classes: int = 32,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    labels = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    labels = np.sort(labels)[:max_display_classes]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_percent[np.isnan(cm_percent)] = 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_percent, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(f"{classifier_name} — {model_label}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_task_numerosity_estimation(
    model,
    train_dataset: Dict[str, torch.Tensor],
    test_dataset: Dict[str, torch.Tensor],
    output_dir: Path,
    model_label: str,
    classifiers: Optional[Iterable[str]] = None,
    distribution: Optional[str] = None,
    max_display_classes: int = 32,
    wandb_run=None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_dataset["data"]
    Y_train = train_dataset["labels"]
    X_test = test_dataset["data"]
    Y_test = test_dataset["labels"]

    X_train_flat = _flatten_tensor(X_train)
    X_test_flat = _flatten_tensor(X_test)

    if Y_train.ndim > 2:
        Y_train_flat = _flatten_tensor(Y_train)
        Y_test_flat = _flatten_tensor(Y_test)
    else:
        Y_train_flat = _flatten_labels(Y_train)
        Y_test_flat = _flatten_labels(Y_test)

    classifier_list = list(classifiers or CLASSIFIER_REGISTRY.keys())
    results = {}

    for clf_name in classifier_list:
        try:
            acc_tr, pred_tr_cls, _, _, acc_te, pred_te_cls, _, _ = _run_classifier(
                clf_name,
                torch.tensor(X_train_flat),
                torch.tensor(X_test_flat),
                torch.tensor(Y_train_flat),
                torch.tensor(Y_test_flat),
            )

            clf_dir = output_dir / clf_name
            clf_dir.mkdir(parents=True, exist_ok=True)
            _plot_confusion(
                Y_test_flat,
                pred_te_cls,
                clf_name,
                model_label,
                clf_dir / f"{clf_name}_confusion.png",
                max_display_classes=max_display_classes,
            )

            results[clf_name] = {
                "accuracy_train": float(acc_tr),
                "accuracy_test": float(acc_te),
            }

            if wandb_run:
                try:
                    import wandb

                    wandb_run.log({
                        f"behavioral/estimation/{clf_name}/accuracy_test": acc_te,
                        f"behavioral/estimation/{clf_name}/accuracy_train": acc_tr,
                    })
                except Exception:
                    pass

        except Exception as exc:
            print(f"[Behavioral] Estimation classifier {clf_name} failed: {exc}")

    return results
