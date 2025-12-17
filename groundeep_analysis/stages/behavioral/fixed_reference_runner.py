"""Legacy-style fixed reference comparison runner."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import io

# NOTE: This module depends on legacy dunja_scripts which are not included
# in the portable version. If you need fixed_reference tasks, you'll need to:
# 1. Copy dunja_scripts/ from the original Groundeep repository
# 2. Uncomment the imports below
# 3. Place dunja_scripts/ in this directory (behavioral/)

try:
    from .dunja_scripts.datasets_utils import single_stimuli_dataset_modified
    from .dunja_scripts.CCNL_models import forwardDBN
    from .dunja_scripts.CLs import SGD_class_fixed, beta_extraction_ref_z
    _HAS_DUNJA_SCRIPTS = True
except ImportError:
    _HAS_DUNJA_SCRIPTS = False
    single_stimuli_dataset_modified = None
    forwardDBN = None
    SGD_class_fixed = None
    beta_extraction_ref_z = None


def _default_layer_grid() -> List[Tuple[int, int]]:
    return [
        (500, 500),
        (500, 1000),
        (500, 1500),
        (500, 2000),
        (1000, 500),
        (1000, 1000),
        (1000, 1500),
        (1000, 2000),
        (1500, 500),
        (1500, 1500),
        (1500, 2000),
        (1500, 1000),
    ]


def _tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = torch.float32 if tensor.dtype in (torch.int32, torch.int64, torch.float64) else tensor.dtype
    return tensor.to(device=device, dtype=dtype)


def _to_one_hot(labels: torch.Tensor) -> torch.Tensor:
    if labels.shape[-1] == 2:
        return labels
    zeros = (labels < 0.5).to(dtype=torch.float32, device=labels.device)
    ones = (labels >= 0.5).to(dtype=torch.float32, device=labels.device)
    return torch.stack([zeros, ones], dim=-1)


def _compute_percentages(mat_path: Path, ref_num: int, limits: Sequence[float], percentages: Sequence[float]) -> Dict[int, float]:
    if len(limits) < 2 or len(percentages) != len(limits):
        raise ValueError("ratio_buckets requires len(limits) >= 2 and len(percentages) == len(limits)")
    contents = io.loadmat(str(mat_path))
    n_list = np.squeeze(contents["N_list"]).astype(float)
    unique_n = np.unique(n_list)
    ratios = unique_n / ref_num
    distributed = {int(val): 0.0 for val in unique_n}
    for idx in range(len(limits) - 1):
        lower = limits[idx]
        upper = limits[idx + 1]
        bucket = [
            pos for pos in np.where((ratios > lower) & (ratios <= upper))[0]
            if unique_n[pos] != ref_num
        ]
        if not bucket:
            continue
        share = float(percentages[idx + 1]) / len(bucket)
        for pos in bucket:
            distributed[int(unique_n[pos])] = share
    distributed[int(ref_num)] = 0.0
    return distributed


def _format_dbn_path(template: str, ref: int, run: int, h1: int, h2: int, placeholders: Dict[str, Any]) -> Path:
    context = {
        "ref": ref,
        "run": run,
        "h1": h1,
        "h2": h2,
    }
    context.update(placeholders)
    return Path(template.format(**context))


def run_fixed_reference_comparison(
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    spec_placeholders: Dict[str, Any],
    output_dir: Path,
) -> None:
    train_mat = Path(cfg["train_mat"])
    test_mat = Path(cfg["test_mat"])
    refs: List[int] = cfg.get("references") or [8, 14, 16, 20]
    limits: Sequence[float] = cfg.get("limits", [0.49, 2.0, 4.0])
    percentages: Sequence[float] = cfg.get("percentages", [0.0, 100.0, 0.0])
    num_samples: int = int(cfg.get("num_samples", 15200))
    batch_size: int = int(cfg.get("batch_size", 100))
    binarize: bool = bool(cfg.get("binarize", False))
    runs: int = int(cfg.get("runs", 1))
    layer_entries: Iterable = cfg.get("layer_sizes") or _default_layer_grid()
    layer_grid: List[Tuple[int, int]] = []
    for entry in layer_entries:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            layer_grid.append((int(entry[0]), int(entry[1])))
        else:
            text = str(entry).lower().replace("x", ",")
            parts = [p for p in text.split(",") if p]
            if len(parts) != 2:
                raise ValueError(f"Invalid layer size '{entry}'.")
            layer_grid.append((int(parts[0]), int(parts[1])))

    dbn_template = cfg.get("dbn_template")
    explicit_dbns = [Path(cfg["dbn_path"])] if "dbn_path" in cfg else [Path(p) for p in cfg.get("dbn_paths", [])]
    if not dbn_template and not explicit_dbns:
        raise ValueError("comparison_runner requires one of: 'dbn_path', 'dbn_paths', or 'dbn_template'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = Path(cfg.get("output_excel", output_dir / "fixed_reference_comparison.xlsx"))

    results: List[Dict[str, Any]] = []

    for ref in refs:
        pct_map = _compute_percentages(train_mat, ref, limits, percentages)
        train_dataset = single_stimuli_dataset_modified(
            str(train_mat),
            ref_num=ref,
            num_samples=num_samples,
            batch_size=batch_size,
            num_percentage_dict=pct_map,
            binarize=binarize,
        )
        test_dataset = single_stimuli_dataset_modified(
            str(test_mat),
            ref_num=ref,
            num_samples=num_samples,
            batch_size=batch_size,
            num_percentage_dict=pct_map,
            binarize=binarize,
        )

        N_list = io.loadmat(str(test_mat))["N_list"]
        TSA_list = io.loadmat(str(test_mat))["TSA_list"]
        FA_list = io.loadmat(str(test_mat))["FA_list"]

        for run in range(1, runs + 1):
            dbn_paths: List[Path] = []
            if dbn_template:
                grid = layer_grid or [(0, 0)]
                for h1, h2 in grid:
                    dbn_paths.append(_format_dbn_path(dbn_template, ref, run, h1, h2, spec_placeholders))
            else:
                dbn_paths.extend(explicit_dbns)

            for dbn_path in dbn_paths:
                if not dbn_path.exists():
                    print(f"[fixed_ref_runner] missing DBN: {dbn_path}")
                    continue
                with dbn_path.open("rb") as f:
                    dbn = pickle.load(f)

                X_train = _tensor_to_device(train_dataset["data"], device)
                Y_train = _to_one_hot(_tensor_to_device(train_dataset["labels"], device))
                idx_train = _tensor_to_device(train_dataset["idxs"], device)
                X_test = _tensor_to_device(test_dataset["data"], device)
                Y_test = _to_one_hot(_tensor_to_device(test_dataset["labels"], device))
                idx_test = _tensor_to_device(test_dataset["idxs"], device)

                X_train_comp = forwardDBN(dbn, X_train).clone()
                X_test_comp = forwardDBN(dbn, X_test).clone()

                acc_tr, _, acc_te, pred_te = SGD_class_fixed(
                    X_train_comp,
                    X_test_comp,
                    Y_train,
                    Y_test,
                )

                (
                    model_fit,
                    betas,
                    wf,
                    *_,
                ) = beta_extraction_ref_z(
                    pred_te,
                    idx_test,
                    N_list,
                    TSA_list,
                    FA_list,
                    ref_num=ref,
                )

                results.append({
                    "reference": ref,
                    "dbn": str(dbn_path),
                    "accuracy_train": acc_tr,
                    "accuracy_test": acc_te,
                    "intercept": model_fit,
                    "beta_number": betas[0],
                    "beta_size": betas[1],
                    "beta_spacing": betas[2],
                    "weber_fraction": wf,
                    "run": run,
                })

    if not results:
        print("[fixed_ref_runner] No results collected.")
        return

    df = pd.DataFrame(results)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)
    print(f"[fixed_ref_runner] Saved comparison to {excel_path}")
