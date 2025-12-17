"""Utility wrappers to build behavioral datasets from legacy Groundeep scripts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

# Optional import for legacy dunja_scripts
try:
    from .dunja_scripts import datasets_utils as legacy_utils
    _HAS_LEGACY_UTILS = True
except ImportError:
    legacy_utils = None
    _HAS_LEGACY_UTILS = False


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_dataset(payload: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return output_path


def _parse_percentages(percentages: Mapping[int, float]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in percentages.items()}


def generate_fixed_reference_dataset(
    mat_file: Path,
    ref_num: int,
    *,
    num_samples: int = 15200,
    batch_size: int = 100,
    binarize: bool = False,
    percentages: Optional[Union[Mapping[int, float], str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Build (but do not save) a fixed-reference dataset."""
    _set_seed(seed)
    pct_map = load_percentages_map(percentages)
    return legacy_utils.single_stimuli_dataset_modified(
        str(mat_file),
        ref_num=ref_num,
        num_samples=num_samples,
        batch_size=batch_size,
        num_percentage_dict=pct_map,
        binarize=binarize,
    )


def build_fixed_reference_dataset(
    mat_file: Path,
    ref_num: int,
    *,
    output_path: Path,
    num_samples: int = 15200,
    batch_size: int = 100,
    binarize: bool = False,
    percentages: Optional[Union[Mapping[int, float], str]] = None,
    seed: Optional[int] = None,
) -> Path:
    """Generate and save a single-stimulus dataset for a fixed-reference task."""
    dataset = generate_fixed_reference_dataset(
        mat_file,
        ref_num,
        num_samples=num_samples,
        batch_size=batch_size,
        binarize=binarize,
        percentages=percentages,
        seed=seed,
    )
    return _save_dataset(dataset, output_path)


def generate_naming_dataset(
    mat_file: Path,
    *,
    num_samples: int = 15200,
    batch_size: int = 100,
    label_mode: str = "int",
    add_zero_images: bool = False,
    limit_fa: bool = True,
    limit_radius: int = 40,
    percentages: Optional[Union[Mapping[int, float], str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Build (but do not save) a numerosity naming dataset."""
    _set_seed(seed)
    pct_map = load_percentages_map(percentages)
    return legacy_utils.single_stimuli_dataset_naming(
        str(mat_file),
        num_samples=num_samples,
        batch_size=batch_size,
        num_percentage_dict=pct_map,
        label_type=label_mode,
        add_zero_images=add_zero_images,
        limit_FA=limit_fa,
        limit_radius=limit_radius,
    )


def build_naming_dataset(
    mat_file: Path,
    *,
    output_path: Path,
    num_samples: int = 15200,
    batch_size: int = 100,
    label_mode: str = "int",
    add_zero_images: bool = False,
    limit_fa: bool = True,
    limit_radius: int = 40,
    percentages: Optional[Union[Mapping[int, float], str]] = None,
    seed: Optional[int] = None,
) -> Path:
    """Generate and save a dataset for the numerosity estimation task."""
    dataset = generate_naming_dataset(
        mat_file,
        num_samples=num_samples,
        batch_size=batch_size,
        label_mode=label_mode,
        add_zero_images=add_zero_images,
        limit_fa=limit_fa,
        limit_radius=limit_radius,
        percentages=percentages,
        seed=seed,
    )
    return _save_dataset(dataset, output_path)


def reduce_behavioral_dataset(
    pickle_path: Path,
    mat_file: Path,
    *,
    reduction_ratio: float,
    batch_size: int = 152,
    output_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Path:
    """Apply the legacy stratified down-sampling to an existing behavioral dataset."""
    if not 0 < reduction_ratio <= 1:
        raise ValueError("reduction_ratio must be in (0, 1].")

    with pickle_path.open("rb") as f:
        payload = pickle.load(f)
    _set_seed(seed)
    reduced = legacy_utils.stratified_dataset_size_reduction(
        payload,
        str(mat_file),
        reduction_ratio,
        batch_size=batch_size,
    )
    dest = output_path or pickle_path.with_name(f"{pickle_path.stem}_reduced{pickle_path.suffix}")
    return _save_dataset(reduced, dest)


def export_dataset_summary(pickle_path: Path, csv_path: Path) -> Path:
    """Dump basic statistics about a behavioral dataset to CSV."""
    with pickle_path.open("rb") as f:
        payload = pickle.load(f)

    data = payload.get("data")
    labels = payload.get("labels")
    idxs = payload.get("idxs")
    summary = {
        "num_batches": int(getattr(data, "shape", [0])[0]),
        "batch_size": int(getattr(data, "shape", [0, 0])[1]),
        "num_features": int(getattr(data, "shape", [0, 0, 0])[2]),
        "label_shape": tuple(getattr(labels, "shape", ())),
        "idx_shape": tuple(getattr(idxs, "shape", ())),
    }
    import pandas as pd  # local import to keep optional dependency lightweight

    df = pd.DataFrame([summary])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_percentages_map(
    path_or_spec: Optional[Union[str, Mapping[int, float]]]
) -> Optional[Dict[int, float]]:
    """Parse a numerosity percentage mapping from CLI input or inline data."""
    if path_or_spec is None:
        return None
    if isinstance(path_or_spec, Mapping):
        return _parse_percentages(path_or_spec)

    candidate = Path(path_or_spec)
    if candidate.exists():
        if candidate.suffix in {".json", ".yaml", ".yml"}:
            if candidate.suffix in {".yaml", ".yml"}:
                import yaml

                content = yaml.safe_load(candidate.read_text())
            else:
                content = json.loads(candidate.read_text())
        else:
            raise ValueError(f"Unsupported mapping format: {candidate.suffix}")
    else:
        content = {}
        for chunk in path_or_spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            key, value = chunk.split(":")
            content[int(key)] = float(value)
    return _parse_percentages(content)
