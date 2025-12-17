"""Behavioral tasks stage."""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from scipy import io

from .analysis import (
    load_behavioral_inputs,
    run_behavioral_analysis,
)
from .comparison import run_task_comparison
from .fixed_reference_runner import run_fixed_reference_comparison
from .dataset_builder import (
    generate_fixed_reference_dataset,
    load_percentages_map,
)

def _discover_local_analyses_dir() -> Optional[Path]:
    """Return repo-local analyses directory if it exists."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "local_assets" / "src" / "analyses"
        if candidate.exists():
            return candidate
    return None


LOCAL_ANALYSES_DIR = _discover_local_analyses_dir()
if LOCAL_ANALYSES_DIR:
    local_str = str(LOCAL_ANALYSES_DIR)
    if local_str not in sys.path:
        sys.path.insert(0, local_str)


OPTIONAL_IMPORT_ERRORS: Dict[str, str] = {}


def _try_optional_import(module_name: str, attr: str):
    """Attempt to import an optional symbol and capture the reason if it fails."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except Exception as exc:  # pragma: no cover - diagnostic path
        OPTIONAL_IMPORT_ERRORS[module_name] = f"{exc.__class__.__name__}: {exc}"
        return None


def _missing_reason(module_candidates: Sequence[str]) -> str:
    for name in module_candidates:
        reason = OPTIONAL_IMPORT_ERRORS.get(name)
        if reason:
            return reason
    return "module not found"


FIXED_REFERENCE_MODULES = (
    "groundeep_analysis.internal.behavioral.task_fixed_reference",
    "task_fixed_reference",
)
ESTIMATION_MODULES = (
    "groundeep_analysis.internal.behavioral.task_numerosity_estimation",
    "task_numerosity_estimation",
)


load_fixed_reference_inputs = _try_optional_import(FIXED_REFERENCE_MODULES[0], "load_fixed_reference_inputs")
run_task_fixed_reference = _try_optional_import(FIXED_REFERENCE_MODULES[0], "run_task_fixed_reference")
if load_fixed_reference_inputs is None:
    load_fixed_reference_inputs = _try_optional_import(FIXED_REFERENCE_MODULES[1], "load_fixed_reference_inputs")
if run_task_fixed_reference is None:
    run_task_fixed_reference = _try_optional_import(FIXED_REFERENCE_MODULES[1], "run_task_fixed_reference")
HAS_FIXED_REFERENCE = load_fixed_reference_inputs is not None and run_task_fixed_reference is not None


load_estimation_dataset = _try_optional_import(ESTIMATION_MODULES[0], "load_estimation_dataset")
run_task_numerosity_estimation = _try_optional_import(ESTIMATION_MODULES[0], "run_task_numerosity_estimation")
if load_estimation_dataset is None:
    load_estimation_dataset = _try_optional_import(ESTIMATION_MODULES[1], "load_estimation_dataset")
if run_task_numerosity_estimation is None:
    run_task_numerosity_estimation = _try_optional_import(ESTIMATION_MODULES[1], "run_task_numerosity_estimation")
HAS_ESTIMATION = load_estimation_dataset is not None and run_task_numerosity_estimation is not None


def _compute_percentages_from_ratios(
    mat_path: Path, ref_num: int, limits: Sequence[float], percentages: Sequence[float]
) -> Dict[int, float]:
    if len(limits) < 2 or len(percentages) != len(limits):
        raise ValueError("ratio_buckets requires len(limits) >= 2 and len(percentages) == len(limits)")
    contents = io.loadmat(mat_path)
    n_list = np.squeeze(contents["N_list"]).astype(float)
    unique_n = np.unique(n_list)
    distributed: Dict[int, float] = {int(val): 0.0 for val in unique_n}
    ratios = unique_n / ref_num
    for idx in range(len(limits) - 1):
        lower = limits[idx]
        upper = limits[idx + 1]
        bucket_indices = [
            pos for pos in np.where((ratios > lower) & (ratios <= upper))[0] if unique_n[pos] != ref_num
        ]
        if not bucket_indices:
            continue
        share = float(percentages[idx + 1]) / len(bucket_indices)
        for pos in bucket_indices:
            distributed[int(unique_n[pos])] = share
    distributed[int(ref_num)] = 0.0
    return distributed


def _resolve_inline_percentages(build_cfg: Dict[str, Any], ref_num: int) -> Optional[Dict[int, float]]:
    map_spec = build_cfg.get("percentages")
    if map_spec is not None:
        return load_percentages_map(map_spec)

    ratio_cfg = build_cfg.get("ratio_buckets") or {}
    limits = ratio_cfg.get("limits")
    pct = ratio_cfg.get("percentages")
    if limits and pct:
        mat_for_stats = ratio_cfg.get("mat_file") or build_cfg.get("train_mat") or build_cfg.get("test_mat")
        if not mat_for_stats:
            raise ValueError("ratio_buckets requires either mat_file or train_mat/test_mat to be set.")
        return _compute_percentages_from_ratios(Path(mat_for_stats), ref_num, limits, pct)
    return None


def _tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = torch.float32 if tensor.dtype in (torch.int32, torch.int64, torch.float64) else tensor.dtype
    return tensor.to(device=device, dtype=dtype)


def _to_one_hot(labels: torch.Tensor) -> torch.Tensor:
    if labels.shape[-1] == 2:
        return labels
    zeros = (labels < 0.5).to(dtype=torch.float32, device=labels.device)
    ones = (labels >= 0.5).to(dtype=torch.float32, device=labels.device)
    return torch.stack([zeros, ones], dim=-1)


class BehavioralStage:
    """Stage for behavioral task suite (comparison, fixed reference, estimation)."""

    name = "behavioral"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        return settings.get('enabled', False)

    def _get_model_device(self, model_obj) -> torch.device:
        """Infer device from model."""
        try:
            import torch.nn as nn
            if isinstance(model_obj, nn.Module):
                try:
                    return next(model_obj.parameters()).device
                except StopIteration:
                    pass
        except Exception:
            pass

        try:
            layers = getattr(model_obj, "layers", [])
            if layers:
                first_rbm = layers[0]
                for attr_name in ("W", "c", "b", "weights"):
                    attr_val = getattr(first_rbm, attr_name, None)
                    if isinstance(attr_val, torch.Tensor):
                        return attr_val.device
        except Exception:
            pass
        return torch.device("cpu")

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Execute behavioral tasks."""
        tasks_cfg = settings.get("tasks", {})
        model_for_behavior = (
            ctx.get_model("uniform")
            if ctx.spec.distribution == "uniform"
            else ctx.get_model("zipfian")
        )
        behaviors_dir = output_dir / "behavioral"
        behavior_label = f"{ctx.spec.arch_name}_{ctx.spec.distribution}"

        # Load base behavioral inputs
        train_path = settings.get("train_pickle")
        test_path = settings.get("test_pickle")
        mat_path = settings.get("mat_file")

        if not (train_path and test_path and mat_path):
            print("[Behavioral] Missing paths: train_pickle/test_pickle/mat_file")
            return

        try:
            device_behavior = self._get_model_device(model_for_behavior)
            behavioral_inputs = load_behavioral_inputs(
                Path(train_path), Path(test_path), Path(mat_path), device_behavior
            )
        except Exception as exc:
            print(f"[Behavioral] Failed to load datasets ({exc})")
            return

        # Run base behavioral analysis
        guess_rate = float(settings.get("guess_rate", 0.01))
        results_beh = run_behavioral_analysis(
            model_for_behavior, behavioral_inputs,
            behaviors_dir, behavior_label, guess_rate=guess_rate
        )

        if ctx.wandb_run:
            try:
                import wandb
                ctx.wandb_run.log({
                    "behavioral/accuracy_test": results_beh.get("accuracy_test"),
                    "behavioral/accuracy_train": results_beh.get("accuracy_train"),
                    "behavioral/beta_number": results_beh.get("beta_number"),
                    "behavioral/beta_size": results_beh.get("beta_size"),
                    "behavioral/beta_spacing": results_beh.get("beta_spacing"),
                    "behavioral/weber_fraction": results_beh.get("weber_fraction"),
                })
            except Exception:
                pass

        # Comparison task
        comparison_cfg = tasks_cfg.get("comparison", {})
        if comparison_cfg.get("enabled", False):
            self._run_comparison_task(
                ctx, model_for_behavior, behavioral_inputs,
                comparison_cfg, settings, behaviors_dir, behavior_label, device_behavior
            )

        # Fixed reference task
        fixed_cfg = tasks_cfg.get("fixed_reference", {})
        if fixed_cfg.get("enabled", False):
            self._run_fixed_reference_task(
                ctx, model_for_behavior, fixed_cfg, settings,
                behaviors_dir, behavior_label, device_behavior
            )

        # Estimation task
        estimation_cfg = tasks_cfg.get("estimation", {})
        if estimation_cfg.get("enabled", False):
            self._run_estimation_task(
                ctx, estimation_cfg, behaviors_dir, behavior_label
            )

    def _run_comparison_task(self, ctx, model, behavioral_inputs, comparison_cfg,
                            settings, behaviors_dir, behavior_label, device_behavior):
        """Run comparison task."""
        out_cmp = behaviors_dir / "comparison"
        guess_rate_cmp = float(comparison_cfg.get("guess_rate", settings.get("guess_rate", 0.01)))

        try:
            comparison_inputs = behavioral_inputs
            cmp_train = comparison_cfg.get("train_pickle")
            cmp_test = comparison_cfg.get("test_pickle")
            if cmp_train and cmp_test:
                cmp_mat = comparison_cfg.get("mat_file") or settings.get("mat_file")
                comparison_inputs = load_behavioral_inputs(
                    Path(cmp_train), Path(cmp_test),
                    Path(cmp_mat), device_behavior or self._get_model_device(model)
                )

            results_cmp = run_task_comparison(
                model, comparison_inputs, out_cmp, behavior_label, guess_rate=guess_rate_cmp
            )

            if ctx.wandb_run:
                ctx.wandb_run.log({
                    "behavioral/comparison/accuracy_test": results_cmp.get("accuracy_test"),
                    "behavioral/comparison/accuracy_train": results_cmp.get("accuracy_train"),
                    "behavioral/comparison/weber_fraction": results_cmp.get("weber_fraction"),
                    "behavioral/comparison/beta_number": results_cmp.get("beta_number"),
                })
        except Exception as exc:
            print(f"[Behavioral] Comparison task failed: {exc}")

    def _build_inline_fixed_ref_inputs(
        self,
        ref_num: int,
        build_cfg: Dict[str, Any],
        device_behavior: torch.device,
        default_mat_path: Optional[Path],
    ) -> Dict[str, torch.Tensor]:
        train_mat = build_cfg.get("train_mat")
        test_mat = build_cfg.get("test_mat")
        if not (train_mat and test_mat):
            raise ValueError("build_from_mat requires both 'train_mat' and 'test_mat'.")

        binarize = bool(build_cfg.get("binarize", False))
        base_samples = int(build_cfg.get("num_samples", 15200))
        train_samples = int(build_cfg.get("train_num_samples", base_samples))
        test_samples = int(build_cfg.get("test_num_samples", base_samples))
        base_batch = int(build_cfg.get("batch_size", 100))
        train_batch = int(build_cfg.get("train_batch_size", base_batch))
        test_batch = int(build_cfg.get("test_batch_size", base_batch))

        seed = build_cfg.get("seed")
        train_seed = build_cfg.get("train_seed", seed)
        test_seed = build_cfg.get("test_seed", (None if seed is None else seed + 1))

        percentages = _resolve_inline_percentages(build_cfg, ref_num)

        train_dataset = generate_fixed_reference_dataset(
            Path(train_mat),
            ref_num,
            num_samples=train_samples,
            batch_size=train_batch,
            binarize=binarize,
            percentages=percentages,
            seed=train_seed,
        )
        test_dataset = generate_fixed_reference_dataset(
            Path(test_mat),
            ref_num,
            num_samples=test_samples,
            batch_size=test_batch,
            binarize=binarize,
            percentages=percentages,
            seed=test_seed,
        )

        mat_source = (
            build_cfg.get("analysis_mat")
            or build_cfg.get("mat_file")
            or default_mat_path
            or test_mat
        )
        mat_path = Path(mat_source)
        mat_contents = io.loadmat(str(mat_path))

        return {
            "X_train": _tensor_to_device(train_dataset["data"], device_behavior),
            "Y_train": _to_one_hot(_tensor_to_device(train_dataset["labels"], device_behavior)),
            "idx_train": _tensor_to_device(train_dataset["idxs"], device_behavior),
            "X_test": _tensor_to_device(test_dataset["data"], device_behavior),
            "Y_test": _to_one_hot(_tensor_to_device(test_dataset["labels"], device_behavior)),
            "idx_test": _tensor_to_device(test_dataset["idxs"], device_behavior),
            "mat": mat_contents,
        }

    def _run_fixed_reference_task(self, ctx, model, fixed_cfg, settings,
                                  behaviors_dir, behavior_label, device_behavior):
        """Run fixed reference task."""
        if not HAS_FIXED_REFERENCE:
            reason = _missing_reason(FIXED_REFERENCE_MODULES)
            print(f"[Behavioral] Fixed reference task unavailable ({reason}).")
            return

        runner_cfg = fixed_cfg.get("comparison_runner")
        if runner_cfg and runner_cfg.get("enabled", False):
            placeholders = {
                "arch": getattr(ctx.spec, "arch_name", ""),
                "distribution": getattr(ctx.spec, "distribution", ""),
                "dataset": getattr(ctx.spec, "distribution", ""),
            }
            try:
                run_fixed_reference_comparison(
                    runner_cfg,
                    device=device_behavior or self._get_model_device(model),
                    spec_placeholders=placeholders,
                    output_dir=behaviors_dir / "fixed_reference_runner",
                )
            except Exception as exc:
                print(f"[Behavioral] Fixed reference runner failed: {exc}")

        refs = fixed_cfg.get("references", [])
        train_template = fixed_cfg.get("train_template")
        test_template = fixed_cfg.get("test_template")
        mat_path = fixed_cfg.get("mat_file") or settings.get("mat_file")
        build_cfg = fixed_cfg.get("build_from_mat")
        guess_rate = float(fixed_cfg.get("guess_rate", settings.get("guess_rate", 0.01)))

        if build_cfg:
            if not mat_path:
                mat_path = build_cfg.get("analysis_mat") or build_cfg.get("test_mat")
            if not mat_path:
                print("[Behavioral] Fixed reference: build_from_mat requires 'analysis_mat' or mat_file.")
                return
        elif not (train_template and test_template and mat_path):
            print("[Behavioral] Fixed reference: missing paths")
            return
        mat_path = Path(mat_path)

        for ref in refs:
            try:
                if build_cfg:
                    fixed_inputs = self._build_inline_fixed_ref_inputs(
                        ref,
                        build_cfg,
                        device_behavior or self._get_model_device(model),
                        mat_path,
                    )
                else:
                    train_p = Path(str(train_template).format(ref=ref))
                    test_p = Path(str(test_template).format(ref=ref))
                    fixed_inputs = load_fixed_reference_inputs(
                        train_p,
                        test_p,
                        mat_path,
                        device_behavior or self._get_model_device(model),
                    )

                out_dir = behaviors_dir / "fixed_reference" / f"ref{ref}"
                results_fr = run_task_fixed_reference(
                    model, fixed_inputs, out_dir,
                    f"{behavior_label}_ref{ref}",
                    ref_num=ref,
                    guess_rate=guess_rate
                )

                if ctx.wandb_run:
                    ctx.wandb_run.log({
                        f"behavioral/fixed_ref{ref}/accuracy": results_fr.get("accuracy"),
                        f"behavioral/fixed_ref{ref}/weber_fraction": results_fr.get("weber_fraction"),
                    })
            except Exception as exc:
                print(f"[Behavioral] Fixed reference ref={ref} failed: {exc}")

    def _run_estimation_task(self, ctx, estimation_cfg, behaviors_dir, behavior_label):
        """Run estimation task."""
        if not HAS_ESTIMATION:
            reason = _missing_reason(ESTIMATION_MODULES)
            print(f"[Behavioral] Estimation task unavailable ({reason}).")
            return
        datasets_cfg = estimation_cfg.get("datasets", {})
        uniform_cfg = datasets_cfg.get("uniform", {})
        zipfian_cfg = datasets_cfg.get("zipfian", {})

        dist_name = ctx.spec.distribution
        target_cfg = uniform_cfg if dist_name == "uniform" else zipfian_cfg

        if not target_cfg:
            print(f"[Behavioral] Estimation: no config for {dist_name}")
            return

        try:
            train_pkl = target_cfg.get("train_pickle")
            test_pkl = target_cfg.get("test_pickle")
            if not (train_pkl and test_pkl):
                print(f"[Behavioral] Estimation: missing dataset paths for {dist_name}")
                return

            model = (
                ctx.get_model("uniform")
                if dist_name == "uniform"
                else ctx.get_model("zipfian")
            )
            device = self._get_model_device(model)

            train_dataset = load_estimation_dataset(Path(train_pkl), device)
            test_dataset = load_estimation_dataset(Path(test_pkl), device)

            out_dir = behaviors_dir / "estimation" / dist_name
            classifiers = estimation_cfg.get("classifiers", ["SGD_regression"])
            max_display = estimation_cfg.get("max_display_classes", 32)

            results_est = run_task_numerosity_estimation(
                model,
                train_dataset,
                test_dataset,
                out_dir,
                f"{behavior_label}_estimation",
                classifiers=classifiers,
                distribution=dist_name,
                max_display_classes=max_display,
                wandb_run=ctx.wandb_run,
            )

            if ctx.wandb_run:
                for clf_name, metrics in results_est.items():
                    ctx.wandb_run.log({
                        f"behavioral/estimation/{clf_name}/accuracy_test": metrics.get("accuracy_test"),
                        f"behavioral/estimation/{clf_name}/accuracy_train": metrics.get("accuracy_train"),
                    })
        except Exception as exc:
            print(f"[Behavioral] Estimation task failed: {exc}")
