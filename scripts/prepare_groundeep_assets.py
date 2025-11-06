#!/usr/bin/env python3
"""
Copy or symlink assets from the private Groundeep workspace into this repo.

Usage (copy mode):
    python scripts/prepare_groundeep_assets.py \
        --config ../Groundeep/src/configs/analysis.yaml \
        --source ../Groundeep \
        --dest local_assets

Usage (symlink mode):
    python scripts/prepare_groundeep_assets.py --mode link

The script parses the provided Hydra config, collects all referenced datasets,
models, and behavioral resources, then mirrors them under ``dest`` while
preserving the relative structure.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with `pip install pyyaml`.", file=sys.stderr)
    sys.exit(1)


def _to_abs(path_value: str | Path, base: Path, source_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate

    return (source_root / path).resolve()


def _collect_from_behavioral(cfg: Dict[str, Any], acc: Set[Path], base: Path, source_root: Path):
    behavioral = cfg.get("behavioral") or {}
    if not isinstance(behavioral, dict):
        return

    def _add(path_like):
        if not path_like:
            return
        resolved = _to_abs(Path(path_like), base, source_root)
        try:
            resolved.relative_to(source_root)
        except ValueError:
            return
        acc.add(resolved)

    for key in ("train_pickle", "test_pickle", "mat_file"):
        _add(behavioral.get(key))

    tasks = behavioral.get("tasks") or {}
    if not isinstance(tasks, dict):
        return

    comparison = tasks.get("comparison") or {}
    for key in ("train_pickle", "test_pickle", "mat_file"):
        _add(comparison.get(key))

    fixed_ref = tasks.get("fixed_reference") or {}
    refs = fixed_ref.get("references", [])
    train_template = fixed_ref.get("train_template")
    test_template = fixed_ref.get("test_template")
    mat_file = fixed_ref.get("mat_file")
    guess_rate = fixed_ref.get("guess_rate")  # noqa: F841 (documented for clarity)

    if mat_file:
        _add(mat_file)

    if isinstance(refs, Iterable):
        for ref in refs:
            if train_template:
                _add(Path(str(train_template).format(ref=ref)))
            if test_template:
                _add(Path(str(test_template).format(ref=ref)))

    estimation = tasks.get("estimation") or {}
    datasets = estimation.get("datasets") or {}
    if isinstance(datasets, dict):
        for dist_cfg in datasets.values():
            if not isinstance(dist_cfg, dict):
                continue
            for key in ("train_pickle", "test_pickle"):
                _add(dist_cfg.get(key))


def collect_asset_paths(config_path: Path, source_root: Path) -> Set[Path]:
    cfg_dir = config_path.parent.resolve()
    cfg = yaml.safe_load(config_path.read_text()) or {}
    paths: Set[Path] = set()

    models_cfg = cfg.get("models") or []
    if isinstance(models_cfg, list):
        for entry in models_cfg:
            if not isinstance(entry, dict):
                continue
            dataset_path = entry.get("dataset_path")
            dataset_name = entry.get("dataset_name")
            if dataset_path:
                dataset_dir = _to_abs(Path(dataset_path), cfg_dir, source_root)
                try:
                    dataset_dir.relative_to(source_root)
                    paths.add(dataset_dir)
                except ValueError:
                    pass
                if dataset_name:
                    dataset_file = dataset_dir / dataset_name
                    try:
                        dataset_file.relative_to(source_root)
                        paths.add(dataset_file)
                    except ValueError:
                        pass
            for key in ("model_uniform", "model_zipfian"):
                model_path = entry.get(key)
                if model_path:
                    abs_path = _to_abs(Path(model_path), cfg_dir, source_root)
                    try:
                        abs_path.relative_to(source_root)
                        paths.add(abs_path)
                    except ValueError:
                        pass

    _collect_from_behavioral(cfg, paths, cfg_dir, source_root)
    return paths


def copy_or_link(src: Path, src_root: Path, dest_root: Path, mode: str, force: bool, dry_run: bool) -> None:
    try:
        rel = src.relative_to(src_root)
    except ValueError:
        print(f"[skip] {src} (outside source root)")
        return

    dest = dest_root / rel
    if dry_run:
        print(f"[dry-run] {mode} {src} -> {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if not force:
            print(f"[skip] {dest} already exists (use --force to overwrite)")
            return
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    if mode == "link":
        os.symlink(src, dest, target_is_directory=src.is_dir())
        print(f"[link] {dest} -> {src}")
    else:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        print(f"[copy] {src} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local copies of Groundeep assets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../Groundeep/src/configs/analysis.yaml"),
        help="Hydra config to parse (defaults to the private Groundeep analysis config).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../Groundeep"),
        help="Root directory of the Groundeep workspace.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("local_assets"),
        help="Destination directory inside this repo.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "link"),
        default="copy",
        help="Copy files/directories (default) or create symlinks instead.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without copying/linking anything.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    source_root = args.source.resolve()
    dest_root = args.dest.resolve()

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not source_root.exists():
        print(f"ERROR: source root not found: {source_root}", file=sys.stderr)
        sys.exit(1)

    asset_paths = collect_asset_paths(config_path, source_root)
    if not asset_paths:
        print("No assets detected in config.")
        return

    print(f"Found {len(asset_paths)} asset(s) to process.")
    for path in sorted(asset_paths):
        if not path.exists():
            print(f"[missing] {path}")
            continue
        copy_or_link(path, source_root, dest_root, args.mode, args.force, args.dry_run)

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to perform the operations.")


if __name__ == "__main__":
    main()
