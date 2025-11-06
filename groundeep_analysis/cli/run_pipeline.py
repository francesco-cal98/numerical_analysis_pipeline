"""
Simple CLI for running the GROUNDEEP analysis pipeline.

Example:
    python -m groundeep_analysis.cli.run_pipeline --config examples/configs/sample_analysis.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from groundeep_analysis.core.analysis_types import ModelSpec, AnalysisSettings
from groundeep_analysis.pipeline import run_analysis_pipeline


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if path.suffix == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported config format: {path.suffix}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GROUNDEEP analysis pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the output root directory defined in the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the random seed defined in the config.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb to be installed).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (optional).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional).",
    )

    args = parser.parse_args(argv)
    cfg = _load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    project_root = args.config.parent.resolve()

    if args.output_root:
        output_root = args.output_root
    else:
        raw_output = Path(cfg.get("output_root", "results/analysis"))
        output_root = raw_output if raw_output.is_absolute() else (project_root / raw_output)
    use_wandb = args.use_wandb or bool(cfg.get("use_wandb", False))
    wandb_project = args.wandb_project or cfg.get("wandb_project")
    wandb_run_name = args.wandb_run_name

    settings = AnalysisSettings.from_cfg(cfg)

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("No models specified in configuration file.")

    for model_cfg in models_cfg:
        spec = ModelSpec.from_config(model_cfg, project_root)
        run_analysis_pipeline(
            spec=spec,
            settings=settings,
            output_root=output_root,
            seed=seed,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )


if __name__ == "__main__":
    main()
