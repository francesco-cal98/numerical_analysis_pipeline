#!/usr/bin/env python3
"""Generate behavioral datasets used by the analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from groundeep_analysis.internal.behavioral.dataset_builder import (
    build_fixed_reference_dataset,
    build_naming_dataset,
    export_dataset_summary,
    load_percentages_map,
    reduce_behavioral_dataset,
)


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def _ratio(value: str) -> float:
    fvalue = float(value)
    if not 0 < fvalue <= 1:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1].")
    return fvalue


def add_fixed_ref_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fixed-ref", help="Build fixed-reference (single stimulus) datasets."
    )
    parser.add_argument("--mat-file", type=Path, required=True, help="Path to the .mat stimulus file.")
    parser.add_argument("--ref", type=int, required=True, help="Reference numerosity (e.g., 8).")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pkl file.")
    parser.add_argument("--num-samples", type=_positive_int, default=15200, help="Total samples to draw.")
    parser.add_argument("--batch-size", type=_positive_int, default=100, help="Samples per batch in the pickle.")
    parser.add_argument(
        "--percentages",
        type=str,
        default=None,
        help="Mapping of numerosity->percentage. Accepts 'N:value,...' or a JSON/YAML file.",
    )
    parser.add_argument("--binarize", action="store_true", help="Binarize images before saving.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for numpy/torch.")


def add_naming_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("naming", help="Build numerosity naming datasets.")
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=_positive_int, default=15200)
    parser.add_argument("--batch-size", type=_positive_int, default=100)
    parser.add_argument(
        "--percentages",
        type=str,
        default=None,
        help="Mapping of numerosity->percentage. Accepts 'N:value,...' or a JSON/YAML file.",
    )
    parser.add_argument(
        "--label-mode", choices=("int", "log"), default="int", help="How to encode targets in the pickle."
    )
    parser.add_argument("--add-zero-images", action="store_true", help="Append blank images as numerosity zero.")
    parser.add_argument(
        "--limit-fa", action="store_true", default=True, help="Filter stimuli to keep dots near the center."
    )
    parser.add_argument(
        "--no-limit-fa",
        action="store_false",
        dest="limit_fa",
        help="Disable the center-radius filter applied by --limit-fa.",
    )
    parser.add_argument("--limit-radius", type=_positive_int, default=40, help="Radius used with --limit-fa.")
    parser.add_argument("--seed", type=int, default=None)


def add_reduce_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("reduce", help="Apply stratified down-sampling to an existing dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--ratio", type=_ratio, required=True, help="Fraction of samples to keep (0-1].")
    parser.add_argument("--batch-size", type=_positive_int, default=152)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)


def add_summary_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("summary", help="Write basic dataset stats to CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_fixed_ref_parser(subparsers)
    add_naming_parser(subparsers)
    add_reduce_parser(subparsers)
    add_summary_parser(subparsers)
    args = parser.parse_args()

    if args.command == "fixed-ref":
        pct = load_percentages_map(args.percentages)
        build_fixed_reference_dataset(
            args.mat_file,
            args.ref,
            output_path=args.output,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            binarize=args.binarize,
            percentages=pct,
            seed=args.seed,
        )
        print(f"[fixed-ref] saved {args.output}")
    elif args.command == "naming":
        pct = load_percentages_map(args.percentages)
        build_naming_dataset(
            args.mat_file,
            output_path=args.output,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            label_mode=args.label_mode,
            add_zero_images=args.add_zero_images,
            limit_fa=args.limit_fa,
            limit_radius=args.limit_radius,
            percentages=pct,
            seed=args.seed,
        )
        print(f"[naming] saved {args.output}")
    elif args.command == "reduce":
        dest = reduce_behavioral_dataset(
            args.input,
            args.mat_file,
            reduction_ratio=args.ratio,
            batch_size=args.batch_size,
            output_path=args.output,
            seed=args.seed,
        )
        print(f"[reduce] saved {dest}")
    elif args.command == "summary":
        export_dataset_summary(args.input, args.output)
        print(f"[summary] wrote {args.output}")
    else:  # pragma: no cover
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
