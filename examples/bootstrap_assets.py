#!/usr/bin/env python3
"""
Generate a toy dataset and lightweight models for the public demo.

Running this script will populate `examples/assets/` with:
    - toy_dataset.npz: synthetic numerosity dataset
    - toy_uniform_model.pkl
    - toy_zipfian_model.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundeep_analysis.examples.models import ToyMLP

@dataclass
class ToyDatasetConfig:
    image_size: int = 32
    num_classes: int = 8
    samples_per_class: int = 20
    radius_range: Tuple[int, int] = (2, 4)


def _draw_blob(canvas: np.ndarray, x: int, y: int, radius: int) -> None:
    yy, xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    canvas[mask] = 1.0


def generate_toy_dataset(cfg: ToyDatasetConfig, out_path: Path) -> None:
    rng = np.random.default_rng(1234)
    h = w = cfg.image_size
    n_samples = cfg.num_classes * cfg.samples_per_class

    images = np.zeros((n_samples, h * w), dtype=np.float32)
    numerosity = np.zeros(n_samples, dtype=np.int32)
    cum_area = np.zeros(n_samples, dtype=np.float32)
    convex_hull = np.zeros(n_samples, dtype=np.float32)
    density = np.zeros(n_samples, dtype=np.float32)
    mean_item_size = np.zeros(n_samples, dtype=np.float32)
    std_item_size = np.zeros(n_samples, dtype=np.float32)

    idx = 0
    for n in range(1, cfg.num_classes + 1):
        for _ in range(cfg.samples_per_class):
            canvas = np.zeros((h, w), dtype=np.float32)
            radii = []
            for _ in range(n):
                radius = rng.integers(cfg.radius_range[0], cfg.radius_range[1] + 1)
                x = rng.integers(radius, w - radius)
                y = rng.integers(radius, h - radius)
                _draw_blob(canvas, x, y, radius)
                radii.append(radius)

            images[idx] = canvas.reshape(-1)
            numerosity[idx] = n
            cum_area[idx] = float(canvas.sum())
            convex_hull[idx] = float(canvas.sum() ** 0.5 + rng.normal(0, 5.0))
            density[idx] = float(n / (h * w))
            mean_item_size[idx] = float(np.mean(radii))
            std_item_size[idx] = float(np.std(radii))
            idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        D=images,
        N_list=numerosity,
        cumArea_list=cum_area,
        CH_list=convex_hull,
        density=density,
        mean_item_size=mean_item_size,
        std_item_size=std_item_size,
    )
    print(f"[assets] Saved dataset to {out_path}")


def save_toy_model(path: Path, input_dim: int, seed: int) -> None:
    torch.manual_seed(seed)
    model = ToyMLP(input_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    print(f"[assets] Saved model to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy assets for GROUNDEEP analysis.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="Directory where assets will be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_path = output_dir / "toy_dataset.npz"
    generate_toy_dataset(ToyDatasetConfig(), dataset_path)

    input_dim = ToyDatasetConfig().image_size ** 2
    save_toy_model(output_dir / "toy_uniform_model.pkl", input_dim, seed=42)
    save_toy_model(output_dir / "toy_zipfian_model.pkl", input_dim, seed=1337)

    print("\nAssets ready!")
    print(f"- Dataset: {dataset_path}")
    print(f"- Uniform model: {output_dir / 'toy_uniform_model.pkl'}")
    print(f"- Zipfian model: {output_dir / 'toy_zipfian_model.pkl'}")
    print("\nUpdate your config file to point to these paths.")


if __name__ == "__main__":
    main()
