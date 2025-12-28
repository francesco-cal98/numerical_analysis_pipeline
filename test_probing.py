#!/usr/bin/env python3
"""Quick test script for multimodal probing."""

import pickle
import numpy as np
import torch
from pathlib import Path
from groundeep_analysis.core.probe_model import ProbeReadyModel
from groundeep_analysis.stages.probes.utils import log_linear_probe
from groundeep_analysis.core.dataset_manager import DatasetManager

# Load iMDBN model
print("Loading iMDBN model...")
model_path = '/home/student/Desktop/Groundeep/networks/uniform/imdbn/imdbn_trained_uniform.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Create dataset manager (multimodal=True)
print("Creating dataset manager...")
dm = DatasetManager(
    dataset_path='local_assets/stimuli_dataset_10_10',
    dataset_name='stimuli_dataset.npz',
    default_val_size=0.10,
    multimodal=True,  # Important for iMDBN!
)

# Get validation dataloader
val_loader = dm.get_dataloader('uniform', split='val', full_batch=False)

# Extract features
print("Extracting features...")
features = dm.get_features('uniform', split='val')

# Create ProbeReadyModel
print("Creating ProbeReadyModel...")
output_dir = Path('idbn_analyses/results/groundeep/uniform/iDBN_1500_1500_multimodal/probes_test')
output_dir.mkdir(parents=True, exist_ok=True)

prm = ProbeReadyModel(
    raw_model=model,
    val_loader=val_loader,
    features_dict=features,
    out_dir=output_dir,
    wandb_run=None,
)

print(f"✓ Model is iMDBN: {prm.is_imdbn}")
print(f"✓ Joint layer index: {prm.joint_layer_idx}")

# Run probing on joint layer (top)
print("\nRunning linear probes on joint layer...")
summary_rows = log_linear_probe(
    model=prm,
    epoch=0,
    n_bins=5,
    test_size=0.2,
    steps=1000,
    lr=0.01,
    rng_seed=42,
    patience=20,
    min_delta=0.0,
    save_csv=True,
    upto_layer=None,  # None = top = joint layer for iMDBN
    layer_tag="joint",
)

print("\n=== PROBING RESULTS ===")
for row in summary_rows:
    print(f"{row['metric']}: {row['accuracy']:.4f}")

print("\n✓ Probing completed successfully!")
print(f"Results saved to: {output_dir}")
