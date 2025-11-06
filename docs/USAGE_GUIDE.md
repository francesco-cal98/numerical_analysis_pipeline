---
title: GROUNDEEP Analysis – Usage Guide
---

# GROUNDEEP Analysis – Usage Guide

This document explains how to run the public pipeline end-to-end, how to reuse it inside the private **Groundeep** workspace, and which optional dependencies unlock the advanced analysis stages.

> **TL;DR**  
> 1. Install the package and dependencies  
> 2. Generate the toy assets (`python examples/bootstrap_assets.py`)  
> 3. Run the pipeline (`python -m groundeep_analysis.cli.run_pipeline --config examples/configs/sample_analysis.yaml`)  
> 4. Point the config to your real datasets/models when you are ready

---

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional extras (stage-specific):

| Feature / Stage          | Extra dependency           |
|-------------------------|----------------------------|
| Geometry (FDR)          | `statsmodels`              |
| SSIM reconstruction     | `scikit-image`             |
| Behavioral suite        | Private data + torch CUDA  |
| Weights & Biases logging| `wandb`                    |

Install on demand, e.g. `pip install statsmodels scikit-image wandb`.

---

## 2. Generate Demo Assets

To try the pipeline without touching private data, generate the synthetic dataset and toy models:

```bash
python examples/bootstrap_assets.py
```

This creates:

- `examples/assets/toy_dataset.npz`
- `examples/assets/toy_uniform_model.pkl`
- `examples/assets/toy_zipfian_model.pkl`

These assets mimic the numerosity dataset layout expected by `UniformDataset` and two tiny PyTorch encoders wrapped by the adapter system.

---

## 3. Run the Pipeline (CLI)

Use the provided sample config:

```bash
python -m groundeep_analysis.cli.run_pipeline \
    --config examples/configs/sample_analysis.yaml
```

Outputs land under `examples/results/uniform/toy_mlp/`:

- `powerfit_pairs/` – power-law parameters and plots
- `probes/` – confusion matrices + CSV summaries
- `feature_analysis/` – Kendall correlations
- `label_histograms/`

Only the `powerlaw` and `probing` stages are enabled in the demo configuration. Enable more stages by flipping the corresponding `enabled` flags once you have the dependencies/data required by each module.

**Helpful flags**

```bash
python -m groundeep_analysis.cli.run_pipeline \
    --config my_config.yaml \
    --output-root /tmp/results \
    --seed 99 \
    --use-wandb \
    --wandb-project numerosity-demo
```

---

## 4. Configuration Anatomy

```yaml
# Global settings
seed: 42
output_root: ../results
use_wandb: false
layers: top  # 'top' | 'all' | [1, 2, 3]

# Stage toggles
probing:
  enabled: true
  n_bins: 5
  steps: 1000
  patience: 20

rsa: {enabled: false}
...

# Model definitions
models:
  - arch: iDBN_1500_500
    distribution: uniform          # Controls logging + bundle selection
    dataset_path: /path/to/dataset # Folder containing .npz
    dataset_name: stimuli.npz
    model_uniform: /path/uniform_model.pkl
    model_zipfian: /path/zipf_model.pkl
    val_size: 0.1
```

Tips:

- `dataset_path` and `model_*` values can be relative; they are resolved against the directory containing the config file.
- All stages are optional. Leave `enabled: false` (or omit the section) to skip a stage.
- The behavioral suite expects multiple pickle/mat files – keep it disabled unless you have the private datasets.

---

## 5. Bring Groundeep Assets Into This Repo

If you want the public repository to run the full Groundeep analysis without touching the private tree, mirror the required datasets and models locally.

1. **Prepare local copies (or symlinks)**
   ```bash
   # From groundeep-analysis/
   python scripts/prepare_groundeep_assets.py \
       --config ../Groundeep/src/configs/analysis.yaml \
       --source ../Groundeep \
       --dest   local_assets
   ```
   - Use `--mode link` to create symlinks instead of copying gigabytes of data.
   - Add `--dry-run` to preview what would be copied.
2. **Use the ready-made config**  
   `examples/configs/groundeep_analysis.yaml` already points to `../local_assets/...`.
3. **Run the full pipeline**
   ```bash
   python -m groundeep_analysis.cli.run_pipeline \
       --config examples/configs/groundeep_analysis.yaml
   ```

The directory `local_assets/` is ignored by git, so large files never leak into commits.

---

## 6. Usage Scenarios

### CLI – Toy assets
```bash
python -m groundeep_analysis.cli.run_pipeline \
    --config examples/configs/sample_analysis.yaml
```

### CLI – Groundeep assets
```bash
python scripts/prepare_groundeep_assets.py --mode link   # if not already run
python -m groundeep_analysis.cli.run_pipeline \
    --config examples/configs/groundeep_analysis.yaml \
    --output-root ../results/groundeep_run
```

### Programmatic – Custom stage orchestration
```python
from pathlib import Path
import yaml

from groundeep_analysis.core.analysis_types import ModelSpec, AnalysisSettings
from groundeep_analysis.pipeline import run_analysis_pipeline

cfg_path = Path("examples/configs/groundeep_analysis.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
settings = AnalysisSettings.from_cfg(cfg)

for model_cfg in cfg["models"]:
    spec = ModelSpec.from_config(model_cfg, cfg_path.parent)
    run_analysis_pipeline(spec, settings, output_root=Path("results/manual"))
```

---

## 7. Verification Checklist

After setting everything up, run through this list to confirm the environment is healthy:

1. `python -m groundeep_analysis.cli.run_pipeline --config examples/configs/sample_analysis.yaml`
   - Expect `examples/results/uniform/toy_mlp/powerfit_pairs/fit_linear_*.png` and a `probes/` directory.
2. `python scripts/prepare_groundeep_assets.py --dry-run`  
   - Should list every dataset/model referenced in the Groundeep config without errors.
3. `python -m groundeep_analysis.cli.run_pipeline --config examples/configs/groundeep_analysis.yaml`
   - Each stage reports progress; check for folders like `.../behavioral`, `.../geometry/layers/`.
4. Spot-check outputs: open a probe CSV, RSA plot, reconstruction heatmap to ensure they’re populated.
5. Optional: run key stages individually to isolate issues, e.g. disable everything except `probing` to verify data loading.

If any step fails, refer to the troubleshooting table below.

---

## 8. Optional Stages & Dependencies

| Stage / Feature               | Notes                                                                 |
|-------------------------------|-----------------------------------------------------------------------|
| Geometry (RSA/RDM/monotonicity)| Requires `statsmodels` for FDR correction (fallback prints warning)  |
| Reconstruction (SSIM)         | Needs `scikit-image`; otherwise stage aborts with a helpful error     |
| Behavioral suite              | Needs private pickle/mat datasets + CUDA-capable DBN (or CPU fallback)|
| Weights & Biases              | Install `wandb` and pass `--use-wandb` + project name                 |

Stages automatically print a warning instead of crashing if an optional dependency is missing.

---

## 9. Troubleshooting

| Symptom                                   | Fix                                                                 |
|-------------------------------------------|----------------------------------------------------------------------|
| `ModuleNotFoundError: statsmodels`        | `pip install statsmodels` or keep geometry stage disabled            |
| `ModuleNotFoundError: skimage`            | `pip install scikit-image` or disable SSIM                          |
| `A load persistent id instruction...`     | Ensure models are pickled modules (use the provided asset generator) |
| CUDA warnings on CPU machines             | Safe to ignore; PyTorch falls back to CPU                           |
| Behavioral import errors                  | Stage relies on private helpers – keep `behavioral.enabled = false` |

---

## 10. Next Steps

- Replace toy assets with your real dataset/models in the config.
- Enable additional stages one by one, validating outputs.
- Contribute adapters or stages via pull requests (see `CONTRIBUTING.md` if available).
- Share configs/results with students – they only need the public repo + instructions above.

Happy analyzing!
