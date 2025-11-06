# GROUNDEEP Analysis

<div align="center">

**Model-Agnostic Deep Learning Analysis Framework**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Comprehensive analysis toolkit for neural network representations with a unified adapter system supporting DBNs, VAEs, CNNs, and Transformers.*

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Roadmap](#-roadmap)

</div>

---

## 🎯 Overview

**GROUNDEEP Analysis** is a modular, extensible framework for analyzing neural network representations across different architectures. Originally developed for Deep Belief Networks studying numerosity representations, it now supports any PyTorch model through a flexible **adapter system**.

### Why GROUNDEEP Analysis?

- **🔄 Model-Agnostic**: Works with DBNs, VAEs, CNNs, Transformers via unified adapter interface
- **🧩 Modular Design**: 7+ analysis stages, each independently configurable
- **📊 Comprehensive**: From geometry (RSA, RDM) to behavioral tasks to dimensionality reduction
- **🚀 Production-Ready**: Type-safe, well-documented, extensively tested
- **🔌 Extensible**: Add custom analysis stages or model adapters in minutes

---

## ✨ Features

### Core Analysis Stages

| Stage | Description | Outputs |
|-------|-------------|---------|
| **Power-Law Scaling** | Fit power-law relationships between representations and features | Parameters, R², residuals |
| **Linear Probes** | Train linear classifiers on embeddings | Accuracy, confusion matrices |
| **Geometry** | RSA, RDM, monotonicity, partial correlations | Correlation matrices, heatmaps |
| **Reconstruction** | MSE, AFP, SSIM quality metrics | Per-bin quality scores, visualizations |
| **Dimensionality** | PCA, UMAP, t-SNE, PC traversals | 2D/3D plots, variance explained |
| **CKA** | Centered Kernel Alignment across models/layers | Similarity matrices, bootstrap stats |
| **Behavioral** | Numerosity comparison, estimation, fixed-reference tasks | Accuracy, Weber fractions, psychometric curves |

### Adapter System

```python
# Works with ANY model architecture
from groundeep_analysis.adapters import create_adapter

# Auto-detect model type
adapter = create_adapter(your_model)

# Universal interface
embeddings = adapter.encode(x)
reconstructed = adapter.decode(embeddings)
layer_activations = adapter.encode_layerwise(x)
```

**Built-in Adapters:**
- `DBNAdapter` - Deep Belief Networks, RBM stacks
- `VAEAdapter` - Variational Autoencoders
- `PyTorchAdapter` - Universal fallback for any `torch.nn.Module`

**Custom Adapters:** Implement `BaseAdapter` with just one required method (`encode`).

---

## 🚀 Quick Start

> Need the full walkthrough? See [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md).

### 1. Install

```bash
git clone https://github.com/francesco-cal98/groundeep-analysis.git
cd groundeep-analysis

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Bootstrap the demo assets (optional)

```bash
python examples/bootstrap_assets.py
```

Creates `examples/assets/toy_dataset.npz` and two toy models so you can run the pipeline without private data.

To reuse the **real** Groundeep datasets/models inside this repo:

```bash
python scripts/prepare_groundeep_assets.py \
    --config ../Groundeep/src/configs/analysis.yaml \
    --source ../Groundeep \
    --dest   local_assets
```

Then run the full pipeline with `examples/configs/groundeep_analysis.yaml`.

### 3. Run the pipeline (CLI)

```bash
python -m groundeep_analysis.cli.run_pipeline \
    --config examples/configs/sample_analysis.yaml
```

Outputs arrive under `examples/results/`.

### Programmatic usage

```python
from pathlib import Path
import yaml

from groundeep_analysis.core.analysis_types import ModelSpec, AnalysisSettings
from groundeep_analysis.pipeline import run_analysis_pipeline

cfg_path = Path("examples/configs/sample_analysis.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
settings = AnalysisSettings.from_cfg(cfg)

for model_cfg in cfg["models"]:
    spec = ModelSpec.from_config(model_cfg, cfg_path.parent)
    run_analysis_pipeline(spec, settings, output_root=Path("results"))
```

`ModelManager` and `EmbeddingExtractor` remain available for lower-level usage when integrating custom stages or adapters.

---

## 📚 Documentation

### Core Concepts

#### 1. **Adapters** - Unified Model Interface

Adapters wrap different model architectures with a standard interface:

```python
class YourAdapter(BaseAdapter):
    def encode(self, x):
        """Required: extract embeddings"""
        return self.model(x)

    def decode(self, z):
        """Optional: reconstruct input"""
        return self.model.decoder(z)

    def encode_layerwise(self, x, layers=None):
        """Optional: extract from multiple layers"""
        return [self.model.layer[i](x) for i in layers]
```

#### 2. **Stages** - Modular Analysis Components

Each stage is independently configurable:

```yaml
# config.yaml
stages:
  probes:
    enabled: true
    layers: [top]
    n_bins: 5
    steps: 1000

  rsa:
    enabled: true
    metric: cosine
    alpha: 0.01
```

#### 3. **Pipeline** - Orchestration

Manages data flow, model loading, and stage execution:

```
Config → DatasetManager → ModelManager → EmbeddingExtractor → Stages → Results
```

### Configuration

See [`examples/config_templates/`](examples/config_templates/) for complete examples:

- `minimal_config.yaml` - Bare minimum configuration
- `full_config.yaml` - All stages with detailed options
- `custom_model_config.yaml` - Using custom adapters

---

## �� Examples

### Example 1: Analyze a Pre-trained VAE

```python
from pathlib import Path
import torch

from groundeep_analysis.core.analysis_types import ModelSpec, AnalysisSettings
from groundeep_analysis.pipeline import run_analysis_pipeline

# Save your model (torch.save keeps the adapter auto-detection happy)
vae = torch.load("vae.pth")
torch.save(vae, "vae_model.pkl")

spec = ModelSpec(
    arch_name="vae_baseline",
    distribution="uniform",
    dataset_path=Path("data"),
    dataset_name="stimuli.npz",
    model_uniform=Path("vae_model.pkl"),
    model_zipfian=Path("vae_model.pkl"),
    val_size=0.1,
)

settings = AnalysisSettings.from_cfg(
    {
        "probing": {"enabled": True, "layers": ["top"], "n_bins": 5, "steps": 500},
        "rsa": {"enabled": False},
        "rdm": {"enabled": False},
    }
)

run_analysis_pipeline(spec, settings, output_root=Path("results"))
```

### Example 2: Compare ResNet Layers Programmatically

```python
import torchvision.models as models
from groundeep_analysis.adapters import PyTorchAdapter

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Extract activations from multiple blocks
adapter = PyTorchAdapter(resnet, flatten_output=True)
features = adapter.encode_layerwise(
    images, layers=["layer1", "layer2", "layer3", "layer4"]
)

for name, tensor in zip(["layer1", "layer2", "layer3", "layer4"], features):
    print(name, tensor.shape)
```

### Example 3: Custom Behavioral Task

```python
from groundeep_analysis.stages import BaseStage

class CustomTask(BaseStage):
    name = "custom_task"

    def run(self, context, settings, output_dir):
        embeddings = context.embeddings["uniform"]
        labels = context.features["labels"]

        # Your custom analysis
        accuracy = run_my_task(embeddings, labels)

        # Save results
        results = {"accuracy": accuracy}
        self.save_results(results, output_dir / "custom_task.json")
        return results

# Register and use
pipeline.register_stage(CustomTask())
pipeline.run()
```

---

## 🏗️ Architecture

```
groundeep_analysis/
├── core/
│   ├── adapters/           # Model adapter system
│   │   ├── base.py        # BaseAdapter protocol
│   │   ├── dbn_adapter.py
│   │   ├── vae_adapter.py
│   │   └── pytorch_adapter.py
│   ├── model_manager.py    # Model loading + adapter creation
│   ├── dataset_manager.py  # Data loading utilities
│   ├── embedding_extractor.py  # Embedding extraction
│   └── context.py          # Analysis context management
├── stages/                 # Analysis stages
│   ├── powerlaw.py
│   ├── probes.py
│   ├── geometry.py
│   ├── reconstruction.py
│   ├── dimensionality.py
│   ├── cka.py
│   └── behavioral.py
├── utils/                  # Helper functions
└── examples/               # Usage examples
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test adapters specifically
pytest tests/test_adapters.py -v

# Test with your model
python tests/test_custom_model.py --model path/to/model.pth
```

---

## 🗺️ Roadmap

### ✅ Current (v0.1.0)
- [x] Adapter system for DBN, VAE, PyTorch models
- [x] 7 analysis stages (probes, geometry, dimensionality, etc.)
- [x] YAML configuration system
- [x] Comprehensive documentation

### 🚧 In Progress
- [ ] **Training Pipeline** - Modular training framework with callbacks (Q4 2024)
- [ ] **Multimodal iDBN** - Joint image-label analysis tools (Q4 2024)
- [ ] PyPI package (`pip install groundeep-analysis`)

### 🔮 Planned
- [ ] Hugging Face Transformers integration
- [ ] W&B integration for experiment tracking
- [ ] Interactive visualization dashboard
- [ ] Pre-trained model zoo

---

## 📖 Citation

If you use GROUNDEEP Analysis in your research, please cite:

```bibtex
@software{groundeep_analysis2024,
  author = {Francesco Maria Calistroni},
  title = {GROUNDEEP Analysis: Model-Agnostic Deep Learning Analysis Framework},
  year = {2024},
  url = {https://github.com/francesco-cal98/groundeep-analysis}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas where we'd love help:

- **New Adapters**: Support for more architectures (Diffusion models, Graph NNs, etc.)
- **New Stages**: Additional analysis methods
- **Documentation**: Tutorials, examples, translations
- **Testing**: Edge cases, integration tests

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License - see [`LICENSE`](LICENSE) for details.

---

## 🔗 Related Projects

**GROUNDEEP Suite** (same author):
- **GROUNDEEP** (main repo) - Training pipeline for Deep Belief Networks
- **GROUNDEEP Analysis** (this repo) - Model-agnostic analysis framework
- **GROUNDEEP Multimodal** *(coming soon)* - Joint image-label modeling

**Built With:**
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) - Numerical computing
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities
- [UMAP](https://umap-learn.readthedocs.io/) - Dimensionality reduction
- [Hydra](https://hydra.cc/) - Configuration management

---

## 💬 Contact

- **Issues**: [GitHub Issues](https://github.com/francesco-cal98/groundeep-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/francesco-cal98/groundeep-analysis/discussions)
- **Email**: fra.calistroni@gmail.com

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the deep learning community

</div>
