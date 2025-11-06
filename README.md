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

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/groundeep-analysis.git
cd groundeep-analysis

# Install dependencies
pip install -r requirements.txt

# Install package (editable mode for development)
pip install -e .
```

### Basic Usage

```python
from groundeep_analysis import AnalysisPipeline

# Load configuration
pipeline = AnalysisPipeline.from_yaml("config.yaml")

# Run all enabled stages
results = pipeline.run()

# Or run specific stages
results = pipeline.run(stages=["probes", "geometry", "dimensionality"])
```

### With Custom Models

```python
from groundeep_analysis.core import ModelManager, EmbeddingExtractor
from groundeep_analysis.adapters import create_adapter

# 1. Load your model
mm = ModelManager()
mm.load_model("path/to/model.pth", label="my_model")

# 2. Get adapter (auto-detected)
adapter = mm.get_adapter("my_model")
print(f"Using {adapter.__class__.__name__}")

# 3. Extract embeddings
extractor = EmbeddingExtractor(mm)
embeddings = extractor.extract("my_model", dataloader)

# 4. Run analysis
from groundeep_analysis.stages import LinearProbesStage
probe_stage = LinearProbesStage()
results = probe_stage.run(context, settings, output_dir)
```

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

### Example 1: Analyze Pre-trained VAE

```python
from groundeep_analysis import AnalysisPipeline
import torch

# Your VAE
class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ...
        self.decoder = ...

vae = VAE()
vae.load_state_dict(torch.load("vae.pth"))

# Auto-detected as VAE, analyzed automatically
pipeline = AnalysisPipeline(
    model=vae,
    dataset_path="data/",
    config="config.yaml"
)
pipeline.run()
```

### Example 2: Compare ResNet Layers

```python
from groundeep_analysis.adapters import PyTorchAdapter
import torchvision.models as models

resnet = models.resnet50(pretrained=True)

# Extract from multiple layers
adapter = PyTorchAdapter(resnet)
layer_embeddings = adapter.encode_layerwise(
    images,
    layers=["layer1", "layer2", "layer3", "layer4"]
)

# Run geometry analysis on each layer
for i, emb in enumerate(layer_embeddings):
    analyze_geometry(emb, output_dir=f"results/layer{i+1}")
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
  author = {Your Name},
  title = {GROUNDEEP Analysis: Model-Agnostic Deep Learning Analysis Framework},
  year = {2024},
  url = {https://github.com/yourusername/groundeep-analysis}
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

- **Issues**: [GitHub Issues](https://github.com/yourusername/groundeep-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/groundeep-analysis/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the deep learning community

</div>
