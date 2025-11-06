"""
GROUNDEEP Analysis - Model-Agnostic Deep Learning Analysis Framework

A comprehensive pipeline for analyzing neural network representations with
support for multiple model architectures through an adapter system.

Quick Start:
    >>> from groundeep_analysis.core import ModelManager, DatasetManager, EmbeddingExtractor
    >>> from groundeep_analysis.core.adapters import create_adapter
    >>>
    >>> # Load your model
    >>> mm = ModelManager()
    >>> mm.load_model("path/to/model.pth", label="my_model")
    >>>
    >>> # Get adapter
    >>> adapter = mm.get_adapter("my_model")
    >>>
    >>> # Extract embeddings
    >>> dm = DatasetManager("data_path", "dataset.npz")
    >>> loader = dm.get_dataloader("uniform", split="val")
    >>> extractor = EmbeddingExtractor(mm)
    >>> embeddings = extractor.extract("my_model", loader)

Main Components:
    - core.adapters: Model adapter system (DBN, VAE, PyTorch)
    - core.model_manager: Model loading and management
    - core.dataset_manager: Dataset handling
    - core.embedding_extractor: Feature extraction
    - stages: Analysis stages (probes, RSA, dimensionality, etc.)

For detailed documentation, see: https://github.com/yourusername/groundeep-analysis
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from groundeep_analysis.core.adapters import (
    BaseAdapter,
    DBNAdapter,
    VAEAdapter,
    PyTorchAdapter,
    create_adapter,
)

from groundeep_analysis.core.model_manager import ModelManager
from groundeep_analysis.core.dataset_manager import DatasetManager
from groundeep_analysis.core.embedding_extractor import EmbeddingExtractor

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Core components
    "ModelManager",
    "DatasetManager",
    "EmbeddingExtractor",

    # Adapters
    "BaseAdapter",
    "DBNAdapter",
    "VAEAdapter",
    "PyTorchAdapter",
    "create_adapter",
]
