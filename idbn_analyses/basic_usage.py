"""
Basic usage example for GROUNDEEP Analysis Pipeline.

This script demonstrates how to:
1. Load a model with the adapter system
2. Extract embeddings
3. Run analysis stages
"""

import torch
from pathlib import Path

# Import core components
from groundeep_analysis.core import ModelManager, DatasetManager, EmbeddingExtractor
from groundeep_analysis.core.adapters import create_adapter
from groundeep_analysis.stages import LinearProbesStage, GeometryStage


def example_with_dbn():
    """Example: Analyze a DBN/iDBN model."""

    print("=== DBN Analysis Example ===\n")

    # 1. Load model
    print("1. Loading DBN model...")
    mm = ModelManager(adapter_type="auto")  # Auto-detect model type
    mm.load_model("path/to/dbn_model.pkl", label="my_dbn")

    # 2. Get adapter
    adapter = mm.get_adapter("my_dbn")
    print(f"   Adapter type: {type(adapter).__name__}")
    print(f"   Device: {adapter.get_device()}")
    print(f"   Supports reconstruction: {adapter.supports_reconstruction()}")

    # 3. Load dataset
    print("\n2. Loading dataset...")
    dm = DatasetManager("path/to/data", "dataset.npz")
    val_loader = dm.get_dataloader("uniform", split="val", batch_size=32)

    # 4. Extract embeddings
    print("\n3. Extracting embeddings...")
    extractor = EmbeddingExtractor(mm, use_adapters=True)
    embeddings = extractor.extract("my_dbn", val_loader, layer="top")
    print(f"   Embeddings shape: {embeddings.shape}")

    # 5. Run analysis stages
    print("\n4. Running analysis stages...")

    # Example: Linear probes
    from groundeep_analysis.core.context import AnalysisContext
    ctx = AnalysisContext(
        model_manager=mm,
        dataset_manager=dm,
        embeddings={"top": embeddings},
        distribution="uniform"
    )

    output_dir = Path("results/dbn_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run probing stage
    probe_stage = LinearProbesStage()
    probe_settings = {
        "enabled": True,
        "layers": ["top"],
        "n_bins": 5,
        "steps": 1000,
        "lr": 0.01
    }
    probe_stage.run(ctx, probe_settings, output_dir)

    print("\n✓ Analysis complete!")


def example_with_vae():
    """Example: Analyze a VAE model."""

    print("\n=== VAE Analysis Example ===\n")

    # 1. Load VAE
    print("1. Loading VAE model...")
    mm = ModelManager(adapter_type="vae")
    mm.load_model("path/to/vae_model.pth", label="my_vae")

    # 2. Get adapter with custom config
    adapter = mm.get_adapter("my_vae")
    print(f"   Adapter type: {type(adapter).__name__}")

    # 3. Test encoding/decoding
    print("\n2. Testing encode/decode...")
    dummy_input = torch.randn(4, 1, 28, 28)  # MNIST-like

    # Encode
    z = adapter.encode(dummy_input)
    print(f"   Encoded: {dummy_input.shape} -> {z.shape}")

    # Decode
    if adapter.supports_reconstruction():
        reconstructed = adapter.decode(z)
        print(f"   Decoded: {z.shape} -> {reconstructed.shape}")

    print("\n✓ VAE adapter working!")


def example_with_pytorch_model():
    """Example: Analyze a generic PyTorch model (e.g., ResNet)."""

    print("\n=== PyTorch Model Example ===\n")

    # 1. Create a simple CNN
    print("1. Creating PyTorch model...")
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleCNN()

    # 2. Create adapter
    print("\n2. Creating adapter...")
    from groundeep_analysis.core.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(
        model,
        output_layer="features",  # Extract from features layer
        flatten_output=True
    )

    # 3. Test extraction
    print("\n3. Testing feature extraction...")
    dummy_input = torch.randn(4, 3, 32, 32)
    features = adapter.encode(dummy_input)
    print(f"   Extracted features: {dummy_input.shape} -> {features.shape}")

    # 4. List available layers
    print("\n4. Available layers for extraction:")
    for layer_name in adapter.list_layers()[:10]:
        print(f"   - {layer_name}")

    print("\n✓ PyTorch adapter working!")


def example_adapter_validation():
    """Example: Validate an adapter implementation."""

    print("\n=== Adapter Validation Example ===\n")

    # Create a model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Create adapter
    adapter = create_adapter(model, adapter_type="pytorch")

    # Validate
    from groundeep_analysis.core.adapters import validate_adapter
    is_valid = validate_adapter(adapter, verbose=True)

    if is_valid:
        print("\n✓ Adapter is valid!")
    else:
        print("\n✗ Adapter has issues")


if __name__ == "__main__":
    print("GROUNDEEP Analysis - Usage Examples")
    print("=" * 50)

    # Run examples
    # NOTE: Update paths before running!

    # example_with_dbn()
    # example_with_vae()
    example_with_pytorch_model()
    example_adapter_validation()

    print("\n" + "=" * 50)
    print("For more examples, see documentation:")
    print("https://github.com/yourusername/groundeep-analysis")
