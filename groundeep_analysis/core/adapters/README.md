# Adapter System - Technical Documentation

## Overview

The adapter system provides a **uniform interface** for working with different neural network architectures in the GROUNDEEP analysis pipeline.

### Why Adapters?

**Problem:** Different model types have different interfaces:
- DBN: `.layers[i].forward(x)` and `.backward(x)`
- VAE: `.encoder(x)` and `.decoder(z)`
- CNN: `model(x)` with hooks for layer extraction
- Transformer: Different again!

**Solution:** Adapters wrap all models with a standard interface:
```python
adapter.encode(x)           # Always works, regardless of model type
adapter.decode(z)           # If reconstruction is supported
adapter.encode_layerwise(x) # If multi-layer extraction is supported
```

---

## Architecture

```
BaseAdapter (abstract)
├── encode(x) [REQUIRED]
├── decode(z) [OPTIONAL]
├── encode_layerwise(x) [OPTIONAL]
├── get_device() [default impl]
├── get_num_layers() [default impl]
└── get_metadata() [default impl]

Concrete Adapters:
├── DBNAdapter (for iDBN/gDBN)
├── VAEAdapter (for VAE models)
└── PyTorchAdapter (universal fallback)
```

---

## BaseAdapter API

### Required Methods

#### `encode(x: Tensor) -> Tensor`
Extract top-level embedding from input.

**This is the ONLY required method.**

```python
def encode(self, x):
    x = self.prepare_input(x)  # Device + preprocessing (provided by base)
    return self.model(x)  # Your model-specific code
```

### Optional Methods (with Defaults)

#### `decode(z: Tensor, from_layer: int = None) -> Tensor`
Reconstruct input from embedding.

**Default:** Raises `NotImplementedError` (for non-generative models)

```python
def decode(self, z, from_layer=None):
    return self.model.decoder(z)
```

#### `encode_layerwise(x: Tensor, layers: List[int] = None) -> List[Tensor]`
Extract embeddings at multiple layers.

**Default:** Returns `[self.encode(x)]` (single top-level embedding)

```python
def encode_layerwise(self, x, layers=None):
    if layers is None:
        layers = range(1, self.get_num_layers() + 1)

    embeddings = []
    for i in layers:
        emb = self._extract_layer(x, i)
        embeddings.append(emb)
    return embeddings
```

#### `get_device() -> torch.device`
Get device where model parameters are stored.

**Default:** Uses `model.parameters()` (works for nn.Module)

```python
def get_device(self):
    return next(self.model.parameters()).device
```

#### `get_num_layers() -> int`
Get number of extractable layers.

**Default:** Returns `1` (only top-level)

```python
def get_num_layers(self):
    return len(self.model.layers)  # Or however you count layers
```

#### `get_metadata() -> Dict[str, Any]`
Get adapter/model metadata for logging.

**Default:** Returns basic info (adapter type, device, capabilities)

```python
def get_metadata(self):
    meta = super().get_metadata()  # Get defaults
    meta["latent_dim"] = self.model.latent_dim  # Add custom info
    return meta
```

### Utility Methods (Don't Override)

#### `prepare_input(x: Tensor) -> Tensor`
Preprocess input (device transfer + dtype conversion).

```python
x = adapter.prepare_input(x)  # Moves to device, converts to float32
```

#### `supports_reconstruction() -> bool`
Check if model supports reconstruction.

```python
if adapter.supports_reconstruction():
    reconstructed = adapter.decode(z)
```

#### `supports_layerwise_extraction() -> bool`
Check if model supports layer-wise extraction.

```python
if adapter.supports_layerwise_extraction():
    layers = adapter.encode_layerwise(x)
```

#### `validate() -> Dict[str, bool]`
Run validation checks on adapter implementation.

```python
results = adapter.validate()
print(results)  # {"encode": True, "decode": False, ...}
```

---

## Creating Custom Adapters

### Minimal Example

```python
from pipeline_refactored.core.adapters import BaseAdapter

class MinimalAdapter(BaseAdapter):
    """Simplest possible adapter - only encode() required."""

    def encode(self, x):
        x = self.prepare_input(x)  # Handle device automatically
        return self.model(x)
```

### Full Example

```python
from pipeline_refactored.core.adapters import BaseAdapter
import torch
from typing import List, Optional

class MyModelAdapter(BaseAdapter):
    """Adapter for my custom model architecture."""

    def __init__(self, model, config=None):
        super().__init__(model)
        self.config = config or {}

    # REQUIRED
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract top-level features."""
        x = self.prepare_input(x)
        return self.model.encoder(x)

    # OPTIONAL: Reconstruction
    def decode(self, z: torch.Tensor, from_layer: Optional[int] = None) -> torch.Tensor:
        """Reconstruct from latent code."""
        if from_layer is not None:
            raise NotImplementedError("Partial decoding not supported")
        return self.model.decoder(z)

    # OPTIONAL: Layer-wise extraction
    def encode_layerwise(self, x: torch.Tensor, layers: Optional[List[int]] = None) -> List[torch.Tensor]:
        """Extract from intermediate layers."""
        x = self.prepare_input(x)

        if layers is None:
            layers = list(range(1, self.get_num_layers() + 1))

        activations = []
        for i in layers:
            act = self.model.get_layer_activation(i - 1, x)  # 0-indexed internally
            activations.append(act)

        return activations

    # OPTIONAL: Device inference (if model has custom device handling)
    def get_device(self) -> torch.device:
        """Get model device."""
        if hasattr(self.model, 'device'):
            return self.model.device
        return super().get_device()  # Use default

    # OPTIONAL: Layer count
    def get_num_layers(self) -> int:
        """Number of extractable layers."""
        return len(self.model.encoder.layers)

    # OPTIONAL: Metadata
    def get_metadata(self) -> dict:
        """Extended metadata."""
        meta = super().get_metadata()
        meta.update({
            "latent_dim": self.model.latent_dim,
            "input_shape": self.model.input_shape,
            "config": self.config,
        })
        return meta
```

---

## Adapter Registration

### Auto-Detection

Add your adapter to the factory in `__init__.py`:

```python
# pipeline_refactored/core/adapters/__init__.py

from .my_model_adapter import MyModelAdapter

def _auto_detect_adapter(model, **kwargs):
    # ... existing checks ...

    # Add your detection logic
    if hasattr(model, "my_special_attribute"):
        print("[Adapter] Auto-detected MyModel")
        return MyModelAdapter(model, **kwargs)

    # ... rest of code ...
```

### Manual Registration

Users can always specify your adapter explicitly:

```python
from my_adapters import MyModelAdapter
from pipeline_refactored.core.adapters import create_adapter

adapter = create_adapter(model, adapter_type=MyModelAdapter)
```

---

## Testing Your Adapter

### Unit Test Template

```python
import torch
from my_adapters import MyModelAdapter

def test_my_adapter():
    # Load your model
    model = MyModel()

    # Create adapter
    adapter = MyModelAdapter(model)

    # Test basic functionality
    dummy_input = torch.randn(4, 3, 64, 64)

    # 1. Test encode
    embeddings = adapter.encode(dummy_input)
    assert embeddings.shape[0] == 4, "Batch size mismatch"
    print(f"✓ Encode: {dummy_input.shape} → {embeddings.shape}")

    # 2. Test decode (if supported)
    if adapter.supports_reconstruction():
        reconstructed = adapter.decode(embeddings)
        assert reconstructed.shape == dummy_input.shape
        print(f"✓ Decode: {embeddings.shape} → {reconstructed.shape}")

    # 3. Test layerwise (if supported)
    if adapter.supports_layerwise_extraction():
        layers = adapter.encode_layerwise(dummy_input)
        print(f"✓ Layerwise: extracted {len(layers)} layers")

    # 4. Test metadata
    metadata = adapter.get_metadata()
    print(f"✓ Metadata: {metadata}")

    # 5. Run validation
    results = adapter.validate()
    assert results["encode"], "Encode validation failed"
    print("✓ Validation passed")

if __name__ == "__main__":
    test_my_adapter()
```

### Integration Test

Use the full pipeline test:

```python
# test_my_model_integration.py
from pipeline_refactored.core import ModelManager, EmbeddingExtractor

mm = ModelManager(adapter_type="auto")  # Will use your adapter if registered
mm.load_model("path/to/model.pth", label="test")

adapter = mm.get_adapter("test")
print(adapter)  # Should be MyModelAdapter instance
```

---

## Best Practices

### 1. Keep adapters simple
```python
# Good: Adapter delegates to model
def encode(self, x):
    x = self.prepare_input(x)
    return self.model.encode(x)

# Bad: Adapter reimplements model logic
def encode(self, x):
    x = self.preprocess(x)
    for layer in self.model.layers:
        x = layer(x)
    return x
```

### 2. Use base class utilities
```python
# Good: Use provided utilities
def encode(self, x):
    x = self.prepare_input(x)  # ← Handles device + dtype
    return self.model(x)

# Bad: Reimplement device handling
def encode(self, x):
    device = next(self.model.parameters()).device
    x = x.to(device).float()
    return self.model(x)
```

### 3. Graceful degradation
```python
# Good: Optional features raise NotImplementedError
def decode(self, z):
    raise NotImplementedError("Model does not support reconstruction")

# Bad: Silent failure or crash
def decode(self, z):
    return None  # ← Pipeline will crash later
```

### 4. Document requirements
```python
class MyAdapter(BaseAdapter):
    """
    Adapter for MyModel architecture.

    Requirements:
    - Model must have .encoder and .decoder attributes
    - Encoder output must be (batch, latent_dim) tensor
    - Decoder input must match encoder output shape

    Example:
        >>> model = MyModel(input_dim=784, latent_dim=128)
        >>> adapter = MyAdapter(model)
        >>> z = adapter.encode(x)
    """
```

---

## Common Patterns

### Pattern 1: Encoder-Decoder Models

```python
class EncoderDecoderAdapter(BaseAdapter):
    def encode(self, x):
        x = self.prepare_input(x)
        return self.model.encoder(x)

    def decode(self, z):
        return self.model.decoder(z)
```

### Pattern 2: Sequential Models

```python
class SequentialAdapter(BaseAdapter):
    def encode(self, x):
        x = self.prepare_input(x)
        for layer in self.model.layers:
            x = layer(x)
        return x

    def encode_layerwise(self, x, layers=None):
        if layers is None:
            layers = range(1, len(self.model.layers) + 1)

        x = self.prepare_input(x)
        activations = []

        for i, layer in enumerate(self.model.layers, 1):
            x = layer(x)
            if i in layers:
                activations.append(x.detach())

        return activations
```

### Pattern 3: Hook-Based Extraction

```python
class HookAdapter(BaseAdapter):
    def encode_layerwise(self, x, layers=None):
        x = self.prepare_input(x)

        if layers is None:
            layers = list(dict(self.model.named_modules()).keys())

        activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                activations[name] = output.detach()
            return hook_fn

        # Register hooks
        for name, module in self.model.named_modules():
            if name in layers:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        # Forward pass
        self.model(x)

        # Cleanup
        for h in hooks:
            h.remove()

        return [activations[name] for name in layers if name in activations]
```

---

## Troubleshooting

### Device Mismatch Errors

```python
# Problem: Input and model on different devices
RuntimeError: Expected all tensors to be on the same device

# Solution: Use prepare_input()
def encode(self, x):
    x = self.prepare_input(x)  # ← Automatically moves to model device
    return self.model(x)
```

### Shape Mismatch in Layerwise

```python
# Problem: Layers return different shapes
# Layer 1: [batch, channels, h, w]
# Layer 2: [batch, features]

# Solution: Flatten consistently
def encode_layerwise(self, x, layers=None):
    activations = []
    for emb in self._extract_raw(x, layers):
        if emb.dim() > 2:
            emb = emb.reshape(emb.size(0), -1)  # Flatten spatial dims
        activations.append(emb)
    return activations
```

### Reconstruction Not Supported

```python
# Problem: Analysis stage tries to reconstruct but model doesn't support it

# Solution: Check capability before using
if adapter.supports_reconstruction():
    reconstructed = adapter.decode(z)
else:
    print("Skipping reconstruction (not supported)")
```

---

## FAQ

**Q: Do I need to implement all methods?**
A: No! Only `encode()` is required. Implement others only if your model supports them.

**Q: Can I have multiple adapters for the same model?**
A: Yes! For example, one for top-level features, another for specific layer extraction.

**Q: How do I handle models with text + image inputs?**
A: Either:
1. Create separate adapters for each modality
2. Create one adapter that handles both via `encode(x, modality="image")`

**Q: Can adapters have state?**
A: Yes, but be careful with caching. Use `self._cache` for mutable state.

**Q: What if my model doesn't fit any existing pattern?**
A: Create a completely custom adapter! The base class is just a protocol, not a straitjacket.

---

## Examples in Codebase

- **DBNAdapter**: `dbn_adapter.py` - Sequential RBM stack
- **VAEAdapter**: `vae_adapter.py` - Encoder-decoder with sampling
- **PyTorchAdapter**: `pytorch_adapter.py` - Hook-based universal adapter

---

**End of Technical Documentation**

For usage guide, see: `/PIPELINE_USAGE_GUIDE.md`
