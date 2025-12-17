"""
Adapter for Bimodal DBN (iMDBN_BiModal).

Supports two modality configurations:
- labels: Numerosity images + one-hot encoded labels (32 or 1568 dim)
- mnist100: Numerosity images + MNIST-100 images (28x28)
"""

from .base import BaseAdapter
import torch
from typing import List, Optional, Dict, Any, Tuple

__all__ = ["BimodalDBNAdapter"]


class BimodalDBNAdapter(BaseAdapter):
    """
    Adapter for Bimodal DBN with automatic modality detection.

    The model consists of:
    - mod1_dbn: First modality encoder (numerosity images)
    - mod2_dbn: Second modality encoder (labels OR mnist100)
    - joint_rbm: Joint layer combining both modalities

    Attributes:
        mod1_dbn: First modality DBN (numerosity)
        mod2_dbn: Second modality DBN (labels/mnist100)
        joint_rbm: Joint RBM layer
        modality_type: "labels" or "mnist100"
        z1_dim: Dimension of mod1 latent space
        z2_dim: Dimension of mod2 latent space

    Example:
        >>> # Load model
        >>> model_data = torch.load("bimodal_mnist100.pkl")
        >>> adapter = BimodalDBNAdapter(model_data)
        >>>
        >>> # Check modality type
        >>> print(adapter.modality_type)  # "mnist100"
        >>>
        >>> # Encode based on modality
        >>> if adapter.modality_type == "labels":
        >>>     batch = (numerosity_imgs, labels_onehot)
        >>> else:
        >>>     batch = (numerosity_imgs, mnist_imgs)
        >>> embeddings = adapter.encode(batch)
    """

    def __init__(self, model):
        """
        Initialize Bimodal DBN adapter with auto-detection.

        Args:
            model: Model dict with keys:
                   - "mod1_dbn": First modality DBN
                   - "mod2_dbn": Second modality DBN
                   - "joint_rbm": Joint RBM
                   - "metadata": Dict with "modality_type" key
                   - Optional: "params", "architecture", etc.

        Raises:
            ValueError: If required components missing or modality_type unknown
        """
        super().__init__(model)

        # Extract components
        if isinstance(model, dict):
            self.mod1_dbn = model.get("mod1_dbn")
            self.mod2_dbn = model.get("mod2_dbn")
            self.joint_rbm = model.get("joint_rbm")
            self.params = model.get("params", {})
            self.metadata = model.get("metadata", {})

            # Auto-detect modality type
            self.modality_type = self.metadata.get("modality_type")
            if self.modality_type is None:
                # Fallback: infer from mod2_dbn architecture
                self.modality_type = self._infer_modality_type()
        else:
            # Object-based model
            self.mod1_dbn = getattr(model, "mod1_dbn", None)
            self.mod2_dbn = getattr(model, "mod2_dbn", None)
            self.joint_rbm = getattr(model, "joint_rbm", None)
            self.params = getattr(model, "params", {})
            self.metadata = getattr(model, "metadata", {})
            self.modality_type = self.metadata.get("modality_type") or self._infer_modality_type()

        # Validate components
        if self.mod1_dbn is None:
            raise ValueError("BimodalDBNAdapter requires 'mod1_dbn' component")
        if self.mod2_dbn is None:
            raise ValueError("BimodalDBNAdapter requires 'mod2_dbn' component")
        if self.joint_rbm is None:
            raise ValueError("BimodalDBNAdapter requires 'joint_rbm' component")
        if self.modality_type not in ("labels", "mnist100"):
            raise ValueError(
                f"Unknown modality_type: {self.modality_type}. "
                "Expected 'labels' or 'mnist100'."
            )

        # Extract dimensions
        self.z1_dim = self._get_dbn_output_dim(self.mod1_dbn)
        self.z2_dim = self._get_dbn_output_dim(self.mod2_dbn)

    def _infer_modality_type(self) -> str:
        """Infer modality type from mod2_dbn architecture."""
        if self.mod2_dbn is None:
            return "unknown"

        # Check first layer input size
        if hasattr(self.mod2_dbn, "layers") and len(self.mod2_dbn.layers) > 0:
            first_layer = self.mod2_dbn.layers[0]
            if hasattr(first_layer, "num_visible"):
                input_dim = first_layer.num_visible
                # MNIST-100: 28*28 = 784, plus some margin for variations
                if 700 <= input_dim <= 900:
                    return "mnist100"
                # Labels: typically 32 or 1568
                elif input_dim <= 100:
                    return "labels"

        # Default fallback
        return "labels"

    def _get_dbn_output_dim(self, dbn) -> int:
        """Get output dimension of a DBN."""
        if hasattr(dbn, "layers") and len(dbn.layers) > 0:
            last_layer = dbn.layers[-1]
            if hasattr(last_layer, "num_hidden"):
                return last_layer.num_hidden
            elif hasattr(last_layer, "W"):
                return last_layer.W.shape[1]
        return 0

    def encode(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Extract joint embeddings from both modalities.

        Args:
            batch: Tuple of (mod1_data, mod2_data)
                   - If modality_type == "labels":
                     mod1_data: [B, ...] numerosity images
                     mod2_data: [B, K] one-hot labels
                   - If modality_type == "mnist100":
                     mod1_data: [B, ...] numerosity images
                     mod2_data: [B, 1, 28, 28] MNIST images

        Returns:
            Joint embeddings [batch, joint_dim]

        Example:
            >>> if adapter.modality_type == "labels":
            >>>     batch = (num_imgs, labels_onehot)
            >>> else:
            >>>     batch = (num_imgs, mnist_imgs)
            >>> z_joint = adapter.encode(batch)
        """
        mod1_data, mod2_data = batch

        # Prepare inputs
        mod1_data = self.prepare_input(mod1_data)
        mod2_data = self.prepare_input(mod2_data)

        device = self.get_device()
        mod1_data = mod1_data.to(device).view(mod1_data.size(0), -1).float()

        # Handle mod2 based on modality type
        if self.modality_type == "labels":
            # Labels are already flat
            mod2_data = mod2_data.to(device).float()
        else:  # mnist100
            # Flatten MNIST images
            mod2_data = mod2_data.to(device).view(mod2_data.size(0), -1).float()

        with torch.no_grad():
            # Encode both modalities
            z1 = self.mod1_dbn.represent(mod1_data)
            z2 = self.mod2_dbn.represent(mod2_data)

            # Concatenate and pass through joint layer
            z_concat = torch.cat([z1, z2], dim=1)
            z_joint = self.joint_rbm.forward(z_concat)

        return z_joint

    def encode_mod1_only(self, mod1_data: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from first modality only (numerosity).

        Args:
            mod1_data: [batch, ...] numerosity images

        Returns:
            Mod1 embeddings [batch, z1_dim]
        """
        mod1_data = self.prepare_input(mod1_data)
        device = self.get_device()
        mod1_data = mod1_data.to(device).view(mod1_data.size(0), -1).float()

        with torch.no_grad():
            z1 = self.mod1_dbn.represent(mod1_data)

        return z1

    def encode_mod2_only(self, mod2_data: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from second modality only (labels/mnist100).

        Args:
            mod2_data: [batch, ...] second modality data

        Returns:
            Mod2 embeddings [batch, z2_dim]
        """
        mod2_data = self.prepare_input(mod2_data)
        device = self.get_device()

        if self.modality_type == "labels":
            mod2_data = mod2_data.to(device).float()
        else:
            mod2_data = mod2_data.to(device).view(mod2_data.size(0), -1).float()

        with torch.no_grad():
            z2 = self.mod2_dbn.represent(mod2_data)

        return z2

    def encode_layerwise(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        layers: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract embeddings at each layer.

        Args:
            batch: Tuple of (mod1_data, mod2_data)
            layers: Layer indices to extract. None = all layers.
                   Layers 1..N are mod1, N+1..N+M are mod2, N+M+1 is joint

        Returns:
            List of embeddings per layer
        """
        mod1_data, mod2_data = batch

        mod1_data = self.prepare_input(mod1_data)
        mod2_data = self.prepare_input(mod2_data)
        device = self.get_device()

        mod1_data = mod1_data.to(device).view(mod1_data.size(0), -1).float()
        if self.modality_type == "labels":
            mod2_data = mod2_data.to(device).float()
        else:
            mod2_data = mod2_data.to(device).view(mod2_data.size(0), -1).float()

        n_mod1_layers = len(self.mod1_dbn.layers)
        n_mod2_layers = len(self.mod2_dbn.layers)
        total_layers = n_mod1_layers + n_mod2_layers + 1  # +1 for joint

        if layers is None:
            layers = list(range(1, total_layers + 1))

        embeddings = []

        with torch.no_grad():
            # Mod1 layers
            cur1 = mod1_data
            for li, rbm in enumerate(self.mod1_dbn.layers, start=1):
                cur1 = rbm.forward(cur1)
                if li in layers:
                    embeddings.append(cur1.detach())

            # Mod2 layers
            cur2 = mod2_data
            for li, rbm in enumerate(self.mod2_dbn.layers, start=n_mod1_layers + 1):
                cur2 = rbm.forward(cur2)
                if li in layers:
                    embeddings.append(cur2.detach())

            # Joint layer
            joint_idx = total_layers
            if joint_idx in layers:
                z_concat = torch.cat([cur1, cur2], dim=1)
                z_joint = self.joint_rbm.forward(z_concat)
                embeddings.append(z_joint.detach())

        return embeddings

    def get_device(self) -> torch.device:
        """Infer device from first RBM's weights."""
        if self._device is not None:
            return self._device

        # Try mod1_dbn first
        if hasattr(self.mod1_dbn, "layers") and len(self.mod1_dbn.layers) > 0:
            first_rbm = self.mod1_dbn.layers[0]
            for attr in ("W", "weight", "hid_bias", "vis_bias"):
                param = getattr(first_rbm, attr, None)
                if isinstance(param, torch.Tensor):
                    self._device = param.device
                    return self._device

        # Fallback
        self._device = torch.device('cpu')
        return self._device

    def get_num_layers(self) -> int:
        """Return total number of layers (mod1 + mod2 + joint)."""
        n1 = len(self.mod1_dbn.layers) if hasattr(self.mod1_dbn, "layers") else 0
        n2 = len(self.mod2_dbn.layers) if hasattr(self.mod2_dbn, "layers") else 0
        return n1 + n2 + 1

    def get_metadata(self) -> Dict[str, Any]:
        """Extended metadata with bimodal-specific info."""
        meta = super().get_metadata()

        meta.update({
            "model_type": "BimodalDBN",
            "modality_type": self.modality_type,
            "z1_dim": self.z1_dim,
            "z2_dim": self.z2_dim,
            "num_mod1_layers": len(self.mod1_dbn.layers) if hasattr(self.mod1_dbn, "layers") else 0,
            "num_mod2_layers": len(self.mod2_dbn.layers) if hasattr(self.mod2_dbn, "layers") else 0,
            "joint_dim": self.joint_rbm.num_hidden if hasattr(self.joint_rbm, "num_hidden") else None,
        })

        if self.params:
            meta["training_params"] = {
                k: v for k, v in self.params.items()
                if k in ("LEARNING_RATE", "EPOCHS_MOD1", "EPOCHS_MOD2", "EPOCHS_JOINT")
            }

        return meta

    def __repr__(self) -> str:
        return (
            f"BimodalDBNAdapter("
            f"modality={self.modality_type}, "
            f"mod1_layers={len(self.mod1_dbn.layers) if hasattr(self.mod1_dbn, 'layers') else 0}, "
            f"mod2_layers={len(self.mod2_dbn.layers) if hasattr(self.mod2_dbn, 'layers') else 0}, "
            f"z1={self.z1_dim}, z2={self.z2_dim}, "
            f"device={self.get_device()})"
        )
