"""
Adapter for iMDBN (image Multimodal Deep Belief Network).

iMDBN consists of:
- image_idbn: Stack of RBMs for image processing
- joint_rbm: Joint RBM combining image latents + label one-hot vectors

This adapter extracts embeddings from the joint layer (multimodal representation).
"""

from .base import BaseAdapter
import torch
from typing import List, Optional, Dict, Any, Tuple


class iMDBNAdapter(BaseAdapter):
    """
    Adapter for iMDBN (image Multimodal Deep Belief Network).

    iMDBN is a multimodal model with:
    - image_idbn: Image encoder (stack of RBMs)
    - joint_rbm: Joint layer combining image latents + labels

    This adapter focuses on the joint multimodal embeddings.

    Attributes:
        image_idbn: Image DBN component
        joint_rbm: Joint RBM component
        num_labels: Number of output classes
        Dz_img: Dimension of image latent space

    Example:
        >>> # Load iMDBN model
        >>> model_data = iMDBN.load_model("path/to/imdbn.pkl")
        >>> adapter = iMDBNAdapter(model_data)
        >>>
        >>> # Extract joint embeddings
        >>> batch = (images, labels)  # images: [B, ...], labels: [B, K] one-hot
        >>> embeddings = adapter.encode(batch)  # Joint layer output
        >>>
        >>> # Or extract image-only embeddings
        >>> img_embeddings = adapter.encode_image_only(images)
    """

    def __init__(self, model):
        """
        Initialize iMDBN adapter.

        Args:
            model: iMDBN model object OR dict with keys:
                   - "image_idbn": Image DBN
                   - "joint_rbm": Joint RBM
                   - "num_labels": Number of classes
                   - "Dz_img": Image latent dimension
                   - Optional: "features", "params", "metadata"

        Raises:
            ValueError: If model doesn't have required components
        """
        super().__init__(model)

        # Extract components from model (support both object and dict format)
        if isinstance(model, dict):
            self.image_idbn = model.get("image_idbn")
            self.joint_rbm = model.get("joint_rbm")
            self.num_labels = model.get("num_labels")
            self.Dz_img = model.get("Dz_img")
            self.params = model.get("params", {})
            self.features = model.get("features")
            self.arch_str = model.get("arch_str", "unknown")
            self.metadata = model.get("metadata", {})

            # Optional statistics
            self.z_class_mean = model.get("z_class_mean")
            self.z_affine_scale = model.get("z_affine_scale")
            self.z_affine_bias = model.get("z_affine_bias")
            self.class_names = model.get("class_names")
        else:
            # Object-based model
            self.image_idbn = getattr(model, "image_idbn", None)
            self.joint_rbm = getattr(model, "joint_rbm", None)
            self.num_labels = getattr(model, "num_labels", None)
            self.Dz_img = getattr(model, "Dz_img", None)
            self.params = getattr(model, "params", {})
            self.features = getattr(model, "features", None)
            self.arch_str = getattr(model, "arch_str", "unknown")
            self.metadata = {}

            self.z_class_mean = getattr(model, "z_class_mean", None)
            self.z_affine_scale = getattr(model, "z_affine_scale", None)
            self.z_affine_bias = getattr(model, "z_affine_bias", None)
            self.class_names = getattr(model, "class_names", None)

        # Validate required components
        if self.image_idbn is None:
            raise ValueError("iMDBNAdapter requires 'image_idbn' component")
        if self.joint_rbm is None:
            raise ValueError("iMDBNAdapter requires 'joint_rbm' component")
        if self.num_labels is None:
            raise ValueError("iMDBNAdapter requires 'num_labels'")

        # Auto-infer Dz_img if not provided
        if self.Dz_img is None:
            if hasattr(self.image_idbn, "layers") and len(self.image_idbn.layers) > 0:
                last_layer = self.image_idbn.layers[-1]
                self.Dz_img = getattr(last_layer, "num_hidden", None)
            if self.Dz_img is None:
                raise ValueError("Could not infer Dz_img from image_idbn")

    def encode(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Extract joint multimodal embeddings.

        This is the main encoding method that extracts embeddings from the
        joint layer (combining image + label information).

        Args:
            batch: Tuple of (images, labels)
                   - images: [batch, ...] image tensors
                   - labels: [batch, num_labels] one-hot encoded labels

        Returns:
            Joint embeddings [batch, joint_hidden_dim]

        Example:
            >>> images = torch.randn(32, 1, 100, 100)
            >>> labels = F.one_hot(torch.randint(0, 32, (32,)), 32).float()
            >>> embeddings = adapter.encode((images, labels))
            >>> embeddings.shape  # [32, joint_hidden_dim]
        """
        images, labels = batch

        # Prepare inputs
        images = self.prepare_input(images)
        labels = self.prepare_input(labels)

        device = self.get_device()
        images = images.to(device).view(images.size(0), -1).float()
        labels = labels.to(device).float()

        # Encode through image_idbn
        with torch.no_grad():
            z_img = self.image_idbn.represent(images)

            # Concatenate image latents + labels
            v_joint = torch.cat([z_img, labels], dim=1)

            # Forward through joint RBM
            h_joint = self.joint_rbm.forward(v_joint)

        return h_joint

    def encode_image_only(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image-only embeddings (before joint layer).

        This extracts embeddings from the top layer of image_idbn,
        without combining with labels.

        Args:
            images: [batch, ...] image tensors

        Returns:
            Image embeddings [batch, Dz_img]

        Example:
            >>> images = torch.randn(32, 1, 100, 100)
            >>> z_img = adapter.encode_image_only(images)
            >>> z_img.shape  # [32, Dz_img]
        """
        images = self.prepare_input(images)
        device = self.get_device()
        images = images.to(device).view(images.size(0), -1).float()

        with torch.no_grad():
            z_img = self.image_idbn.represent(images)

        return z_img

    def encode_layerwise(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        layers: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract embeddings at each layer (image layers + joint layer).

        Args:
            batch: Tuple of (images, labels)
            layers: Layer indices to extract (1-indexed). None = all layers.
                   Example: [1, 2, 3] for image layers 1-2 + joint layer

        Returns:
            List of embeddings, one per requested layer

        Note:
            Layers 1 to N are image_idbn layers, layer N+1 is joint layer.

        Example:
            >>> batch = (images, labels)
            >>> embeddings = adapter.encode_layerwise(batch)
            >>> # embeddings[0] = first image layer
            >>> # embeddings[-1] = joint layer
        """
        images, labels = batch

        # Prepare inputs
        images = self.prepare_input(images)
        labels = self.prepare_input(labels)

        device = self.get_device()
        images = images.to(device).view(images.size(0), -1).float()
        labels = labels.to(device).float()

        n_image_layers = len(self.image_idbn.layers)

        if layers is None:
            layers = list(range(1, n_image_layers + 2))  # All layers + joint

        embeddings = []

        with torch.no_grad():
            # Extract image layers
            cur = images
            for li, rbm in enumerate(self.image_idbn.layers, start=1):
                cur = rbm.forward(cur)
                if li in layers:
                    embeddings.append(cur.detach())

            # Extract joint layer if requested
            joint_layer_idx = n_image_layers + 1
            if joint_layer_idx in layers:
                z_img = cur  # Last image layer output
                v_joint = torch.cat([z_img, labels], dim=1)
                h_joint = self.joint_rbm.forward(v_joint)
                embeddings.append(h_joint.detach())

        return embeddings

    def decode(
        self,
        z: torch.Tensor,
        from_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reconstruct input from embeddings.

        Note: This reconstructs images from image_idbn embeddings.
        For joint layer embeddings, use decode_from_joint().

        Args:
            z: Embedding tensor [batch, embedding_dim]
            from_layer: Image layer to decode from (1-indexed). None = from top.

        Returns:
            Reconstructed images [batch, input_dim]

        Example:
            >>> z_img = adapter.encode_image_only(images)
            >>> reconstructed = adapter.decode(z_img)
        """
        if from_layer is None:
            from_layer = len(self.image_idbn.layers)

        z = z.to(self.get_device())

        with torch.no_grad():
            # Backward through image RBM stack
            cur = z
            for rbm in reversed(self.image_idbn.layers[:from_layer]):
                if not hasattr(rbm, 'backward'):
                    raise NotImplementedError(
                        f"RBM layer does not have .backward() method. "
                        "Reconstruction not supported."
                    )
                cur = rbm.backward(cur)

        return cur

    def decode_from_joint(self, h_joint: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct visible units from joint layer embeddings.

        This returns concatenated [z_img, y] from joint hidden state.

        Args:
            h_joint: Joint embeddings [batch, joint_hidden_dim]

        Returns:
            Visible units [batch, Dz_img + num_labels]

        Example:
            >>> h_joint = adapter.encode(batch)
            >>> v_recon = adapter.decode_from_joint(h_joint)
            >>> z_img_recon = v_recon[:, :adapter.Dz_img]
            >>> y_recon = v_recon[:, adapter.Dz_img:]
        """
        h_joint = h_joint.to(self.get_device())

        with torch.no_grad():
            v_recon = self.joint_rbm.backward(h_joint)

        return v_recon

    def supports_reconstruction(self) -> bool:
        """Check if reconstruction is supported."""
        # Check if image_idbn layers have backward method
        if not hasattr(self.image_idbn, "layers"):
            return False

        for rbm in self.image_idbn.layers:
            if not hasattr(rbm, 'backward'):
                return False

        return True

    def supports_layerwise_extraction(self) -> bool:
        """Check if layerwise extraction is supported."""
        return True

    def get_device(self) -> torch.device:
        """
        Infer device from model parameters.

        Returns:
            Device where model parameters are stored
        """
        if self._device is not None:
            return self._device

        # Try to get device from image_idbn first layer
        if hasattr(self.image_idbn, "layers") and len(self.image_idbn.layers) > 0:
            first_rbm = self.image_idbn.layers[0]
            for attr_name in ("W", "hid_bias", "vis_bias", "weight", "bias"):
                attr_val = getattr(first_rbm, attr_name, None)
                if isinstance(attr_val, torch.Tensor):
                    self._device = attr_val.device
                    return self._device

        # Fallback to joint_rbm
        if self.joint_rbm is not None:
            for attr_name in ("W", "hid_bias", "vis_bias", "weight", "bias"):
                attr_val = getattr(self.joint_rbm, attr_name, None)
                if isinstance(attr_val, torch.Tensor):
                    self._device = attr_val.device
                    return self._device

        # Final fallback
        self._device = torch.device('cpu')
        return self._device

    def get_num_layers(self) -> int:
        """
        Return number of layers (image layers + joint layer).

        Returns:
            Total number of layers
        """
        n_image = len(self.image_idbn.layers) if hasattr(self.image_idbn, "layers") else 0
        return n_image + 1  # +1 for joint layer

    def get_layer_sizes(self) -> List[tuple]:
        """
        Get (input_size, output_size) for each layer.

        Returns:
            List of (num_visible, num_hidden) tuples for all layers

        Example:
            >>> adapter.get_layer_sizes()
            [(10000, 1500), (1500, 500), (532, 256)]  # Last is joint: (500+32, 256)
        """
        sizes = []

        # Image layers
        if hasattr(self.image_idbn, "layers"):
            for rbm in self.image_idbn.layers:
                if hasattr(rbm, 'num_visible') and hasattr(rbm, 'num_hidden'):
                    sizes.append((rbm.num_visible, rbm.num_hidden))
                else:
                    W = getattr(rbm, 'W', None)
                    if W is not None:
                        sizes.append(tuple(W.shape))
                    else:
                        sizes.append((None, None))

        # Joint layer
        if self.joint_rbm is not None:
            if hasattr(self.joint_rbm, 'num_visible') and hasattr(self.joint_rbm, 'num_hidden'):
                sizes.append((self.joint_rbm.num_visible, self.joint_rbm.num_hidden))
            else:
                W = getattr(self.joint_rbm, 'W', None)
                if W is not None:
                    sizes.append(tuple(W.shape))
                else:
                    sizes.append((None, None))

        return sizes

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extended metadata with iMDBN-specific information.

        Returns:
            Dictionary with adapter and model metadata
        """
        meta = super().get_metadata()

        # Add iMDBN-specific info
        layer_sizes = self.get_layer_sizes()
        meta.update({
            "model_type": "iMDBN",
            "architecture": self.arch_str,
            "layer_sizes": layer_sizes,
            "num_image_layers": len(self.image_idbn.layers) if hasattr(self.image_idbn, "layers") else 0,
            "num_labels": self.num_labels,
            "Dz_img": self.Dz_img,
            "has_features": self.features is not None,
            "has_class_mean": self.z_class_mean is not None,
        })

        # Add features info if available
        if self.features is not None:
            meta["features"] = list(self.features.keys())

        # Add class names if available
        if self.class_names is not None:
            meta["class_names"] = self.class_names

        # Add training params if available
        if self.params:
            meta["training_params"] = {
                k: v for k, v in self.params.items()
                if k in ("LEARNING_RATE", "JOINT_LEARNING_RATE", "CD", "JOINT_CD", "EPOCHS")
            }

        # Add saved metadata
        if self.metadata:
            meta["saved_metadata"] = self.metadata

        return meta

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"iMDBNAdapter("
            f"image_layers={len(self.image_idbn.layers) if hasattr(self.image_idbn, 'layers') else 0}, "
            f"joint_dim={self.joint_rbm.num_hidden if hasattr(self.joint_rbm, 'num_hidden') else '?'}, "
            f"arch={self.arch_str}, "
            f"device={self.get_device()})"
        )
