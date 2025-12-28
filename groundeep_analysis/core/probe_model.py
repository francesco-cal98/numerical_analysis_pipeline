"""Probe-ready model wrapper.

This module contains ProbeReadyModel, previously in analyze.py.
Now it's clean with zero dependencies on analyze.py.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np


class ProbeReadyModel:
    """Adapter for offline probe training.

    Wraps a raw model and provides a clean interface for extracting
    representations at different layers.

    Attributes:
        raw: The underlying model (iDBN, gDBN, etc.)
        val_loader: Validation dataloader
        features: Dictionary of features (labels, cumArea, CH, etc.)
        wandb_run: Optional WandB run for logging
        arch_dir: Output directory path
        text_flag: Whether model processes text (always False for iDBN)
        device: Device where model is located
    """

    def __init__(
        self,
        raw_model: Any,
        val_loader: Any,
        features_dict: Dict[str, np.ndarray],
        out_dir: Path,
        wandb_run: Optional[Any] = None,
    ):
        """Initialize ProbeReadyModel.

        Args:
            raw_model: The model to wrap
            val_loader: Validation dataloader
            features_dict: Dictionary of features for probing
            out_dir: Output directory
            wandb_run: Optional WandB run
        """
        self.raw = raw_model
        self.val_loader = val_loader
        self.features = features_dict
        self.wandb_run = wandb_run
        self.arch_dir = str(out_dir)
        self.text_flag = False
        self.device = self._get_model_device(raw_model)

        # Detect if this is an iMDBN model
        self.is_imdbn = self._is_imdbn_model(raw_model)

        # Get number of image layers and joint layer index if iMDBN
        if self.is_imdbn:
            self.num_image_layers = self._get_num_image_layers(raw_model)
            self.joint_layer_idx = self.num_image_layers + 1
            self.num_labels = self._get_num_labels(raw_model)
        else:
            self.num_image_layers = None
            self.joint_layer_idx = None
            self.num_labels = None

    def _is_imdbn_model(self, model: Any) -> bool:
        """Check if model is iMDBN (multimodal)."""
        if isinstance(model, dict):
            return 'image_idbn' in model and 'joint_rbm' in model
        return hasattr(model, 'image_idbn') and hasattr(model, 'joint_rbm')

    def _get_num_image_layers(self, model: Any) -> int:
        """Get number of image layers in iMDBN."""
        if isinstance(model, dict):
            image_idbn = model.get('image_idbn')
        else:
            image_idbn = getattr(model, 'image_idbn', None)

        if image_idbn is None:
            return 0

        layers = getattr(image_idbn, 'layers', [])
        return len(layers)

    def _get_num_labels(self, model: Any) -> int:
        """Get number of labels for iMDBN."""
        if isinstance(model, dict):
            return model.get('num_labels', 32)
        return getattr(model, 'num_labels', 32)

    def _get_model_device(self, model: Any) -> torch.device:
        """Infer device from model parameters.

        Args:
            model: The model

        Returns:
            Device (cuda or cpu)
        """
        try:
            # For iMDBN: check image_idbn layers
            if self._is_imdbn_model(model):
                if isinstance(model, dict):
                    image_idbn = model.get('image_idbn')
                else:
                    image_idbn = getattr(model, 'image_idbn', None)

                if image_idbn and hasattr(image_idbn, 'layers'):
                    layers = image_idbn.layers
                    if layers:
                        first_rbm = layers[0]
                        for attr_name in ("W", "c", "b", "weights", "hid_bias", "vis_bias"):
                            attr_val = getattr(first_rbm, attr_name, None)
                            if isinstance(attr_val, torch.Tensor):
                                return attr_val.device

            # Try to get device from model layers (for RBM stack)
            layers = getattr(model, "layers", [])
            if layers:
                first_rbm = layers[0]
                for attr_name in ("W", "c", "b", "weights"):
                    attr_val = getattr(first_rbm, attr_name, None)
                    if isinstance(attr_val, torch.Tensor):
                        return attr_val.device
        except Exception:
            pass

        # Fallback to CPU
        return torch.device("cpu")

    @torch.no_grad()
    def represent(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        upto_layer: Optional[int] = None
    ) -> torch.Tensor:
        """Extract representations from the model.

        Args:
            x: Input tensor (images)
            labels: Label tensor (required for iMDBN joint layer, optional otherwise)
            upto_layer: Layer index to stop at (None = top layer, which is joint for iMDBN)

        Returns:
            Representations at specified layer
        """
        # For iMDBN models
        if self.is_imdbn:
            # Extract model components
            if isinstance(self.raw, dict):
                image_idbn = self.raw['image_idbn']
                joint_rbm = self.raw['joint_rbm']
            else:
                image_idbn = self.raw.image_idbn
                joint_rbm = self.raw.joint_rbm

            # Determine target layer
            if upto_layer is None:
                # None means "top" which is joint layer for iMDBN
                target_layer = self.joint_layer_idx
            else:
                target_layer = upto_layer

            # Forward through image layers
            xt = x.to(self.device).view(x.size(0), -1).float()
            image_layers = image_idbn.layers

            if target_layer <= self.num_image_layers:
                # Extract from image layer only
                for i, rbm in enumerate(image_layers, start=1):
                    xt = rbm.forward(xt)
                    if i == target_layer:
                        return xt
                return xt
            else:
                # Extract joint layer (target_layer == self.joint_layer_idx)
                # Forward through all image layers
                for rbm in image_layers:
                    xt = rbm.forward(xt)

                # Now xt is z_img from top image layer
                # Need labels to compute joint layer
                if labels is None:
                    raise ValueError(
                        f"Labels required for joint layer extraction (layer {self.joint_layer_idx}). "
                        "Pass labels to represent() method."
                    )

                # Convert labels to one-hot if needed
                labels_t = labels.to(self.device).float()
                if labels_t.ndim == 1:
                    # Convert scalar labels to one-hot
                    labels_t = F.one_hot(labels_t.long(), num_classes=self.num_labels).float()

                # Concatenate image latents + labels
                v_joint = torch.cat([xt, labels_t], dim=1)

                # Forward through joint RBM
                h_joint = joint_rbm.forward(v_joint)
                return h_joint

        # For standard DBN models
        # Check if model has built-in represent method
        if hasattr(self.raw, "represent"):
            if upto_layer is not None:
                return self.raw.represent(x, upto_layer=upto_layer)
            else:
                return self.raw.represent(x)

        # Fallback: manually forward through RBM stack
        xt = x
        layers = getattr(self.raw, "layers", [])
        upto = len(layers) if upto_layer is None else min(upto_layer, len(layers))

        for rbm in layers[:upto]:
            xt = rbm.forward(xt)

        return xt
