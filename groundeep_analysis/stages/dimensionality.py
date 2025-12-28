"""Dimensionality reduction stage - PCA, TSNE, UMAP."""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt

from groundeep_analysis.stages.pca.geometry import run_pca_geometry
from groundeep_analysis.stages.pca.report import generate_pca_decomposition_report, generate_pca_feature_colored_plots


class DimensionalityStage:
    """Stage for dimensionality reduction: PCA geometry, TSNE, UMAP.

    This stage performs dimensionality reduction analysis on embeddings.
    """

    name = "dimensionality"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        # Enabled if any sub-analysis is enabled
        return (settings.get('pca_geometry', {}).get('enabled', False) or
                settings.get('pca_report', {}).get('enabled', False) or
                settings.get('tsne', {}).get('enabled', False) or
                settings.get('umap', {}).get('enabled', False))

    def _get_model_layers(self, model_obj):
        """Get layers from model, handling both DBN and iMDBN."""
        # Handle iMDBN (dict format)
        if isinstance(model_obj, dict):
            if 'image_idbn' in model_obj:
                return getattr(model_obj['image_idbn'], 'layers', [])
        # Handle iMDBN (object format)
        elif hasattr(model_obj, 'image_idbn'):
            return getattr(model_obj.image_idbn, 'layers', [])
        # Handle standard DBN
        return getattr(model_obj, 'layers', [])

    def _get_model_device(self, model_obj) -> torch.device:
        """Infer device from model."""
        try:
            import torch.nn as nn
            if isinstance(model_obj, nn.Module):
                try:
                    return next(model_obj.parameters()).device
                except StopIteration:
                    pass
        except Exception:
            pass

        try:
            layers = self._get_model_layers(model_obj)
            if layers:
                first_rbm = layers[0]
                for attr_name in ("W", "c", "b", "weights"):
                    attr_val = getattr(first_rbm, attr_name, None)
                    if isinstance(attr_val, torch.Tensor):
                        return attr_val.device
        except Exception:
            pass
        return torch.device("cpu")

    def _extract_features(self, ctx: Any) -> Dict[str, np.ndarray]:
        """Extract visual features from context bundle."""
        features = {}

        # Extract labels (numerosity)
        if hasattr(ctx.bundle, 'labels'):
            features['labels'] = np.asarray(ctx.bundle.labels)

        # Extract cumulative area
        if hasattr(ctx.bundle, 'cum_area'):
            cum_area = ctx.bundle.cum_area
            if cum_area is not None:
                features['cum_area'] = np.asarray(cum_area).flatten()

        # Extract convex hull
        if hasattr(ctx.bundle, 'convex_hull'):
            ch = ctx.bundle.convex_hull
            if ch is not None:
                features['convex_hull'] = np.asarray(ch).flatten()

        # Extract density
        if hasattr(ctx.bundle, 'density'):
            density = ctx.bundle.density
            if density is not None:
                features['density'] = np.asarray(density).flatten()

        # Extract mean_item_size if available
        if hasattr(ctx.bundle, 'mean_item_size'):
            mean_item_size = ctx.bundle.mean_item_size
            if mean_item_size is not None:
                features['mean_item_size'] = np.asarray(mean_item_size).flatten()

        return features

    def _extract_layer_embeddings(self, ctx: Any, layers: List[int]) -> Dict[int, np.ndarray]:
        """Extract embeddings for specified layers."""
        dist_name = ctx.spec.distribution
        model_sel = (
            ctx.get_model("uniform")
            if dist_name == "uniform"
            else ctx.get_model("zipfian")
        )

        model_layers = self._get_model_layers(model_sel)
        if not model_layers:
            return {}

        device = self._get_model_device(model_sel)
        inputs_cpu = ctx.base_batch
        layer_embeddings = {}

        # Check if this is an iMDBN model
        is_imdbn = isinstance(model_sel, dict) and 'image_idbn' in model_sel and 'joint_rbm' in model_sel
        if not is_imdbn and hasattr(model_sel, 'image_idbn') and hasattr(model_sel, 'joint_rbm'):
            is_imdbn = True

        with torch.no_grad():
            inputs_device = inputs_cpu.to(device).view(inputs_cpu.shape[0], -1)
            cur = inputs_device

            # Extract image layers
            for li, rbm in enumerate(model_layers, start=1):
                cur = rbm.forward(cur)
                if li in layers:
                    layer_embeddings[li] = cur.detach().cpu().numpy()

            # For iMDBN, extract joint layer if requested
            if is_imdbn:
                joint_layer_idx = len(model_layers) + 1
                if joint_layer_idx in layers:
                    # Get labels and convert to one-hot
                    labels_np = ctx.bundle.labels
                    import torch.nn.functional as F
                    labels_tensor = torch.tensor(labels_np, dtype=torch.long, device=device)
                    # Shift labels if they're 1-indexed (1-32 -> 0-31)
                    if labels_np.min() >= 1:
                        labels_tensor = labels_tensor - 1
                    num_classes = int(labels_np.max()) if labels_np.min() >= 1 else int(labels_np.max()) + 1
                    labels_onehot = F.one_hot(labels_tensor, num_classes=num_classes).float()

                    # Get joint_rbm
                    if isinstance(model_sel, dict):
                        joint_rbm = model_sel['joint_rbm']
                    else:
                        joint_rbm = model_sel.joint_rbm

                    # Concatenate image latents + labels
                    z_img = cur  # Last image layer output
                    v_joint = torch.cat([z_img, labels_onehot], dim=1)

                    # Forward through joint RBM
                    h_joint = joint_rbm.forward(v_joint)
                    layer_embeddings[joint_layer_idx] = h_joint.detach().cpu().numpy()

            del inputs_device

        return layer_embeddings

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Run dimensionality reduction analyses on specified layers."""
        # Get model info
        dist_name = ctx.spec.distribution
        model_sel = (
            ctx.get_model("uniform")
            if dist_name == "uniform"
            else ctx.get_model("zipfian")
        )
        model_layers = self._get_model_layers(model_sel)

        # Check if iMDBN
        is_imdbn = isinstance(model_sel, dict) and 'image_idbn' in model_sel and 'joint_rbm' in model_sel
        if not is_imdbn and hasattr(model_sel, 'image_idbn') and hasattr(model_sel, 'joint_rbm'):
            is_imdbn = True

        # Determine which layers to analyze
        layers_config = settings.get('layers', 'top')
        if layers_config == 'all':
            layers = list(range(1, len(model_layers) + 1))
            # For iMDBN, include joint layer
            if is_imdbn:
                layers.append(len(model_layers) + 1)
        elif layers_config == 'top':
            # For iMDBN, 'top' means joint layer (last layer of full model)
            if is_imdbn:
                layers = [len(model_layers) + 1]  # Joint layer
            else:
                layers = [len(model_layers)]  # Last image layer
        elif isinstance(layers_config, list):
            layers = [int(l) for l in layers_config]
        else:
            layers = [int(layers_config)]

        print(f"[Dimensionality] Analyzing layers: {layers}")

        # Extract embeddings for all layers
        layer_embeddings = self._extract_layer_embeddings(ctx, layers)

        if not layer_embeddings:
            print("[Dimensionality] No layers to analyze")
            return

        # Get labels
        labels = ctx.bundle.labels

        # Run analyses for each layer
        for li, Z in layer_embeddings.items():
            layer_dir = output_dir / "dimensionality" / f"layer{li}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # PCA geometry
            if settings.get('pca_geometry', {}).get('enabled', False):
                self._run_pca_geometry(Z, labels, layer_dir, li, ctx, settings)

            # PCA decomposition report
            if settings.get('pca_report', {}).get('enabled', False):
                self._run_pca_report(Z, labels, layer_dir, li, ctx, settings)

            # TSNE
            if settings.get('tsne', {}).get('enabled', False):
                self._run_tsne(Z, labels, layer_dir, li, ctx, settings)

            # UMAP
            if settings.get('umap', {}).get('enabled', False):
                self._run_umap(Z, labels, layer_dir, li, ctx, settings)

    def _run_pca_geometry(self, Z: np.ndarray, labels: np.ndarray,
                         layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run PCA geometry analysis for one layer."""
        pca_geo_dir = layer_dir / "pca_geometry"
        pca_geo_dir.mkdir(parents=True, exist_ok=True)

        pca_cfg = settings.get('pca_geometry', {})
        per_class = pca_cfg.get('per_class', 200)
        run_isomap = pca_cfg.get('isomap', False)

        tag = f"{ctx.spec.arch_name}_{ctx.spec.distribution}_layer{layer_idx}"

        try:
            report = run_pca_geometry(
                embeddings=Z,
                labels=labels,
                name=tag,
                outdir=pca_geo_dir,
                per_class=per_class,
                run_isomap=run_isomap,
            )

            print(f"[PCA Geometry] Layer {layer_idx}: "
                  f"anisotropy_within={report.variance.anisotropy_within:.3f}, "
                  f"anisotropy_between={report.variance.anisotropy_between:.3f}")

            # WandB logging
            if ctx.wandb_run:
                try:
                    import wandb
                    ctx.wandb_run.log({
                        f"dimensionality/layer{layer_idx}/pca_geo/anisotropy_within": report.variance.anisotropy_within,
                        f"dimensionality/layer{layer_idx}/pca_geo/anisotropy_between": report.variance.anisotropy_between,
                        f"dimensionality/layer{layer_idx}/pca_geo/angle_pc1_deg": report.angles.angle_pc1_deg,
                        f"dimensionality/layer{layer_idx}/pca_geo/rho_pc1": report.angles.rho_pc1,
                    })
                except Exception:
                    pass

        except Exception as exc:
            print(f"[PCA Geometry] Layer {layer_idx}: failed ({exc})")

    def _run_pca_report(self, Z: np.ndarray, labels: np.ndarray,
                       layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run PCA decomposition report for one layer."""
        pca_rep_dir = layer_dir / "pca_report"
        pca_rep_dir.mkdir(parents=True, exist_ok=True)

        pca_cfg = settings.get('pca_report', {})
        random_state = pca_cfg.get('random_state', 42)

        regime_name = ctx.spec.distribution
        layer_tag = f"layer{layer_idx}"

        try:
            report_dict = generate_pca_decomposition_report(
                embeddings=Z,
                labels=labels,
                out_dir=pca_rep_dir,
                random_state=random_state,
            )

            if report_dict:
                samples_info = report_dict.get('samples', {})
                rho_pc1 = samples_info.get('rho', [0])[0]
                evr_pc1 = samples_info.get('evr', [0])[0]
                print(f"[PCA Report] Layer {layer_idx}: rho_pc1={rho_pc1:.3f}, evr_pc1={evr_pc1:.3f}")

                # WandB logging
                if ctx.wandb_run:
                    try:
                        import wandb
                        ctx.wandb_run.log({
                            f"dimensionality/layer{layer_idx}/pca_report/rho_pc1": rho_pc1,
                            f"dimensionality/layer{layer_idx}/pca_report/evr_pc1": evr_pc1,
                            f"dimensionality/layer{layer_idx}/pca_report/angle_deg": report_dict.get('angle_deg', 0),
                        })
                    except Exception:
                        pass

            # Generate feature-colored PCA plots
            features_dict = self._extract_features(ctx)
            if features_dict:
                feature_plots_dir = pca_rep_dir / "feature_colored"
                try:
                    generate_pca_feature_colored_plots(
                        embeddings=Z,
                        features=features_dict,
                        out_dir=feature_plots_dir,
                        random_state=random_state,
                    )
                    print(f"[PCA Report] Layer {layer_idx}: generated {len(features_dict)} feature-colored plots")
                except Exception as exc_feat:
                    print(f"[PCA Report] Layer {layer_idx}: feature plots failed ({exc_feat})")

        except Exception as exc:
            print(f"[PCA Report] Layer {layer_idx}: failed ({exc})")

    def _run_tsne(self, Z: np.ndarray, labels: np.ndarray,
                 layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run TSNE dimensionality reduction for one layer."""
        tsne_dir = layer_dir / "tsne"
        tsne_dir.mkdir(parents=True, exist_ok=True)

        tsne_cfg = settings.get('tsne', {})
        perplexity = tsne_cfg.get('perplexity', 30)
        n_iter = tsne_cfg.get('n_iter', 1000)
        random_state = tsne_cfg.get('random_state', 42)

        try:
            from sklearn.manifold import TSNE

            # Subsample if too large
            max_samples = tsne_cfg.get('max_samples', 5000)
            if Z.shape[0] > max_samples:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
                Z_sub = Z[idx]
                labels_sub = labels[idx]
            else:
                Z_sub = Z
                labels_sub = labels

            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
            Z_tsne = tsne.fit_transform(Z_sub)

            # Save embeddings
            np.save(tsne_dir / "tsne_embeddings.npy", Z_tsne)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=labels_sub, cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Numerosity')
            ax.set_title(f"TSNE — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]")
            ax.set_xlabel("TSNE 1")
            ax.set_ylabel("TSNE 2")
            fig.tight_layout()
            fig.savefig(tsne_dir / "tsne_plot.png", dpi=300)
            plt.close(fig)

            print(f"[TSNE] Layer {layer_idx}: completed")

            # WandB logging
            if ctx.wandb_run:
                try:
                    import wandb
                    ctx.wandb_run.log({
                        f"dimensionality/layer{layer_idx}/tsne": wandb.Image(str(tsne_dir / "tsne_plot.png")),
                    })
                except Exception:
                    pass

        except ImportError:
            print(f"[TSNE] Layer {layer_idx}: skipped (sklearn not available)")
        except Exception as exc:
            print(f"[TSNE] Layer {layer_idx}: failed ({exc})")

    def _run_umap(self, Z: np.ndarray, labels: np.ndarray,
                 layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run UMAP dimensionality reduction for one layer."""
        umap_dir = layer_dir / "umap"
        umap_dir.mkdir(parents=True, exist_ok=True)

        umap_cfg = settings.get('umap', {})
        n_neighbors = umap_cfg.get('n_neighbors', 15)
        min_dist = umap_cfg.get('min_dist', 0.1)
        random_state = umap_cfg.get('random_state', 42)

        try:
            import umap as umap_pkg

            # Subsample if too large
            max_samples = umap_cfg.get('max_samples', 5000)
            if Z.shape[0] > max_samples:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
                Z_sub = Z[idx]
                labels_sub = labels[idx]
            else:
                Z_sub = Z
                labels_sub = labels

            reducer = umap_pkg.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state
            )
            Z_umap = reducer.fit_transform(Z_sub)

            # Save embeddings
            np.save(umap_dir / "umap_embeddings.npy", Z_umap)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(Z_umap[:, 0], Z_umap[:, 1], c=labels_sub, cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Numerosity')
            ax.set_title(f"UMAP — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            fig.tight_layout()
            fig.savefig(umap_dir / "umap_plot.png", dpi=300)
            plt.close(fig)

            print(f"[UMAP] Layer {layer_idx}: completed")

            # WandB logging
            if ctx.wandb_run:
                try:
                    import wandb
                    ctx.wandb_run.log({
                        f"dimensionality/layer{layer_idx}/umap": wandb.Image(str(umap_dir / "umap_plot.png")),
                    })
                except Exception:
                    pass

        except ImportError:
            print(f"[UMAP] Layer {layer_idx}: skipped (umap-learn not installed)")
        except Exception as exc:
            print(f"[UMAP] Layer {layer_idx}: failed ({exc})")
