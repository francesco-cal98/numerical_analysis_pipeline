"""
Analysis stages for GROUNDEEP pipeline.

Available stages:
- LinearProbesStage: Train linear probes for feature prediction
- GeometryStage: RSA, RDM, monotonicity analysis
- DimensionalityStage: PCA, UMAP, t-SNE
- ReconstructionStage: MSE, AFP, SSIM metrics
- CKAStage: Centered Kernel Alignment
- BehavioralStage: Numerosity comparison tasks
- PowerLawStage: Power-law fitting analysis
- PCADiagnosticsStage: PCA diagnostics and visualization
"""

from groundeep_analysis.stages.base import BaseStage, StageRegistry
from groundeep_analysis.stages.probes import LinearProbesStage
from groundeep_analysis.stages.geometry import GeometryStage
from groundeep_analysis.stages.dimensionality import DimensionalityStage
from groundeep_analysis.stages.reconstruction import ReconstructionStage
from groundeep_analysis.stages.cka import CKAStage
from groundeep_analysis.stages.behavioral import BehavioralStage
from groundeep_analysis.stages.powerlaw import PowerLawStage
from groundeep_analysis.stages.pca_diagnostics import PCADiagnosticsStage

__all__ = [
    'BaseStage',
    'StageRegistry',
    'LinearProbesStage',
    'GeometryStage',
    'DimensionalityStage',
    'ReconstructionStage',
    'CKAStage',
    'BehavioralStage',
    'PowerLawStage',
    'PCADiagnosticsStage',
]
