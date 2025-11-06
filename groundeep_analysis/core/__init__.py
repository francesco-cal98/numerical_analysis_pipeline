"""
Core components of the GROUNDEEP analysis pipeline.

This module contains the fundamental building blocks:
- adapters: Model adapter system for different architectures
- model_manager: Model loading and management
- dataset_manager: Dataset handling and dataloaders
- embedding_extractor: Feature extraction from models
- context: Shared context for analysis stages
- analysis_types: Type definitions and data structures
"""

from groundeep_analysis.core.dataset_manager import DatasetManager
from groundeep_analysis.core.model_manager import ModelManager
from groundeep_analysis.core.embedding_extractor import EmbeddingExtractor
from groundeep_analysis.core.context import AnalysisContext, StageResult

__all__ = [
    "DatasetManager",
    "ModelManager",
    "EmbeddingExtractor",
    "AnalysisContext",
    "StageResult",
]
