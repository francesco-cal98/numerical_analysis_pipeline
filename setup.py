"""
GROUNDEEP Analysis - Model-Agnostic Deep Learning Analysis Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="groundeep-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Model-agnostic deep learning analysis framework with adapter system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/groundeep-analysis",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "umap-learn>=0.5.0",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "sphinx>=4.5.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "groundeep-analyze=groundeep_analysis.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "groundeep_analysis": [
            "configs/*.yaml",
            "configs/templates/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "deep-learning",
        "neural-networks",
        "analysis",
        "interpretability",
        "representational-similarity",
        "dimensionality-reduction",
        "pytorch",
        "vae",
        "dbn",
        "adapter-pattern",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/groundeep-analysis/issues",
        "Documentation": "https://groundeep-analysis.readthedocs.io/",
        "Source": "https://github.com/yourusername/groundeep-analysis",
    },
)
