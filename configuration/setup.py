"""
Setup script for Atmospheric Radiation Surrogate Model.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="atmos-surrogate",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning surrogate model for atmospheric radiative transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/surrogate",
    py_modules=[
        "models",
        "utils",
        "data",
        "config",
        "run_finetune",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "mypy>=0.990",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
        "logging": [
            "tensorboard>=2.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="atmospheric-science radiative-transfer deep-learning surrogate-model",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/surrogate/issues",
        "Source": "https://github.com/yourusername/surrogate",
    },
)
