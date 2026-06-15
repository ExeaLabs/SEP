"""
setup.py — Species Extinction Risk Predictor
"""

from pathlib import Path

from setuptools import find_packages, setup

_HERE = Path(__file__).resolve().parent

# Read the long description from README.md
_LONG_DESC = ""
readme = _HERE / "README.md"
if readme.exists():
    _LONG_DESC = readme.read_text(encoding="utf-8")

setup(
    name="species-extinction-predictor",
    version="0.1.0",
    description=(
        "Multi-modal deep learning model for predicting IUCN extinction "
        "risk categories from satellite imagery, climate data, and species "
        "occurrence records."
    ),
    long_description=_LONG_DESC,
    long_description_content_type="text/markdown",
    author="Team B - Science/Medical Applications",
    license="MIT",
    url="https://github.com/Arya-Addagarla/SEP",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "folium>=0.14",
        "branca>=0.6",
        "pillow>=10.0",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
        ],
        "wandb": [
            "wandb>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "sep-train=train:main",
            "sep-predict=predict:main",
            "sep-evaluate=evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
