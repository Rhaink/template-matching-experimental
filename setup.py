#!/usr/bin/env python
"""
Setup script for the Matching Experimental platform.

This script enables installation of the experimental template matching platform
with all dependencies and development tools.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
    "pre-commit>=2.20.0",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "nbsphinx>=0.8.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "memory-profiler>=0.60.0",
    "line-profiler>=4.0.0",
]

setup(
    name="matching-experimental",
    version="1.0.0",
    author="Template Matching Research Team",
    author_email="research@example.com",
    description="Advanced Template Matching Research Platform with Eigenpatches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/donrobot/template-matching-experimental",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "full": requirements + dev_requirements,
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "matching-experimental-train=scripts.train_experimental:main",
            "matching-experimental-process=scripts.process_experimental:main",
            "matching-experimental-evaluate=scripts.evaluate_experimental:main",
            "matching-experimental-pipeline=scripts.run_full_pipeline:main",
        ],
    },
    package_data={
        "matching_experimental": [
            "configs/*.yaml",
            "tests/fixtures/*.py",
            "docs/*.md",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "template matching",
        "eigenpatches",
        "landmark detection",
        "medical image analysis",
        "computer vision",
        "pca",
        "shape models",
        "research platform",
    ],
    project_urls={
        "Bug Reports": "https://github.com/donrobot/template-matching-experimental/issues",
        "Source": "https://github.com/donrobot/template-matching-experimental",
        "Documentation": "https://github.com/donrobot/template-matching-experimental/wiki",
        "Research": "https://github.com/donrobot/template-matching-experimental/wiki",
    },
)
