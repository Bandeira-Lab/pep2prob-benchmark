#!/usr/bin/env python3
"""
Setup script for the pep2prob package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Pep2Prob benchmark package for peptide probability prediction"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="pep2prob",
    version="0.1.0",
    author="Hao Xu, Zhichao Wang, and Shengqi Sang",
    author_email="nyx0flower@gmail.com",
    description="Benchmark package for peptide probability prediction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/bandeiralab/pep2prob-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "torch": [
            "torch>=1.9.0",
            "torchvision",
            "torchaudio",
        ],
        "tensorflow": [
            "tensorflow>=2.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pep2prob-train=pep2prob.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pep2prob": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    zip_safe=False,
)
