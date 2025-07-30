"""
Pep2Prob: Benchmark package for peptide probability prediction

This package provides tools and benchmarks for predicting peptide probabilities
from mass spectrometry data using various machine learning models.
"""

__version__ = "0.1.0"
__author__ = "Bandeira Lab"
__email__ = ""

# Import main classes and functions for easy access
from .dataset import Pep2ProbDataset, Pep2ProbDataSet
from .utils import *

# Define what gets imported with "from pep2prob import *"
__all__ = [
    "Pep2ProbDataset",
    "Pep2ProbDataSet",
    # Add other public functions/classes from utils as needed
]
