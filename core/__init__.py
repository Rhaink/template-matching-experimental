"""
Core module for matching_experimental package.

This module provides experimental adaptations of the template matching system,
adding configuration management, enhanced logging, and experimental features
while maintaining compatibility with the original template_matching code.
"""

__version__ = "1.0.0"
__author__ = "Template Matching Research Team"
__email__ = "research@templatematching.org"

# Package metadata
PACKAGE_INFO = {
    'name': 'matching_experimental',
    'version': __version__,
    'description': 'Experimental platform for template matching landmark detection',
    'author': __author__,
    'email': __email__,
    'license': 'MIT',
    'baseline_accuracy': '5.63±0.17 pixels',
    'test_images': 159,
    'dataset': 'coordenadas_prueba_1.csv'
}

# Lazy imports to avoid circular dependency issues
def get_experimental_eigenpatches():
    """Get ExperimentalEigenpatches class."""
    from .experimental_eigenpatches import ExperimentalEigenpatches
    return ExperimentalEigenpatches

def get_experimental_predictor():
    """Get ExperimentalLandmarkPredictor class."""
    from .experimental_predictor import ExperimentalLandmarkPredictor
    return ExperimentalLandmarkPredictor

def get_experimental_evaluator():
    """Get ExperimentalEvaluator class."""
    from .experimental_evaluator import ExperimentalEvaluator
    return ExperimentalEvaluator