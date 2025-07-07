"""
Experimental adapter for eigenpatches model with enhanced configuration and logging.

This module provides an experimental wrapper around the original eigenpatches
implementation, adding YAML configuration support, detailed logging, and
experimental features while maintaining full API compatibility.
"""

import numpy as np
import cv2
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys
import os

# Add template_matching to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_MATCHING_PATH = PROJECT_ROOT / "template_matching" / "src"
sys.path.insert(0, str(TEMPLATE_MATCHING_PATH))

try:
    from core.eigenpatches import EigenpatchesModel, MultiScaleEigenpatches
except ImportError as e:
    raise ImportError(f"Could not import original eigenpatches: {e}")


class ExperimentalEigenpatches:
    """
    Experimental adapter for eigenpatches model with enhanced features.
    
    This class wraps the original EigenpatchesModel and MultiScaleEigenpatches
    classes, adding configuration management, detailed logging, and experimental
    features while maintaining full compatibility with the original API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 config_file: Optional[str] = None):
        """
        Initialize experimental eigenpatches model.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config, config_file)
        self._setup_logging()
        
        # Extract parameters from config
        eigenpatches_config = self.config.get('eigenpatches', {})
        self.patch_size = eigenpatches_config.get('patch_size', 21)
        self.n_components = eigenpatches_config.get('n_components', 20)
        self.pyramid_levels = eigenpatches_config.get('pyramid_levels', 3)
        self.scale_factor = eigenpatches_config.get('scale_factor', 0.5)
        
        # Initialize original model
        self.use_multiscale = self.pyramid_levels > 1
        if self.use_multiscale:
            self.model = MultiScaleEigenpatches(
                patch_size=self.patch_size,
                n_components=self.n_components,
                pyramid_levels=self.pyramid_levels,
                scale_factor=self.scale_factor
            )
        else:
            self.model = EigenpatchesModel(
                patch_size=self.patch_size,
                n_components=self.n_components
            )
        
        # Experimental features
        self.enable_detailed_logging = self.config.get('logging', {}).get('level', 'INFO') == 'DEBUG'
        self.experimental_features = self.config.get('experimental', {})
        
        self.logger.info(f"Initialized ExperimentalEigenpatches with patch_size={self.patch_size}, "
                        f"n_components={self.n_components}, pyramid_levels={self.pyramid_levels}")
    
    def _load_config(self, config: Optional[Dict[str, Any]], 
                    config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from dict or file."""
        if config is not None:
            return config
        
        if config_file is not None:
            config_path = Path(config_file)
            if not config_path.exists():
                # Try relative to project root
                config_path = PROJECT_ROOT / config_file
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Load default configuration
        default_config_path = PROJECT_ROOT / "matching_experimental" / "configs" / "default_config.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Fallback to hardcoded defaults
        return {
            'eigenpatches': {
                'patch_size': 21,
                'n_components': 20,
                'pyramid_levels': 3,
                'scale_factor': 0.5
            },
            'logging': {'level': 'INFO'},
            'experimental': {}
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Configure logger
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.get('format', 
                         '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        if log_config.get('console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file_logging', True):
            log_dir = Path(self.config.get('paths', {}).get('logs_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / 'eigenpatches.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def train(self, images: List[np.ndarray], landmarks: List[np.ndarray]) -> None:
        """
        Train the eigenpatches model.
        
        Args:
            images: List of training images
            landmarks: List of corresponding landmark coordinates
        """
        self.logger.info(f"Starting training with {len(images)} images")
        
        # Log training statistics
        if self.enable_detailed_logging:
            self._log_training_statistics(images, landmarks)
        
        # Train original model
        self.model.train(images, landmarks)
        
        # Log training completion
        self.logger.info("Training completed successfully")
        
        # Experimental: Analyze PCA components
        if self.experimental_features.get('analyze_pca_components', False):
            self._analyze_pca_components()
    
    def _log_training_statistics(self, images: List[np.ndarray], 
                               landmarks: List[np.ndarray]) -> None:
        """Log detailed training statistics."""
        self.logger.debug(f"Training data statistics:")
        self.logger.debug(f"  Number of images: {len(images)}")
        self.logger.debug(f"  Number of landmarks per image: {landmarks[0].shape[0] if landmarks else 0}")
        
        # Image statistics
        if images:
            img_shapes = [img.shape for img in images]
            self.logger.debug(f"  Image shapes: {set(img_shapes)}")
            
            # Intensity statistics
            intensities = [img.mean() for img in images]
            self.logger.debug(f"  Mean intensity: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
    
    def _analyze_pca_components(self) -> None:
        """Analyze PCA components for experimental insights."""
        self.logger.info("Analyzing PCA components")
        
        if hasattr(self.model, 'pca_models'):
            for landmark_idx, pca_model in self.model.pca_models.items():
                explained_variance = pca_model.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                self.logger.debug(f"Landmark {landmark_idx}:")
                self.logger.debug(f"  Explained variance ratio: {explained_variance[:5]}")
                self.logger.debug(f"  Cumulative variance (top 5): {cumulative_variance[:5]}")
                self.logger.debug(f"  Total variance explained: {cumulative_variance[-1]:.3f}")
    
    def predict(self, image: np.ndarray, 
                initial_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict landmarks for a single image.
        
        Args:
            image: Input image
            initial_landmarks: Optional initial landmark positions
            
        Returns:
            Predicted landmark coordinates
        """
        if self.enable_detailed_logging:
            self.logger.debug(f"Predicting landmarks for image shape: {image.shape}")
        
        # Use original model prediction
        if hasattr(self.model, 'predict'):
            return self.model.predict(image, initial_landmarks)
        else:
            # Fallback for EigenpatchesModel
            raise NotImplementedError("Direct prediction not available for base EigenpatchesModel")
    
    def score_patches(self, image: np.ndarray, positions: List[Tuple[float, float]], 
                     landmark_idx: int) -> List[float]:
        """
        Score patches at given positions for a specific landmark.
        
        Args:
            image: Input image
            positions: List of (x, y) positions to score
            landmark_idx: Index of landmark to score for
            
        Returns:
            List of scores for each position
        """
        if self.enable_detailed_logging:
            self.logger.debug(f"Scoring {len(positions)} patches for landmark {landmark_idx}")
        
        return self.model.score_patches(image, positions, landmark_idx)
    
    def visualize_components(self, landmark_idx: int, 
                           n_components: int = 5) -> Optional[np.ndarray]:
        """
        Visualize PCA components for a specific landmark.
        
        Args:
            landmark_idx: Index of landmark
            n_components: Number of components to visualize
            
        Returns:
            Visualization array if available
        """
        if not hasattr(self.model, 'pca_models'):
            self.logger.warning("PCA models not available for visualization")
            return None
        
        if landmark_idx not in self.model.pca_models:
            self.logger.warning(f"Landmark {landmark_idx} not found in PCA models")
            return None
        
        pca_model = self.model.pca_models[landmark_idx]
        components = pca_model.components_[:n_components]
        
        # Reshape components to patch size
        patch_size = self.patch_size
        reshaped_components = components.reshape(n_components, patch_size, patch_size)
        
        self.logger.info(f"Generated visualization for {n_components} components of landmark {landmark_idx}")
        return reshaped_components
    
    def save(self, filepath: str) -> None:
        """
        Save the model to file.
        
        Args:
            filepath: Path to save the model
        """
        self.logger.info(f"Saving model to: {filepath}")
        
        # Save original model
        self.model.save(filepath)
        
        # Save experimental metadata
        metadata_path = Path(filepath).with_suffix('.meta.yaml')
        metadata = {
            'config': self.config,
            'model_type': 'ExperimentalEigenpatches',
            'version': '1.0.0',
            'parameters': {
                'patch_size': self.patch_size,
                'n_components': self.n_components,
                'pyramid_levels': self.pyramid_levels,
                'scale_factor': self.scale_factor
            }
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        self.logger.info(f"Model and metadata saved successfully")
    
    def load(self, filepath: str) -> None:
        """
        Load the model from file.
        
        Args:
            filepath: Path to load the model from
        """
        self.logger.info(f"Loading model from: {filepath}")
        
        # Load original model
        self.model.load(filepath)
        
        # Load experimental metadata if available
        metadata_path = Path(filepath).with_suffix('.meta.yaml')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            self.logger.info(f"Loaded metadata for model version: {metadata.get('version', 'unknown')}")
        
        self.logger.info("Model loaded successfully")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics and model information.
        
        Returns:
            Dictionary containing training statistics
        """
        stats = {
            'patch_size': self.patch_size,
            'n_components': self.n_components,
            'pyramid_levels': self.pyramid_levels,
            'scale_factor': self.scale_factor,
            'use_multiscale': self.use_multiscale,
            'is_trained': getattr(self.model, 'is_trained', False)
        }
        
        # Add PCA statistics if available
        if hasattr(self.model, 'pca_models'):
            stats['n_landmarks'] = len(self.model.pca_models)
            
            # Compute explained variance statistics
            explained_variances = []
            for pca_model in self.model.pca_models.values():
                explained_variances.append(pca_model.explained_variance_ratio_.sum())
            
            if explained_variances:
                stats['mean_explained_variance'] = np.mean(explained_variances)
                stats['std_explained_variance'] = np.std(explained_variances)
        
        return stats