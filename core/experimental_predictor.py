"""
Experimental adapter for template landmark predictor with enhanced features.

This module provides an experimental wrapper around the original TemplateLandmarkPredictor,
adding YAML configuration support, detailed logging, convergence analysis, and experimental
features while maintaining full API compatibility.
"""

import numpy as np
import cv2
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import sys
import time
from dataclasses import dataclass

# Add template_matching to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_MATCHING_PATH = PROJECT_ROOT / "template_matching" / "src"
sys.path.insert(0, str(TEMPLATE_MATCHING_PATH))

try:
    from core.landmark_predictor import TemplateLandmarkPredictor
except ImportError as e:
    raise ImportError(f"Could not import original landmark predictor: {e}")


@dataclass
class PredictionResult:
    """
    Container for prediction results with additional metadata.
    """
    landmarks: np.ndarray
    confidence: Optional[np.ndarray] = None
    iterations: int = 0
    convergence_error: float = 0.0
    processing_time: float = 0.0
    geometric_constraints_applied: bool = False
    initial_landmarks: Optional[np.ndarray] = None
    refinement_history: Optional[List[np.ndarray]] = None


class ExperimentalLandmarkPredictor:
    """
    Experimental adapter for template landmark predictor with enhanced features.
    
    This class wraps the original TemplateLandmarkPredictor, adding configuration
    management, detailed logging, convergence analysis, and experimental features
    while maintaining full compatibility with the original API.
    
    Mathematical Foundation:
    -----------------------
    The template matching uses eigenpatches with geometric constraints:
    
    1. Eigenpatches Scoring:
       S(x,y) = ||P(x,y) - μ||² - ||P_proj(x,y) - μ||²
       where P(x,y) is the patch at position (x,y), μ is the mean patch,
       and P_proj is the PCA reconstruction.
    
    2. Geometric Constraints:
       Shape parameters b must satisfy |b_i| ≤ 3√λ_i
       where λ_i are the eigenvalues of the shape covariance matrix.
    
    3. Procrustes Alignment:
       Minimize ||X - sRY - t1^T||² over scale s, rotation R, translation t.
    
    4. Iterative Refinement:
       x_{k+1} = x_k + α∇S(x_k) subject to geometric constraints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 config_file: Optional[str] = None):
        """
        Initialize experimental landmark predictor.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config, config_file)
        self._setup_logging()
        
        # Extract parameters from config
        eigenpatches_config = self.config.get('eigenpatches', {})
        predictor_config = self.config.get('landmark_predictor', {})
        
        self.patch_size = eigenpatches_config.get('patch_size', 21)
        self.n_components = eigenpatches_config.get('n_components', 20)
        self.pyramid_levels = eigenpatches_config.get('pyramid_levels', 3)
        self.lambda_shape = predictor_config.get('lambda_shape', 0.1)
        self.max_iterations = predictor_config.get('max_iterations', 5)
        self.convergence_threshold = predictor_config.get('convergence_threshold', 0.5)
        self.search_radius = predictor_config.get('search_radius', [20, 10, 5])
        self.step_size = predictor_config.get('step_size', [2, 1])
        
        # Initialize original predictor
        self.predictor = TemplateLandmarkPredictor(
            patch_size=self.patch_size,
            n_components=self.n_components,
            use_multiscale=self.pyramid_levels > 1,
            pyramid_levels=self.pyramid_levels
        )
        
        # Set geometric constraint parameters
        if hasattr(self.predictor, 'lambda_shape'):
            self.predictor.lambda_shape = self.lambda_shape
        
        # Experimental features
        self.enable_detailed_logging = self.config.get('logging', {}).get('level', 'INFO') == 'DEBUG'
        self.experimental_features = self.config.get('experimental', {})
        self.enable_convergence_analysis = self.experimental_features.get('enable_convergence_analysis', True)
        self.enable_confidence_maps = self.experimental_features.get('enable_confidence_maps', False)
        
        # Statistics tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_iterations': 0.0,
            'average_processing_time': 0.0,
            'convergence_failures': 0
        }
        
        self.logger.info(f"Initialized ExperimentalLandmarkPredictor with patch_size={self.patch_size}, "
                        f"n_components={self.n_components}, lambda_shape={self.lambda_shape}")
    
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
                'pyramid_levels': 3
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 5,
                'convergence_threshold': 0.5,
                'search_radius': [20, 10, 5],
                'step_size': [2, 1]
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
            file_handler = logging.FileHandler(log_dir / 'predictor.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def train(self, images: List[np.ndarray], landmarks: List[np.ndarray]) -> None:
        """
        Train the landmark predictor.
        
        Args:
            images: List of training images
            landmarks: List of corresponding landmark coordinates
        """
        self.logger.info(f"Starting training with {len(images)} images")
        
        # Log training statistics
        if self.enable_detailed_logging:
            self._log_training_statistics(images, landmarks)
        
        # Train original predictor
        start_time = time.time()
        self.predictor.train(images, landmarks)
        training_time = time.time() - start_time
        
        # Log training completion
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Experimental: Analyze shape model
        if self.experimental_features.get('analyze_shape_model', True):
            self._analyze_shape_model()
    
    def _log_training_statistics(self, images: List[np.ndarray], 
                               landmarks: List[np.ndarray]) -> None:
        """Log detailed training statistics."""
        self.logger.debug(f"Training data statistics:")
        self.logger.debug(f"  Number of images: {len(images)}")
        self.logger.debug(f"  Number of landmarks per image: {landmarks[0].shape[0] if landmarks else 0}")
        
        # Landmark statistics
        if landmarks:
            all_landmarks = np.array(landmarks)
            landmark_means = np.mean(all_landmarks, axis=0)
            landmark_stds = np.std(all_landmarks, axis=0)
            
            self.logger.debug(f"  Landmark coordinate ranges:")
            for i, (mean, std) in enumerate(zip(landmark_means, landmark_stds)):
                self.logger.debug(f"    Landmark {i}: ({mean[0]:.2f}±{std[0]:.2f}, {mean[1]:.2f}±{std[1]:.2f})")
    
    def _analyze_shape_model(self) -> None:
        """Analyze shape model statistics for experimental insights."""
        self.logger.info("Analyzing shape model statistics")
        
        if hasattr(self.predictor, 'mean_shape') and self.predictor.mean_shape is not None:
            mean_shape = self.predictor.mean_shape
            self.logger.debug(f"Mean shape: {mean_shape.shape}")
            
            # Compute shape statistics
            # Reshape mean_shape to ensure it's (n_landmarks, 2)
            if mean_shape.ndim == 1:
                mean_shape_2d = mean_shape.reshape(-1, 2)
            else:
                mean_shape_2d = mean_shape
            
            centroid = np.mean(mean_shape_2d, axis=0)
            distances = np.linalg.norm(mean_shape_2d - centroid, axis=1)
            
            self.logger.debug(f"Shape centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
            self.logger.debug(f"Average distance from centroid: {np.mean(distances):.2f}±{np.std(distances):.2f}")
        
        if hasattr(self.predictor, 'shape_modes') and self.predictor.shape_modes is not None:
            shape_modes = self.predictor.shape_modes
            self.logger.debug(f"Shape modes: {shape_modes.shape}")
            
            # Analyze eigenvalues
            if hasattr(self.predictor, 'shape_eigenvalues') and self.predictor.shape_eigenvalues is not None:
                eigenvalues = self.predictor.shape_eigenvalues
                explained_variance = eigenvalues / np.sum(eigenvalues)
                cumulative_variance = np.cumsum(explained_variance)
                
                self.logger.debug(f"Shape eigenvalues: {eigenvalues[:5]}")
                self.logger.debug(f"Explained variance (top 5): {explained_variance[:5]}")
                self.logger.debug(f"Cumulative variance (top 5): {cumulative_variance[:5]}")
    
    def predict_landmarks(self, image: np.ndarray, 
                         initial_landmarks: Optional[np.ndarray] = None,
                         return_detailed: bool = False) -> PredictionResult:
        """
        Predict landmarks for a single image with detailed results.
        
        Args:
            image: Input image
            initial_landmarks: Optional initial landmark positions
            return_detailed: Whether to return detailed prediction results
            
        Returns:
            PredictionResult containing landmarks and metadata
        """
        start_time = time.time()
        
        if self.enable_detailed_logging:
            self.logger.debug(f"Predicting landmarks for image shape: {image.shape}")
        
        # Track refinement history if requested
        refinement_history = [] if return_detailed else None
        
        # Use original predictor
        try:
            if hasattr(self.predictor, 'predict_landmarks'):
                landmarks = self.predictor.predict_landmarks(image, initial_landmarks)
                confidence = None
                
                # Try to get confidence if available
                if hasattr(self.predictor, 'predict_with_confidence'):
                    try:
                        landmarks, confidence = self.predictor.predict_with_confidence(image, initial_landmarks)
                    except:
                        pass
                
            else:
                # Fallback method
                landmarks = self.predictor.predict(image, initial_landmarks)
                confidence = None
            
            # Update statistics
            self.prediction_stats['total_predictions'] += 1
            self.prediction_stats['successful_predictions'] += 1
            
            processing_time = time.time() - start_time
            self.prediction_stats['average_processing_time'] = (
                (self.prediction_stats['average_processing_time'] * (self.prediction_stats['total_predictions'] - 1) + 
                 processing_time) / self.prediction_stats['total_predictions']
            )
            
            # Create result
            result = PredictionResult(
                landmarks=landmarks,
                confidence=confidence,
                iterations=getattr(self.predictor, '_last_iterations', 0),
                convergence_error=getattr(self.predictor, '_last_convergence_error', 0.0),
                processing_time=processing_time,
                geometric_constraints_applied=True,
                initial_landmarks=initial_landmarks,
                refinement_history=refinement_history
            )
            
            if self.enable_detailed_logging:
                self.logger.debug(f"Prediction completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            self.prediction_stats['total_predictions'] += 1
            self.prediction_stats['convergence_failures'] += 1
            
            # Return empty result
            return PredictionResult(
                landmarks=np.zeros((15, 2)),  # Default to 15 landmarks
                confidence=None,
                processing_time=time.time() - start_time
            )
    
    def predict_with_confidence(self, image: np.ndarray, 
                              initial_landmarks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict landmarks with confidence scores.
        
        Args:
            image: Input image
            initial_landmarks: Optional initial landmark positions
            
        Returns:
            Tuple of (landmarks, confidence_scores)
        """
        result = self.predict_landmarks(image, initial_landmarks, return_detailed=True)
        
        if result.confidence is not None:
            return result.landmarks, result.confidence
        
        # Generate confidence scores if not available
        confidence_scores = self._estimate_confidence(image, result.landmarks)
        return result.landmarks, confidence_scores
    
    def _estimate_confidence(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Estimate confidence scores for predicted landmarks.
        
        Args:
            image: Input image
            landmarks: Predicted landmark coordinates
            
        Returns:
            Confidence scores for each landmark
        """
        if not self.enable_confidence_maps:
            return np.ones(len(landmarks))
        
        # Simple confidence estimation based on template matching scores
        confidences = []
        
        for i, (x, y) in enumerate(landmarks):
            # Get local patch quality
            try:
                if hasattr(self.predictor, 'eigenpatch_model'):
                    scores = self.predictor.eigenpatch_model.score_patches(
                        image, [(x, y)], i
                    )
                    # Convert score to confidence (higher score = lower confidence)
                    confidence = 1.0 / (1.0 + scores[0]) if scores else 0.5
                else:
                    confidence = 0.5
            except:
                confidence = 0.5
            
            confidences.append(confidence)
        
        return np.array(confidences)
    
    def visualize_geometric_constraints(self, landmarks: np.ndarray, 
                                      shape_parameters: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Visualize geometric constraints and shape model compliance.
        
        Args:
            landmarks: Landmark coordinates
            shape_parameters: Optional shape parameters
            
        Returns:
            Dictionary with constraint analysis
        """
        if not hasattr(self.predictor, 'mean_shape') or self.predictor.mean_shape is None:
            return {'error': 'Shape model not available'}
        
        # Align landmarks to mean shape
        aligned_landmarks = self._align_to_mean_shape(landmarks)
        
        # Compute shape parameters
        if shape_parameters is None and hasattr(self.predictor, 'shape_modes'):
            shape_vector = (aligned_landmarks - self.predictor.mean_shape).flatten()
            shape_parameters = np.dot(self.predictor.shape_modes.T, shape_vector)
        
        # Analyze constraints
        constraint_analysis = {
            'landmarks': landmarks,
            'aligned_landmarks': aligned_landmarks,
            'shape_parameters': shape_parameters,
            'mean_shape': self.predictor.mean_shape,
            'constraint_violations': []
        }
        
        # Check geometric constraints
        if shape_parameters is not None and hasattr(self.predictor, 'shape_eigenvalues'):
            eigenvalues = self.predictor.shape_eigenvalues
            max_deviations = 3 * np.sqrt(eigenvalues)
            
            for i, (param, max_dev) in enumerate(zip(shape_parameters, max_deviations)):
                if abs(param) > max_dev:
                    constraint_analysis['constraint_violations'].append({
                        'parameter_index': i,
                        'value': param,
                        'max_allowed': max_dev,
                        'violation_ratio': abs(param) / max_dev
                    })
        
        return constraint_analysis
    
    def _align_to_mean_shape(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Align landmarks to mean shape using Procrustes analysis.
        
        Args:
            landmarks: Input landmark coordinates
            
        Returns:
            Aligned landmark coordinates
        """
        if not hasattr(self.predictor, 'mean_shape') or self.predictor.mean_shape is None:
            return landmarks
        
        # Simple alignment (this is a simplified version)
        mean_shape = self.predictor.mean_shape
        
        # Center both shapes
        landmarks_centered = landmarks - np.mean(landmarks, axis=0)
        mean_centered = mean_shape - np.mean(mean_shape, axis=0)
        
        # Compute scale
        scale = np.sqrt(np.sum(mean_centered**2) / np.sum(landmarks_centered**2))
        
        # Apply scale
        aligned = landmarks_centered * scale + np.mean(mean_shape, axis=0)
        
        return aligned
    
    def analyze_convergence(self, image: np.ndarray, 
                          initial_landmarks: np.ndarray,
                          max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze convergence behavior for a specific image.
        
        Args:
            image: Input image
            initial_landmarks: Initial landmark positions
            max_iterations: Maximum iterations to analyze
            
        Returns:
            Dictionary with convergence analysis
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # Store original settings
        original_max_iter = self.max_iterations
        original_threshold = self.convergence_threshold
        
        # Set up for convergence analysis
        self.max_iterations = max_iterations
        self.convergence_threshold = 0.001  # Very small threshold
        
        convergence_history = []
        error_history = []
        
        try:
            # Run prediction with detailed tracking
            result = self.predict_landmarks(image, initial_landmarks, return_detailed=True)
            
            # Analyze convergence
            analysis = {
                'final_landmarks': result.landmarks,
                'iterations': result.iterations,
                'convergence_error': result.convergence_error,
                'processing_time': result.processing_time,
                'converged': result.convergence_error < original_threshold,
                'convergence_history': convergence_history,
                'error_history': error_history
            }
            
        except Exception as e:
            analysis = {
                'error': str(e),
                'converged': False
            }
        
        finally:
            # Restore original settings
            self.max_iterations = original_max_iter
            self.convergence_threshold = original_threshold
        
        return analysis
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get prediction statistics and performance metrics.
        
        Returns:
            Dictionary containing prediction statistics
        """
        stats = dict(self.prediction_stats)
        
        # Add derived metrics
        if stats['total_predictions'] > 0:
            stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions']
            stats['failure_rate'] = stats['convergence_failures'] / stats['total_predictions']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Add configuration
        stats['configuration'] = {
            'patch_size': self.patch_size,
            'n_components': self.n_components,
            'pyramid_levels': self.pyramid_levels,
            'lambda_shape': self.lambda_shape,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold
        }
        
        return stats
    
    def save(self, filepath: str) -> None:
        """
        Save the predictor to file.
        
        Args:
            filepath: Path to save the predictor
        """
        self.logger.info(f"Saving predictor to: {filepath}")
        
        # Save original predictor
        self.predictor.save(filepath)
        
        # Save experimental metadata
        metadata_path = Path(filepath).with_suffix('.meta.yaml')
        metadata = {
            'config': self.config,
            'model_type': 'ExperimentalLandmarkPredictor',
            'version': '1.0.0',
            'parameters': {
                'patch_size': self.patch_size,
                'n_components': self.n_components,
                'pyramid_levels': self.pyramid_levels,
                'lambda_shape': self.lambda_shape,
                'max_iterations': self.max_iterations,
                'convergence_threshold': self.convergence_threshold
            },
            'statistics': self.prediction_stats
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        self.logger.info(f"Predictor and metadata saved successfully")
    
    def load(self, filepath: str) -> None:
        """
        Load the predictor from file.
        
        Args:
            filepath: Path to load the predictor from
        """
        self.logger.info(f"Loading predictor from: {filepath}")
        
        # Load original predictor
        self.predictor.load(filepath)
        
        # Load experimental metadata if available
        metadata_path = Path(filepath).with_suffix('.meta.yaml')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            # Restore statistics
            if 'statistics' in metadata:
                self.prediction_stats.update(metadata['statistics'])
            
            self.logger.info(f"Loaded metadata for predictor version: {metadata.get('version', 'unknown')}")
        
        self.logger.info("Predictor loaded successfully")