"""
Test fixtures for matching_experimental test suite.

This module provides synthetic test data, configurations, and utilities
for testing the experimental template matching system.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from pathlib import Path


def create_synthetic_images(n_images: int = 5, 
                          image_size: Tuple[int, int] = (100, 100),
                          seed: int = 42) -> List[np.ndarray]:
    """
    Create synthetic test images with landmark-like features.
    
    Args:
        n_images: Number of images to create
        image_size: Size of images as (height, width)
        seed: Random seed for reproducibility
        
    Returns:
        List of synthetic images
    """
    np.random.seed(seed)
    images = []
    
    for i in range(n_images):
        # Create base image with noise
        img = np.random.randint(50, 200, image_size, dtype=np.uint8)
        
        # Add landmark-like features
        h, w = image_size
        n_features = 15
        
        for j in range(n_features):
            # Position features in a grid-like pattern
            x = int((j % 5) * w / 6 + w / 12)
            y = int((j // 5) * h / 4 + h / 8)
            
            # Add some randomness
            x += np.random.randint(-5, 6)
            y += np.random.randint(-5, 6)
            
            # Ensure within bounds
            x = max(2, min(w-3, x))
            y = max(2, min(h-3, y))
            
            # Add bright feature
            intensity = 220 + np.random.randint(-20, 21)
            img[y-1:y+2, x-1:x+2] = intensity
            
        # Add some noise
        noise = np.random.normal(0, 10, image_size)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        images.append(img)
    
    return images


def create_synthetic_landmarks(n_images: int = 5,
                             image_size: Tuple[int, int] = (100, 100),
                             n_landmarks: int = 15,
                             seed: int = 42) -> List[np.ndarray]:
    """
    Create synthetic landmark coordinates corresponding to synthetic images.
    
    Args:
        n_images: Number of landmark sets to create
        image_size: Size of corresponding images
        n_landmarks: Number of landmarks per image
        seed: Random seed for reproducibility
        
    Returns:
        List of landmark coordinate arrays
    """
    np.random.seed(seed)
    landmarks_list = []
    
    h, w = image_size
    
    for i in range(n_images):
        landmarks = []
        
        for j in range(n_landmarks):
            # Position landmarks in same pattern as image features
            x = (j % 5) * w / 6 + w / 12
            y = (j // 5) * h / 4 + h / 8
            
            # Add same randomness as image features
            x += np.random.randint(-5, 6)
            y += np.random.randint(-5, 6)
            
            # Add landmark-specific noise
            x += np.random.normal(0, 1)
            y += np.random.normal(0, 1)
            
            # Ensure within bounds with margin
            x = max(5, min(w-5, x))
            y = max(5, min(h-5, y))
            
            landmarks.append([x, y])
        
        landmarks_array = np.array(landmarks)
        landmarks_list.append(landmarks_array)
    
    return landmarks_list


def create_test_configuration(minimal: bool = False) -> Dict[str, Any]:
    """
    Create test configuration for experimental components.
    
    Args:
        minimal: If True, create minimal configuration for faster testing
        
    Returns:
        Configuration dictionary
    """
    if minimal:
        return {
            'eigenpatches': {
                'patch_size': 11,
                'n_components': 3,
                'pyramid_levels': 1,
                'scale_factor': 0.5
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 2,
                'convergence_threshold': 1.0,
                'search_radius': [8],
                'step_size': [2]
            },
            'evaluation': {
                'confidence_threshold': 0.7,
                'statistical_tests': False,
                'generate_visualizations': False,
                'save_detailed_results': False
            },
            'logging': {
                'level': 'ERROR',  # Minimal logging for tests
                'console_logging': False,
                'file_logging': False
            }
        }
    else:
        return {
            'eigenpatches': {
                'patch_size': 15,
                'n_components': 5,
                'pyramid_levels': 2,
                'scale_factor': 0.5
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 3,
                'convergence_threshold': 0.5,
                'search_radius': [10, 5],
                'step_size': [2, 1]
            },
            'evaluation': {
                'confidence_threshold': 0.7,
                'statistical_tests': True,
                'generate_visualizations': False,  # Disabled for testing
                'save_detailed_results': False     # Disabled for testing
            },
            'image_processing': {
                'image_size': 100,
                'coordinate_scale_factor': 1.56,
                'normalize_intensity': True
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }


def create_performance_test_data(error_level: float = 3.0,
                               n_images: int = 10,
                               image_size: Tuple[int, int] = (100, 100),
                               seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Create test data for performance evaluation with controlled error level.
    
    Args:
        error_level: Standard deviation of error to add to predictions
        n_images: Number of test images/landmarks
        image_size: Size of images
        seed: Random seed
        
    Returns:
        Tuple of (ground_truth, predictions, images)
    """
    np.random.seed(seed)
    
    # Create ground truth landmarks
    ground_truth = create_synthetic_landmarks(n_images, image_size, seed=seed)
    
    # Create predictions with controlled error
    predictions = []
    for gt in ground_truth:
        error = np.random.normal(0, error_level, gt.shape)
        pred = gt + error
        # Ensure predictions stay within reasonable bounds
        h, w = image_size
        pred[:, 0] = np.clip(pred[:, 0], 5, w-5)
        pred[:, 1] = np.clip(pred[:, 1], 5, h-5)
        predictions.append(pred)
    
    # Create corresponding images
    images = create_synthetic_images(n_images, image_size, seed=seed)
    
    return ground_truth, predictions, images


def create_baseline_comparison_data(baseline_error: float = 5.63,
                                  n_images: int = 20,
                                  seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create test data that simulates baseline performance for comparison tests.
    
    Args:
        baseline_error: Target mean error to simulate
        n_images: Number of test cases
        seed: Random seed
        
    Returns:
        Tuple of (ground_truth, predictions)
    """
    np.random.seed(seed)
    
    ground_truth = create_synthetic_landmarks(n_images, seed=seed)
    predictions = []
    
    for gt in ground_truth:
        # Add error with target mean
        error = np.random.normal(0, baseline_error/3, gt.shape)  # 3-sigma rule
        pred = gt + error
        predictions.append(pred)
    
    return ground_truth, predictions


class TestDataGenerator:
    """
    Utility class for generating various types of test data.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_training_data(self, n_images: int = 10, 
                             image_size: Tuple[int, int] = (100, 100)) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate training data with realistic characteristics."""
        images = create_synthetic_images(n_images, image_size, self.seed)
        landmarks = create_synthetic_landmarks(n_images, image_size, seed=self.seed)
        return images, landmarks
    
    def generate_test_data(self, n_images: int = 5,
                         image_size: Tuple[int, int] = (100, 100)) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate test data with different characteristics than training."""
        # Use different seed for test data
        test_seed = self.seed + 1000
        images = create_synthetic_images(n_images, image_size, test_seed)
        landmarks = create_synthetic_landmarks(n_images, image_size, seed=test_seed)
        return images, landmarks
    
    def generate_challenging_data(self, n_images: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate challenging test cases with noise and distortions."""
        images = []
        landmarks = []
        
        base_images, base_landmarks = self.generate_test_data(n_images)
        
        for i, (img, lm) in enumerate(zip(base_images, base_landmarks)):
            # Add various challenges
            challenging_img = img.copy()
            challenging_lm = lm.copy()
            
            if i % 3 == 0:
                # Add blur
                challenging_img = cv2.GaussianBlur(challenging_img, (3, 3), 1.0)
            elif i % 3 == 1:
                # Add noise
                noise = np.random.normal(0, 20, img.shape)
                challenging_img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            else:
                # Add contrast changes
                challenging_img = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
            
            # Add landmark position uncertainty
            landmark_noise = np.random.normal(0, 2, lm.shape)
            challenging_lm += landmark_noise
            
            # Keep within bounds
            h, w = img.shape
            challenging_lm[:, 0] = np.clip(challenging_lm[:, 0], 5, w-5)
            challenging_lm[:, 1] = np.clip(challenging_lm[:, 1], 5, h-5)
            
            images.append(challenging_img)
            landmarks.append(challenging_lm)
        
        return images, landmarks


# Pre-defined test datasets for common use cases
SMALL_TEST_CONFIG = create_test_configuration(minimal=True)
STANDARD_TEST_CONFIG = create_test_configuration(minimal=False)

# Quick access to common test data
def get_quick_test_data() -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """Get quick test data for simple unit tests."""
    images = create_synthetic_images(3, (64, 64))
    landmarks = create_synthetic_landmarks(3, (64, 64))
    config = SMALL_TEST_CONFIG
    return images, landmarks, config


def get_integration_test_data() -> Tuple[List[np.ndarray], List[np.ndarray], 
                                       List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """Get test data for integration tests."""
    generator = TestDataGenerator(42)
    train_images, train_landmarks = generator.generate_training_data(5)
    test_images, test_landmarks = generator.generate_test_data(3)
    config = STANDARD_TEST_CONFIG
    return train_images, train_landmarks, test_images, test_landmarks, config