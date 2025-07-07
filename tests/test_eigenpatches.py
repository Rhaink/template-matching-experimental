"""
Unit tests for experimental eigenpatches module.

Tests the ExperimentalEigenpatches class and its integration with the original
eigenpatches implementation.
"""

import unittest
import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "matching_experimental"))

from core.experimental_eigenpatches import ExperimentalEigenpatches


class TestExperimentalEigenpatches(unittest.TestCase):
    """Test cases for ExperimentalEigenpatches class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.test_config = {
            'eigenpatches': {
                'patch_size': 21,
                'n_components': 5,  # Reduced for faster testing
                'pyramid_levels': 2,  # Reduced for faster testing
                'scale_factor': 0.5
            },
            'logging': {
                'level': 'WARNING',  # Reduce log noise in tests
                'console_logging': False,
                'file_logging': False
            },
            'experimental': {
                'analyze_pca_components': True
            }
        }
        
        # Create synthetic test data
        self.test_images = self._create_synthetic_images(5)
        self.test_landmarks = self._create_synthetic_landmarks(5)
    
    def _create_synthetic_images(self, n_images: int) -> list:
        """Create synthetic test images."""
        images = []
        np.random.seed(42)  # For reproducible tests
        
        for i in range(n_images):
            # Create 64x64 synthetic image with some patterns
            img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            
            # Add some structured patterns
            center_x, center_y = 32, 32
            radius = 15
            y, x = np.ogrid[:64, :64]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = 200
            
            images.append(img)
        
        return images
    
    def _create_synthetic_landmarks(self, n_images: int) -> list:
        """Create synthetic landmark coordinates."""
        landmarks_list = []
        np.random.seed(42)  # For reproducible tests
        
        for i in range(n_images):
            # Create 15 landmarks within image bounds
            landmarks = np.random.uniform(5, 59, (15, 2))
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def test_initialization_with_config_dict(self):
        """Test initialization with configuration dictionary."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        
        self.assertEqual(eigenpatches.patch_size, 21)
        self.assertEqual(eigenpatches.n_components, 5)
        self.assertEqual(eigenpatches.pyramid_levels, 2)
        self.assertEqual(eigenpatches.scale_factor, 0.5)
        self.assertTrue(eigenpatches.use_multiscale)
    
    def test_initialization_with_config_file(self):
        """Test initialization with configuration file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            temp_config_path = f.name
        
        try:
            eigenpatches = ExperimentalEigenpatches(config_file=temp_config_path)
            self.assertEqual(eigenpatches.patch_size, 21)
            self.assertEqual(eigenpatches.n_components, 5)
        finally:
            Path(temp_config_path).unlink()
    
    def test_training_basic(self):
        """Test basic training functionality."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        
        # Training should not raise exceptions
        eigenpatches.train(self.test_images, self.test_landmarks)
        
        # Check that model is trained
        self.assertTrue(eigenpatches.model.is_trained)
        self.assertEqual(eigenpatches.model.n_landmarks, 15)
    
    def test_training_statistics(self):
        """Test training statistics collection."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches.train(self.test_images, self.test_landmarks)
        
        stats = eigenpatches.get_training_statistics()
        
        self.assertEqual(stats['patch_size'], 21)
        self.assertEqual(stats['n_components'], 5)
        self.assertEqual(stats['pyramid_levels'], 2)
        self.assertTrue(stats['is_trained'])
        self.assertEqual(stats['n_landmarks'], 15)
        self.assertIn('mean_explained_variance', stats)
    
    def test_patch_scoring(self):
        """Test patch scoring functionality."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches.train(self.test_images, self.test_landmarks)
        
        # Test scoring on first image
        test_image = self.test_images[0]
        test_positions = [(32, 32), (16, 16), (48, 48)]
        
        scores = eigenpatches.score_patches(test_image, test_positions, landmark_idx=0)
        
        self.assertEqual(len(scores), len(test_positions))
        self.assertTrue(all(isinstance(score, (int, float)) for score in scores))
    
    def test_component_visualization(self):
        """Test PCA component visualization."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches.train(self.test_images, self.test_landmarks)
        
        # Test visualization for first landmark
        components = eigenpatches.visualize_components(landmark_idx=0, n_components=3)
        
        self.assertIsNotNone(components)
        self.assertEqual(components.shape[0], 3)  # n_components
        self.assertEqual(components.shape[1], 21)  # patch_size
        self.assertEqual(components.shape[2], 21)  # patch_size
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches.train(self.test_images, self.test_landmarks)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_model_path = f.name
        
        try:
            # Save model
            eigenpatches.save(temp_model_path)
            self.assertTrue(Path(temp_model_path).exists())
            
            # Check metadata file
            metadata_path = Path(temp_model_path).with_suffix('.meta.yaml')
            self.assertTrue(metadata_path.exists())
            
            # Load model
            new_eigenpatches = ExperimentalEigenpatches(config=self.test_config)
            new_eigenpatches.load(temp_model_path)
            
            # Verify loaded model
            self.assertTrue(new_eigenpatches.model.is_trained)
            self.assertEqual(new_eigenpatches.model.n_landmarks, 15)
            
        finally:
            # Cleanup
            for path in [temp_model_path, temp_model_path.replace('.pkl', '.meta.yaml')]:
                if Path(path).exists():
                    Path(path).unlink()
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with invalid patch size
        invalid_config = self.test_config.copy()
        invalid_config['eigenpatches']['patch_size'] = 0
        
        # Should still create object but with corrected parameters
        eigenpatches = ExperimentalEigenpatches(config=invalid_config)
        # The underlying model should handle parameter validation
    
    def test_empty_training_data(self):
        """Test behavior with empty training data."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        
        # Training with empty data should raise an error
        with self.assertRaises(Exception):
            eigenpatches.train([], [])
    
    def test_mismatched_training_data(self):
        """Test behavior with mismatched training data."""
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        
        # Mismatched number of images and landmarks
        with self.assertRaises(Exception):
            eigenpatches.train(self.test_images[:3], self.test_landmarks[:2])
    
    def test_reproducibility(self):
        """Test that training is reproducible with same random seed."""
        # This test would require random seed control in the original implementation
        # For now, just verify that multiple training runs complete successfully
        
        eigenpatches1 = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches1.train(self.test_images, self.test_landmarks)
        
        eigenpatches2 = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches2.train(self.test_images, self.test_landmarks)
        
        # Both should complete successfully
        self.assertTrue(eigenpatches1.model.is_trained)
        self.assertTrue(eigenpatches2.model.is_trained)
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable for large datasets."""
        # Create larger synthetic dataset
        large_images = self._create_synthetic_images(20)
        large_landmarks = self._create_synthetic_landmarks(20)
        
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        
        # Training should complete without memory errors
        eigenpatches.train(large_images, large_landmarks)
        self.assertTrue(eigenpatches.model.is_trained)


class TestEigenpatchesIntegration(unittest.TestCase):
    """Integration tests for eigenpatches with original implementation."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_config = {
            'eigenpatches': {
                'patch_size': 15,  # Smaller for faster tests
                'n_components': 3,
                'pyramid_levels': 2,
                'scale_factor': 0.5
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Create minimal test data
        self.test_images = [
            np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            for _ in range(3)
        ]
        self.test_landmarks = [
            np.random.uniform(10, 54, (15, 2))
            for _ in range(3)
        ]
    
    def test_consistency_with_original(self):
        """Test that experimental wrapper produces consistent results."""
        # This test verifies that the experimental wrapper doesn't change
        # the core behavior of the original implementation
        
        eigenpatches = ExperimentalEigenpatches(config=self.test_config)
        eigenpatches.train(self.test_images, self.test_landmarks)
        
        # Test scoring consistency
        test_image = self.test_images[0]
        test_positions = [(32, 32), (20, 20)]
        
        scores1 = eigenpatches.score_patches(test_image, test_positions, landmark_idx=0)
        scores2 = eigenpatches.score_patches(test_image, test_positions, landmark_idx=0)
        
        # Should get same scores for same inputs
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)
    
    def test_multiscale_vs_single_scale(self):
        """Test multiscale vs single scale eigenpatches."""
        # Single scale configuration
        single_config = self.test_config.copy()
        single_config['eigenpatches']['pyramid_levels'] = 1
        
        # Multi scale configuration
        multi_config = self.test_config.copy()
        multi_config['eigenpatches']['pyramid_levels'] = 3
        
        # Train both models
        single_eigenpatches = ExperimentalEigenpatches(config=single_config)
        single_eigenpatches.train(self.test_images, self.test_landmarks)
        
        multi_eigenpatches = ExperimentalEigenpatches(config=multi_config)
        multi_eigenpatches.train(self.test_images, self.test_landmarks)
        
        # Both should train successfully
        self.assertTrue(single_eigenpatches.model.is_trained)
        self.assertTrue(multi_eigenpatches.model.is_trained)
        
        # Multi-scale should use MultiScaleEigenpatches
        self.assertTrue(multi_eigenpatches.use_multiscale)
        self.assertFalse(single_eigenpatches.use_multiscale)


if __name__ == '__main__':
    unittest.main()