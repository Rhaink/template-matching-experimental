"""
Unit tests for experimental landmark predictor module.

Tests the ExperimentalLandmarkPredictor class and its integration with the original
landmark predictor implementation.
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

from core.experimental_predictor import ExperimentalLandmarkPredictor, PredictionResult


class TestExperimentalLandmarkPredictor(unittest.TestCase):
    """Test cases for ExperimentalLandmarkPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.test_config = {
            'eigenpatches': {
                'patch_size': 15,  # Smaller for faster testing
                'n_components': 5,
                'pyramid_levels': 2,
                'scale_factor': 0.5
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 3,  # Reduced for faster testing
                'convergence_threshold': 0.5,
                'search_radius': [10, 5],
                'step_size': [2, 1]
            },
            'logging': {
                'level': 'WARNING',  # Reduce log noise in tests
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Create synthetic test data
        self.test_images = self._create_synthetic_images(3)
        self.test_landmarks = self._create_synthetic_landmarks(3)
    
    def _create_synthetic_images(self, n_images: int) -> list:
        """Create synthetic test images."""
        images = []
        np.random.seed(42)  # For reproducible tests
        
        for i in range(n_images):
            # Create 128x128 synthetic image (larger for landmark prediction)
            img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
            
            # Add some structured patterns for landmarks
            for j in range(15):  # 15 landmarks
                x = 20 + j * 6
                y = 20 + (j % 5) * 20
                # Add small bright spots as landmark features
                if x < 124 and y < 124:
                    img[y:y+4, x:x+4] = 250
            
            images.append(img)
        
        return images
    
    def _create_synthetic_landmarks(self, n_images: int) -> list:
        """Create synthetic landmark coordinates matching image features."""
        landmarks_list = []
        np.random.seed(42)  # For reproducible tests
        
        for i in range(n_images):
            # Create landmarks that roughly correspond to the features added in images
            landmarks = []
            for j in range(15):
                x = 20 + j * 6 + np.random.normal(0, 1)  # Add some noise
                y = 20 + (j % 5) * 20 + np.random.normal(0, 1)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            landmarks = np.clip(landmarks, 5, 123)  # Keep within image bounds
            landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        
        self.assertEqual(predictor.patch_size, 15)
        self.assertEqual(predictor.n_components, 5)
        self.assertEqual(predictor.pyramid_levels, 2)
        self.assertEqual(predictor.lambda_shape, 0.1)
        self.assertEqual(predictor.max_iterations, 3)
    
    def test_training(self):
        """Test predictor training."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        
        # Training should complete without errors
        predictor.train(self.test_images, self.test_landmarks)
        
        # Check training statistics
        stats = predictor.get_prediction_statistics()
        self.assertEqual(stats['configuration']['patch_size'], 15)
        self.assertEqual(stats['configuration']['n_components'], 5)
    
    def test_prediction_basic(self):
        """Test basic landmark prediction."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        # Predict on first test image
        test_image = self.test_images[0]
        result = predictor.predict_landmarks(test_image)
        
        # Check result structure
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.landmarks.shape, (15, 2))
        self.assertGreater(result.processing_time, 0)
        self.assertGreaterEqual(result.iterations, 0)
    
    def test_prediction_with_confidence(self):
        """Test landmark prediction with confidence scores."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        test_image = self.test_images[0]
        landmarks, confidence = predictor.predict_with_confidence(test_image)
        
        self.assertEqual(landmarks.shape, (15, 2))
        self.assertEqual(confidence.shape, (15,))
        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 1))
    
    def test_prediction_with_initial_landmarks(self):
        """Test prediction with initial landmark positions."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        test_image = self.test_images[0]
        initial_landmarks = self.test_landmarks[0].copy()
        # Add some noise to initial landmarks
        initial_landmarks += np.random.normal(0, 2, initial_landmarks.shape)
        
        result = predictor.predict_landmarks(test_image, initial_landmarks)
        
        self.assertEqual(result.landmarks.shape, (15, 2))
        self.assertIsNotNone(result.initial_landmarks)
        np.testing.assert_array_equal(result.initial_landmarks, initial_landmarks)
    
    def test_detailed_prediction_results(self):
        """Test detailed prediction results."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        test_image = self.test_images[0]
        result = predictor.predict_landmarks(test_image, return_detailed=True)
        
        # Check all detailed fields
        self.assertIsInstance(result.landmarks, np.ndarray)
        self.assertIsInstance(result.processing_time, float)
        self.assertIsInstance(result.iterations, int)
        self.assertIsInstance(result.convergence_error, float)
        self.assertIsInstance(result.geometric_constraints_applied, bool)
    
    def test_geometric_constraints_visualization(self):
        """Test geometric constraints visualization."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        # Need to predict first to have shape statistics
        test_image = self.test_images[0]
        result = predictor.predict_landmarks(test_image)
        
        # Test constraint visualization
        constraint_analysis = predictor.visualize_geometric_constraints(result.landmarks)
        
        if 'error' not in constraint_analysis:
            self.assertIn('landmarks', constraint_analysis)
            self.assertIn('aligned_landmarks', constraint_analysis)
            self.assertIn('constraint_violations', constraint_analysis)
    
    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        test_image = self.test_images[0]
        initial_landmarks = self.test_landmarks[0].copy()
        
        convergence_analysis = predictor.analyze_convergence(
            test_image, initial_landmarks, max_iterations=2
        )
        
        if 'error' not in convergence_analysis:
            self.assertIn('final_landmarks', convergence_analysis)
            self.assertIn('iterations', convergence_analysis)
            self.assertIn('convergence_error', convergence_analysis)
            self.assertIn('converged', convergence_analysis)
    
    def test_prediction_statistics_tracking(self):
        """Test prediction statistics tracking."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        # Make several predictions
        for image in self.test_images:
            predictor.predict_landmarks(image)
        
        stats = predictor.get_prediction_statistics()
        
        self.assertEqual(stats['total_predictions'], len(self.test_images))
        self.assertGreater(stats['average_processing_time'], 0)
        self.assertGreaterEqual(stats['success_rate'], 0)
        self.assertLessEqual(stats['success_rate'], 1)
    
    def test_save_and_load_with_metadata(self):
        """Test saving and loading with experimental metadata."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        # Make some predictions to generate statistics
        for image in self.test_images:
            predictor.predict_landmarks(image)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_model_path = f.name
        
        try:
            # Save predictor
            predictor.save(temp_model_path)
            
            # Check metadata file exists
            metadata_path = Path(temp_model_path).with_suffix('.meta.yaml')
            self.assertTrue(metadata_path.exists())
            
            # Load metadata and verify content
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            self.assertEqual(metadata['model_type'], 'ExperimentalLandmarkPredictor')
            self.assertIn('statistics', metadata)
            self.assertIn('parameters', metadata)
            
            # Load predictor
            new_predictor = ExperimentalLandmarkPredictor(config=self.test_config)
            new_predictor.load(temp_model_path)
            
            # Verify statistics were restored
            stats = new_predictor.get_prediction_statistics()
            self.assertEqual(stats['total_predictions'], len(self.test_images))
            
        finally:
            # Cleanup
            for path in [temp_model_path, temp_model_path.replace('.pkl', '.meta.yaml')]:
                if Path(path).exists():
                    Path(path).unlink()
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        # Test with invalid image
        invalid_image = np.zeros((10, 10), dtype=np.uint8)  # Too small
        result = predictor.predict_landmarks(invalid_image)
        
        # Should return a valid result structure even if prediction fails
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.landmarks.shape, (15, 2))
    
    def test_batch_prediction_consistency(self):
        """Test that batch predictions are consistent."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        test_image = self.test_images[0]
        
        # Make multiple predictions on same image
        results = []
        for _ in range(3):
            result = predictor.predict_landmarks(test_image)
            results.append(result.landmarks)
        
        # Results should be identical (assuming deterministic implementation)
        # or at least very close if there's any randomness
        for i in range(1, len(results)):
            diff = np.mean(np.linalg.norm(results[0] - results[i], axis=1))
            self.assertLess(diff, 5.0)  # Allow some tolerance for numerical differences


class TestPredictionResult(unittest.TestCase):
    """Test cases for PredictionResult dataclass."""
    
    def test_prediction_result_creation(self):
        """Test creating PredictionResult instances."""
        landmarks = np.random.rand(15, 2)
        confidence = np.random.rand(15)
        
        result = PredictionResult(
            landmarks=landmarks,
            confidence=confidence,
            iterations=5,
            convergence_error=0.5,
            processing_time=1.2,
            geometric_constraints_applied=True
        )
        
        self.assertEqual(result.iterations, 5)
        self.assertEqual(result.convergence_error, 0.5)
        self.assertEqual(result.processing_time, 1.2)
        self.assertTrue(result.geometric_constraints_applied)
        np.testing.assert_array_equal(result.landmarks, landmarks)
        np.testing.assert_array_equal(result.confidence, confidence)
    
    def test_prediction_result_defaults(self):
        """Test PredictionResult with default values."""
        landmarks = np.random.rand(15, 2)
        
        result = PredictionResult(landmarks=landmarks)
        
        self.assertIsNone(result.confidence)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.convergence_error, 0.0)
        self.assertEqual(result.processing_time, 0.0)
        self.assertFalse(result.geometric_constraints_applied)


class TestPredictorIntegration(unittest.TestCase):
    """Integration tests for predictor with original implementation."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_config = {
            'eigenpatches': {
                'patch_size': 11,
                'n_components': 3,
                'pyramid_levels': 2,
                'scale_factor': 0.5
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 2,
                'convergence_threshold': 1.0,
                'search_radius': [8, 4],
                'step_size': [2, 1]
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Create minimal test data
        self.test_images = [
            np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            for _ in range(2)
        ]
        self.test_landmarks = [
            np.random.uniform(10, 90, (15, 2))
            for _ in range(2)
        ]
    
    def test_multiscale_prediction(self):
        """Test multiscale prediction behavior."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.test_images, self.test_landmarks)
        
        # Should use multiscale due to pyramid_levels > 1
        self.assertTrue(predictor.predictor.use_multiscale)
        
        # Test prediction
        result = predictor.predict_landmarks(self.test_images[0])
        self.assertEqual(result.landmarks.shape, (15, 2))
    
    def test_parameter_effect_on_convergence(self):
        """Test how different parameters affect convergence."""
        # Test with strict convergence
        strict_config = self.test_config.copy()
        strict_config['landmark_predictor']['convergence_threshold'] = 0.1
        strict_config['landmark_predictor']['max_iterations'] = 10
        
        # Test with loose convergence
        loose_config = self.test_config.copy()
        loose_config['landmark_predictor']['convergence_threshold'] = 5.0
        loose_config['landmark_predictor']['max_iterations'] = 1
        
        strict_predictor = ExperimentalLandmarkPredictor(config=strict_config)
        loose_predictor = ExperimentalLandmarkPredictor(config=loose_config)
        
        # Train both
        strict_predictor.train(self.test_images, self.test_landmarks)
        loose_predictor.train(self.test_images, self.test_landmarks)
        
        # Test predictions
        test_image = self.test_images[0]
        strict_result = strict_predictor.predict_landmarks(test_image)
        loose_result = loose_predictor.predict_landmarks(test_image)
        
        # Both should produce valid results
        self.assertEqual(strict_result.landmarks.shape, (15, 2))
        self.assertEqual(loose_result.landmarks.shape, (15, 2))
        
        # Strict might take more iterations (but not guaranteed)
        self.assertGreaterEqual(strict_result.iterations, 0)
        self.assertGreaterEqual(loose_result.iterations, 0)


if __name__ == '__main__':
    unittest.main()