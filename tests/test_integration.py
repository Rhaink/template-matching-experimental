"""
Integration tests for the complete experimental template matching pipeline.

Tests the integration between all components and validates the complete workflow
from training through evaluation.
"""

import unittest
import numpy as np
import tempfile
import yaml
import pickle
from pathlib import Path
import sys
import shutil

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "matching_experimental"))

from core.experimental_predictor import ExperimentalLandmarkPredictor
from core.experimental_evaluator import ExperimentalEvaluator


class TestPipelineIntegration(unittest.TestCase):
    """Test complete pipeline integration."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test configuration
        self.test_config = {
            'eigenpatches': {
                'patch_size': 11,  # Small for faster testing
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
            'evaluation': {
                'confidence_threshold': 0.7,
                'statistical_tests': True,
                'generate_visualizations': False,  # Disable for testing
                'save_detailed_results': False
            },
            'image_processing': {
                'image_size': 100,
                'coordinate_scale_factor': 1.56,  # 100/64
                'normalize_intensity': True
            },
            'paths': {
                'project_root': str(PROJECT_ROOT),
                'models_dir': str(self.temp_dir / 'models'),
                'results_dir': str(self.temp_dir / 'results'),
                'visualizations_dir': str(self.temp_dir / 'visualizations'),
                'logs_dir': str(self.temp_dir / 'logs')
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Create test dataset
        np.random.seed(42)
        self.n_train_images = 5
        self.n_test_images = 3
        
        self.train_images, self.train_landmarks = self._create_test_dataset(self.n_train_images)
        self.test_images, self.test_landmarks = self._create_test_dataset(self.n_test_images)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_dataset(self, n_images):
        """Create synthetic test dataset."""
        images = []
        landmarks_list = []
        
        for i in range(n_images):
            # Create 100x100 test image
            img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            
            # Add some features for landmarks
            for j in range(15):
                x = 10 + (j % 5) * 18
                y = 10 + (j // 5) * 18
                if x < 96 and y < 96:
                    img[y:y+4, x:x+4] = 200 + np.random.randint(-50, 50)
            
            # Create corresponding landmarks
            landmarks = []
            for j in range(15):
                x = 10 + (j % 5) * 18 + np.random.normal(0, 1)
                y = 10 + (j // 5) * 18 + np.random.normal(0, 1)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            landmarks = np.clip(landmarks, 2, 98)  # Keep within bounds
            
            images.append(img)
            landmarks_list.append(landmarks)
        
        return images, landmarks_list
    
    def test_complete_training_to_evaluation_pipeline(self):
        """Test complete pipeline from training to evaluation."""
        # Step 1: Training
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.train_images, self.train_landmarks)
        
        # Verify training completed
        self.assertTrue(hasattr(predictor.predictor, 'eigenpatch_model'))
        
        # Step 2: Save trained model
        model_path = self.temp_dir / 'test_model.pkl'
        predictor.save(str(model_path))
        self.assertTrue(model_path.exists())
        
        # Step 3: Load model and make predictions
        new_predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        new_predictor.load(str(model_path))
        
        predictions = []
        for test_image in self.test_images:
            result = new_predictor.predict_landmarks(test_image)
            predictions.append(result.landmarks)
        
        # Verify predictions
        self.assertEqual(len(predictions), self.n_test_images)
        for pred in predictions:
            self.assertEqual(pred.shape, (15, 2))
        
        # Step 4: Evaluate predictions
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        evaluation_results = evaluator.evaluate_method(
            predictions=predictions,
            ground_truth=self.test_landmarks,
            method_name="Integration Test Method",
            image_names=[f"test_image_{i}.png" for i in range(self.n_test_images)]
        )
        
        # Verify evaluation results
        self.assertIn('basic_metrics', evaluation_results)
        self.assertIn('baseline_comparison', evaluation_results)
        
        basic_metrics = evaluation_results['basic_metrics']
        self.assertGreater(basic_metrics['mean_error'], 0)
        self.assertGreater(basic_metrics['std_error'], 0)
        
        # Step 5: Verify baseline comparison
        baseline_comp = evaluation_results['baseline_comparison']
        self.assertEqual(baseline_comp['baseline_error'], 5.63)
        self.assertIn('performance_category', baseline_comp)
    
    def test_configuration_persistence(self):
        """Test that configuration is properly preserved through pipeline."""
        # Train with specific configuration
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.train_images, self.train_landmarks)
        
        # Save model
        model_path = self.temp_dir / 'config_test_model.pkl'
        predictor.save(str(model_path))
        
        # Load and verify configuration
        new_predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        new_predictor.load(str(model_path))
        
        # Check that parameters are preserved
        self.assertEqual(new_predictor.patch_size, 11)
        self.assertEqual(new_predictor.n_components, 3)
        self.assertEqual(new_predictor.pyramid_levels, 2)
        self.assertEqual(new_predictor.lambda_shape, 0.1)
    
    def test_error_propagation(self):
        """Test error handling throughout the pipeline."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        
        # Test with invalid training data
        with self.assertRaises(Exception):
            predictor.train([], [])
        
        # Test prediction without training
        result = predictor.predict_landmarks(self.test_images[0])
        # Should return a valid result structure even if untrained
        self.assertEqual(result.landmarks.shape, (15, 2))
    
    def test_data_consistency(self):
        """Test data consistency through the pipeline."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.train_images, self.train_landmarks)
        
        # Make predictions
        test_image = self.test_images[0]
        
        # Multiple predictions on same image should be consistent
        result1 = predictor.predict_landmarks(test_image)
        result2 = predictor.predict_landmarks(test_image)
        
        # Should be identical or very close
        diff = np.mean(np.linalg.norm(result1.landmarks - result2.landmarks, axis=1))
        self.assertLess(diff, 1.0)  # Allow small numerical differences
    
    def test_scaling_behavior(self):
        """Test pipeline behavior with different data scales."""
        # Test with different image sizes by scaling coordinates
        large_config = self.test_config.copy()
        large_config['image_processing']['coordinate_scale_factor'] = 3.0
        
        predictor = ExperimentalLandmarkPredictor(config=large_config)
        
        # Scale landmarks for larger "images"
        scaled_landmarks = [lm * 3.0 for lm in self.train_landmarks]
        
        # Should train without issues
        predictor.train(self.train_images, scaled_landmarks)
        
        # Predictions should work
        result = predictor.predict_landmarks(self.test_images[0])
        self.assertEqual(result.landmarks.shape, (15, 2))
    
    def test_performance_regression(self):
        """Test that pipeline performance doesn't regress unexpectedly."""
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(self.train_images, self.train_landmarks)
        
        # Measure prediction time
        import time
        
        test_image = self.test_images[0]
        start_time = time.time()
        result = predictor.predict_landmarks(test_image)
        prediction_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 5 seconds for small test)
        self.assertLess(prediction_time, 5.0)
        
        # Should produce reasonable results
        self.assertLess(result.processing_time, 5.0)
        self.assertGreaterEqual(result.iterations, 0)
    
    def test_multiple_method_comparison(self):
        """Test comparison of multiple methods through the pipeline."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Create multiple sets of predictions with different error levels
        method_results = {}
        
        for i, (method_name, error_level) in enumerate([
            ("Method A", 2.0),
            ("Method B", 4.0),
            ("Method C", 6.0)
        ]):
            # Generate predictions with controlled error
            predictions = []
            for gt in self.test_landmarks:
                error = np.random.normal(0, error_level, gt.shape)
                pred = gt + error
                predictions.append(pred)
            
            results = evaluator.evaluate_method(
                predictions=predictions,
                ground_truth=self.test_landmarks,
                method_name=method_name
            )
            method_results[method_name] = results
        
        # Compare methods
        comparison = evaluator.compare_methods(method_results)
        
        # Verify comparison structure
        self.assertIn('ranking', comparison)
        self.assertIn('statistical_tests', comparison)
        
        # Check ranking order (Method A should be best, C worst)
        ranking = comparison['ranking']['by_mean_error']
        method_names = [r['method'] for r in ranking]
        
        # Method A should rank highest (lowest error)
        self.assertEqual(method_names[0], "Method A")
        # Method C should rank lowest (highest error)
        self.assertEqual(method_names[-1], "Method C")


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""
    
    def test_minimal_configuration(self):
        """Test pipeline with minimal configuration."""
        minimal_config = {
            'eigenpatches': {
                'patch_size': 11,
                'n_components': 2,
                'pyramid_levels': 1
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 1
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Should create predictor successfully
        predictor = ExperimentalLandmarkPredictor(config=minimal_config)
        self.assertEqual(predictor.patch_size, 11)
        self.assertEqual(predictor.n_components, 2)
        self.assertEqual(predictor.pyramid_levels, 1)
    
    def test_configuration_defaults(self):
        """Test that reasonable defaults are used when configuration is incomplete."""
        incomplete_config = {
            'eigenpatches': {
                'patch_size': 15
                # Missing other parameters
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Should still create predictor with defaults
        predictor = ExperimentalLandmarkPredictor(config=incomplete_config)
        self.assertEqual(predictor.patch_size, 15)
        # Should have reasonable defaults for missing parameters
        self.assertGreater(predictor.n_components, 0)
        self.assertGreater(predictor.pyramid_levels, 0)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration values."""
        invalid_config = {
            'eigenpatches': {
                'patch_size': -1,  # Invalid
                'n_components': 0,  # Invalid
                'pyramid_levels': 1
            },
            'landmark_predictor': {
                'lambda_shape': -0.1,  # Potentially invalid
                'max_iterations': 0     # Invalid
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Should either correct invalid values or raise appropriate errors
        # The exact behavior depends on the underlying implementation
        try:
            predictor = ExperimentalLandmarkPredictor(config=invalid_config)
            # If it succeeds, check that values were corrected
            self.assertGreater(predictor.patch_size, 0)
            self.assertGreater(predictor.n_components, 0)
        except (ValueError, AssertionError):
            # Expected behavior for invalid configuration
            pass


class TestMemoryAndPerformance(unittest.TestCase):
    """Test memory usage and performance characteristics."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.test_config = {
            'eigenpatches': {
                'patch_size': 11,
                'n_components': 5,
                'pyramid_levels': 2
            },
            'landmark_predictor': {
                'lambda_shape': 0.1,
                'max_iterations': 3
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
    
    def test_memory_usage_scaling(self):
        """Test memory usage with different dataset sizes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with small dataset
        small_images = [np.random.randint(0, 256, (64, 64), dtype=np.uint8) for _ in range(5)]
        small_landmarks = [np.random.uniform(5, 59, (15, 2)) for _ in range(5)]
        
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(small_images, small_landmarks)
        
        small_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = small_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for small test)
        self.assertLess(memory_increase, 100)
    
    def test_prediction_speed(self):
        """Test prediction speed requirements."""
        import time
        
        # Create test data
        images = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(3)]
        landmarks = [np.random.uniform(5, 95, (15, 2)) for _ in range(3)]
        
        predictor = ExperimentalLandmarkPredictor(config=self.test_config)
        predictor.train(images, landmarks)
        
        # Measure prediction time
        test_image = images[0]
        start_time = time.time()
        result = predictor.predict_landmarks(test_image)
        prediction_time = time.time() - start_time
        
        # Prediction should complete in reasonable time
        self.assertLess(prediction_time, 10.0)  # Less than 10 seconds
        
        # Result should be valid
        self.assertEqual(result.landmarks.shape, (15, 2))
        self.assertGreater(result.processing_time, 0)


if __name__ == '__main__':
    unittest.main()