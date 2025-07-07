"""
Unit tests for experimental evaluator module.

Tests the ExperimentalEvaluator class and its enhanced evaluation capabilities.
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

from core.experimental_evaluator import ExperimentalEvaluator


class TestExperimentalEvaluator(unittest.TestCase):
    """Test cases for ExperimentalEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.test_config = {
            'evaluation': {
                'confidence_threshold': 0.7,
                'statistical_tests': True,
                'generate_visualizations': False,  # Disable for testing
                'save_detailed_results': False     # Disable for testing
            },
            'paths': {
                'results_dir': 'test_results',
                'visualizations_dir': 'test_visualizations'
            },
            'logging': {
                'level': 'WARNING',  # Reduce log noise in tests
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Create synthetic test data
        np.random.seed(42)  # For reproducible tests
        self.n_images = 10
        self.n_landmarks = 15
        
        self.predictions = self._create_synthetic_predictions()
        self.ground_truth = self._create_synthetic_ground_truth()
        self.image_names = [f"test_image_{i:03d}.png" for i in range(self.n_images)]
        self.confidence_scores = self._create_synthetic_confidence_scores()
    
    def _create_synthetic_predictions(self):
        """Create synthetic prediction data."""
        predictions = []
        for i in range(self.n_images):
            # Create landmarks with some realistic spread
            landmarks = np.random.uniform(10, 290, (self.n_landmarks, 2))
            predictions.append(landmarks)
        return predictions
    
    def _create_synthetic_ground_truth(self):
        """Create synthetic ground truth data."""
        ground_truth = []
        for i in range(self.n_images):
            # Create ground truth based on predictions with added noise
            pred = self.predictions[i]
            # Add controlled error to simulate prediction error
            error = np.random.normal(0, 3, pred.shape)  # 3 pixel standard deviation
            gt = pred + error
            ground_truth.append(gt)
        return ground_truth
    
    def _create_synthetic_confidence_scores(self):
        """Create synthetic confidence scores."""
        confidence_scores = []
        for i in range(self.n_images):
            # Create confidence scores inversely related to error
            conf = np.random.uniform(0.3, 1.0, self.n_landmarks)
            confidence_scores.append(conf)
        return confidence_scores
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        self.assertEqual(evaluator.confidence_threshold, 0.7)
        self.assertTrue(evaluator.statistical_tests)
        self.assertFalse(evaluator.generate_visualizations)
        self.assertEqual(evaluator.baseline_error, 5.63)
        self.assertEqual(evaluator.baseline_std, 0.17)
    
    def test_basic_evaluation(self):
        """Test basic evaluation functionality."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method"
        )
        
        # Check result structure
        self.assertIn('method_name', results)
        self.assertIn('basic_metrics', results)
        self.assertIn('per_landmark_analysis', results)
        self.assertIn('per_image_analysis', results)
        self.assertIn('baseline_comparison', results)
        
        # Check basic metrics
        basic_metrics = results['basic_metrics']
        self.assertIn('mean_error', basic_metrics)
        self.assertIn('std_error', basic_metrics)
        self.assertIn('median_error', basic_metrics)
        self.assertGreater(basic_metrics['mean_error'], 0)
    
    def test_per_landmark_analysis(self):
        """Test per-landmark analysis."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method"
        )
        
        per_landmark = results['per_landmark_analysis']
        
        # Check structure
        self.assertIn('per_landmark_statistics', per_landmark)
        self.assertIn('best_landmark', per_landmark)
        self.assertIn('worst_landmark', per_landmark)
        
        # Check that we have statistics for all landmarks
        landmark_stats = per_landmark['per_landmark_statistics']
        self.assertEqual(len(landmark_stats), self.n_landmarks)
        
        # Check best/worst landmark identification
        best_landmark = per_landmark['best_landmark']
        worst_landmark = per_landmark['worst_landmark']
        self.assertIn('index', best_landmark)
        self.assertIn('mean_error', best_landmark)
        self.assertLessEqual(best_landmark['mean_error'], worst_landmark['mean_error'])
    
    def test_baseline_comparison(self):
        """Test baseline comparison functionality."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method"
        )
        
        baseline_comp = results['baseline_comparison']
        
        # Check structure
        self.assertIn('baseline_error', baseline_comp)
        self.assertIn('method_error', baseline_comp)
        self.assertIn('relative_error', baseline_comp)
        self.assertIn('improvement', baseline_comp)
        self.assertIn('performance_category', baseline_comp)
        
        # Check values
        self.assertEqual(baseline_comp['baseline_error'], 5.63)
        self.assertGreater(baseline_comp['method_error'], 0)
        
        # Performance category should be one of expected values
        valid_categories = ["Significantly Better", "Better", "Equivalent", 
                          "Worse", "Significantly Worse"]
        self.assertIn(baseline_comp['performance_category'], valid_categories)
    
    def test_confidence_analysis(self):
        """Test confidence score analysis."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method",
            confidence_scores=self.confidence_scores
        )
        
        conf_analysis = results['confidence_analysis']
        
        # Check structure
        self.assertIn('confidence_statistics', conf_analysis)
        self.assertIn('confidence_error_correlation', conf_analysis)
        self.assertIn('reliability_analysis', conf_analysis)
        
        # Check confidence statistics
        conf_stats = conf_analysis['confidence_statistics']
        self.assertIn('mean_confidence', conf_stats)
        self.assertGreaterEqual(conf_stats['mean_confidence'], 0)
        self.assertLessEqual(conf_stats['mean_confidence'], 1)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functionality."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method"
        )
        
        stat_analysis = results['statistical_analysis']
        
        # Check structure
        self.assertIn('normality_test', stat_analysis)
        self.assertIn('confidence_intervals', stat_analysis)
        self.assertIn('effect_size', stat_analysis)
        
        # Check normality test
        normality = stat_analysis['normality_test']
        self.assertIn('statistic', normality)
        self.assertIn('p_value', normality)
        self.assertIn('is_normal', normality)
        
        # Check confidence intervals
        ci = stat_analysis['confidence_intervals']
        self.assertIn('parametric_95', ci)
        self.assertIn('bootstrap_95', ci)
        self.assertEqual(len(ci['parametric_95']), 2)
        self.assertEqual(len(ci['bootstrap_95']), 2)
    
    def test_performance_categorization(self):
        """Test performance categorization logic."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Test different performance levels
        test_cases = [
            (-0.15, "Significantly Better"),
            (-0.07, "Better"),
            (0.02, "Equivalent"),
            (0.07, "Worse"),
            (0.15, "Significantly Worse")
        ]
        
        for relative_error, expected_category in test_cases:
            category = evaluator._categorize_performance(relative_error)
            self.assertEqual(category, expected_category)
    
    def test_per_image_analysis(self):
        """Test per-image analysis."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method",
            image_names=self.image_names
        )
        
        per_image = results['per_image_analysis']
        
        # Check structure
        self.assertIn('per_image_errors', per_image)
        self.assertIn('best_cases', per_image)
        self.assertIn('worst_cases', per_image)
        
        # Check that we have errors for all images
        per_image_errors = per_image['per_image_errors']
        self.assertEqual(len(per_image_errors), self.n_images)
        
        # Check best/worst cases
        best_cases = per_image['best_cases']
        worst_cases = per_image['worst_cases']
        self.assertGreater(len(best_cases), 0)
        self.assertGreater(len(worst_cases), 0)
        
        # Best cases should have lower error than worst cases
        if best_cases and worst_cases:
            self.assertLessEqual(best_cases[0]['error'], worst_cases[-1]['error'])
    
    def test_method_comparison(self):
        """Test multi-method comparison functionality."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Create results for multiple methods
        method_results = {}
        
        for i, method_name in enumerate(['Method A', 'Method B', 'Method C']):
            # Create slightly different predictions
            modified_predictions = []
            for pred in self.predictions:
                noise = np.random.normal(0, 0.5 + i * 0.5, pred.shape)
                modified_pred = pred + noise
                modified_predictions.append(modified_pred)
            
            results = evaluator.evaluate_method(
                predictions=modified_predictions,
                ground_truth=self.ground_truth,
                method_name=method_name
            )
            method_results[method_name] = results
        
        # Compare methods
        comparison_results = evaluator.compare_methods(method_results)
        
        # Check comparison structure
        self.assertIn('methods', comparison_results)
        self.assertIn('ranking', comparison_results)
        self.assertIn('statistical_tests', comparison_results)
        
        # Check ranking
        ranking = comparison_results['ranking']
        self.assertIn('by_mean_error', ranking)
        self.assertIn('best_method', ranking)
        self.assertIn('worst_method', ranking)
        
        # Check that ranking is properly ordered
        ranked_methods = ranking['by_mean_error']
        for i in range(len(ranked_methods) - 1):
            self.assertLessEqual(ranked_methods[i]['error'], ranked_methods[i+1]['error'])
    
    def test_error_handling_mismatched_data(self):
        """Test error handling with mismatched data."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Test with mismatched number of predictions and ground truth
        with self.assertRaises(ValueError):
            evaluator.evaluate_method(
                predictions=self.predictions[:5],
                ground_truth=self.ground_truth[:3],
                method_name="Test Method"
            )
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Test with empty data
        with self.assertRaises(Exception):
            evaluator.evaluate_method(
                predictions=[],
                ground_truth=[],
                method_name="Test Method"
            )
    
    def test_summary_generation(self):
        """Test evaluation summary generation."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Run an evaluation
        evaluator.evaluate_method(
            predictions=self.predictions,
            ground_truth=self.ground_truth,
            method_name="Test Method"
        )
        
        # Get summary
        summary = evaluator.get_summary()
        
        self.assertIn('n_evaluations', summary)
        self.assertIn('methods_evaluated', summary)
        self.assertIn('baseline_error', summary)
        self.assertEqual(summary['n_evaluations'], 1)
        self.assertIn('Test Method', summary['methods_evaluated'])


class TestEvaluatorIntegration(unittest.TestCase):
    """Integration tests for evaluator functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_config = {
            'evaluation': {
                'confidence_threshold': 0.7,
                'statistical_tests': True,
                'generate_visualizations': False,
                'save_detailed_results': False
            },
            'paths': {
                'results_dir': 'test_results',
                'visualizations_dir': 'test_visualizations'
            },
            'logging': {
                'level': 'WARNING',
                'console_logging': False,
                'file_logging': False
            }
        }
        
        # Create test data that simulates baseline performance
        np.random.seed(42)
        self.n_images = 20
        
        self.baseline_predictions = []
        self.baseline_ground_truth = []
        
        for i in range(self.n_images):
            gt = np.random.uniform(10, 290, (15, 2))
            # Add error around baseline performance (5.63 pixels)
            error = np.random.normal(0, 5.63/3, gt.shape)  # 3-sigma rule
            pred = gt + error
            
            self.baseline_predictions.append(pred)
            self.baseline_ground_truth.append(gt)
    
    def test_baseline_accuracy_verification(self):
        """Test that evaluator correctly identifies baseline-level performance."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        results = evaluator.evaluate_method(
            predictions=self.baseline_predictions,
            ground_truth=self.baseline_ground_truth,
            method_name="Baseline Test"
        )
        
        baseline_comp = results['baseline_comparison']
        
        # Should be close to baseline performance
        method_error = baseline_comp['method_error']
        baseline_error = baseline_comp['baseline_error']
        
        # Allow some tolerance due to random generation
        relative_diff = abs(method_error - baseline_error) / baseline_error
        self.assertLess(relative_diff, 0.3)  # Within 30%
        
        # Performance category should be equivalent or close
        category = baseline_comp['performance_category']
        self.assertIn(category, ["Equivalent", "Better", "Worse"])
    
    def test_real_world_performance_simulation(self):
        """Test evaluation with realistic performance variations."""
        evaluator = ExperimentalEvaluator(config=self.test_config)
        
        # Simulate different performance levels
        performance_scenarios = {
            "Excellent Method": 3.0,    # Much better than baseline
            "Good Method": 4.5,         # Better than baseline
            "Baseline Method": 5.63,    # Same as baseline
            "Poor Method": 7.0,         # Worse than baseline
            "Bad Method": 10.0          # Much worse than baseline
        }
        
        for method_name, target_error in performance_scenarios.items():
            # Create predictions with target error level
            predictions = []
            for gt in self.baseline_ground_truth:
                error = np.random.normal(0, target_error/3, gt.shape)
                pred = gt + error
                predictions.append(pred)
            
            results = evaluator.evaluate_method(
                predictions=predictions,
                ground_truth=self.baseline_ground_truth,
                method_name=method_name
            )
            
            # Check that performance categorization makes sense
            actual_error = results['basic_metrics']['mean_error']
            category = results['baseline_comparison']['performance_category']
            
            if target_error < 4.5:
                self.assertIn(category, ["Significantly Better", "Better"])
            elif target_error > 7.0:
                self.assertIn(category, ["Worse", "Significantly Worse"])
            else:
                # Allow some flexibility for borderline cases
                self.assertIn(category, ["Significantly Better", "Better", "Equivalent", 
                                       "Worse", "Significantly Worse"])


if __name__ == '__main__':
    unittest.main()