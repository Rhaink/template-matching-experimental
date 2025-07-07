"""
Experimental adapter for evaluation utilities with enhanced reporting and visualization.

This module provides an experimental wrapper around the original evaluation utilities,
adding HTML report generation, interactive visualizations, advanced statistical analysis,
and automatic baseline comparison while maintaining full API compatibility.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, some visualizations will be disabled")

from scipy import stats

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available, interactive visualizations will be disabled")

# Add template_matching to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_MATCHING_PATH = PROJECT_ROOT / "template_matching" / "src"
sys.path.insert(0, str(TEMPLATE_MATCHING_PATH))

try:
    from utils.evaluation import LandmarkEvaluator, MethodComparator
except ImportError as e:
    raise ImportError(f"Could not import original evaluation utilities: {e}")


class ExperimentalEvaluator:
    """
    Experimental adapter for evaluation utilities with enhanced reporting.
    
    This class wraps the original LandmarkEvaluator and MethodComparator classes,
    adding HTML report generation, interactive visualizations, advanced statistical
    analysis, and automatic baseline comparison while maintaining full compatibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 config_file: Optional[str] = None):
        """
        Initialize experimental evaluator.
        
        Args:
            config: Configuration dictionary
            config_file: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config, config_file)
        self._setup_logging()
        
        # Initialize original evaluators
        self.landmark_evaluator = LandmarkEvaluator()
        self.method_comparator = MethodComparator()
        
        # Extract evaluation configuration
        eval_config = self.config.get('evaluation', {})
        self.confidence_threshold = eval_config.get('confidence_threshold', 0.7)
        self.statistical_tests = eval_config.get('statistical_tests', True)
        self.generate_visualizations = eval_config.get('generate_visualizations', True)
        self.save_detailed_results = eval_config.get('save_detailed_results', True)
        
        # Baseline comparison
        self.baseline_error = 5.63  # pixels
        self.baseline_std = 0.17    # pixels
        self.baseline_dataset = "coordenadas_prueba_1.csv"
        self.baseline_n_images = 159
        
        # Results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        self.statistical_results = {}
        
        # Paths
        paths_config = self.config.get('paths', {})
        self.results_dir = Path(paths_config.get('results_dir', 'results'))
        self.visualizations_dir = Path(paths_config.get('visualizations_dir', 'visualizations'))
        
        self.logger.info("Initialized ExperimentalEvaluator with enhanced reporting capabilities")
    
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
            'evaluation': {
                'confidence_threshold': 0.7,
                'statistical_tests': True,
                'generate_visualizations': True,
                'save_detailed_results': True
            },
            'paths': {
                'results_dir': 'results',
                'visualizations_dir': 'visualizations'
            },
            'logging': {'level': 'INFO'}
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
            file_handler = logging.FileHandler(log_dir / 'evaluator.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def evaluate_method(self, predictions: List[np.ndarray], 
                       ground_truth: List[np.ndarray],
                       method_name: str = "Experimental Method",
                       confidence_scores: Optional[List[np.ndarray]] = None,
                       image_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a landmark detection method.
        
        Args:
            predictions: List of predicted landmark arrays
            ground_truth: List of ground truth landmark arrays
            method_name: Name of the method being evaluated
            confidence_scores: Optional confidence scores for predictions
            image_names: Optional names of images for detailed analysis
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        self.logger.info(f"Evaluating method: {method_name}")
        self.logger.info(f"Number of predictions: {len(predictions)}")
        self.logger.info(f"Number of ground truth: {len(ground_truth)}")
        
        start_time = time.time()
        
        # Basic validation
        if len(predictions) != len(ground_truth):
            raise ValueError("Number of predictions must match number of ground truth")
        
        # Compute basic metrics using original evaluator
        basic_results = self._compute_basic_metrics(predictions, ground_truth)
        
        # Compute per-landmark statistics
        per_landmark_results = self._compute_per_landmark_statistics(predictions, ground_truth)
        
        # Compute per-image statistics
        per_image_results = self._compute_per_image_statistics(predictions, ground_truth, image_names)
        
        # Statistical analysis
        statistical_analysis = {}
        if self.statistical_tests:
            statistical_analysis = self._compute_statistical_analysis(predictions, ground_truth)
        
        # Baseline comparison
        baseline_comparison = self._compare_with_baseline(basic_results)
        
        # Confidence analysis
        confidence_analysis = {}
        if confidence_scores is not None:
            confidence_analysis = self._analyze_confidence_scores(
                predictions, ground_truth, confidence_scores
            )
        
        # Compile results
        evaluation_results = {
            'method_name': method_name,
            'evaluation_date': datetime.now().isoformat(),
            'processing_time': time.time() - start_time,
            'dataset_info': {
                'n_images': len(predictions),
                'n_landmarks': len(predictions[0]) if predictions else 0,
                'baseline_dataset': self.baseline_dataset,
                'baseline_n_images': self.baseline_n_images
            },
            'basic_metrics': basic_results,
            'per_landmark_analysis': per_landmark_results,
            'per_image_analysis': per_image_results,
            'statistical_analysis': statistical_analysis,
            'baseline_comparison': baseline_comparison,
            'confidence_analysis': confidence_analysis
        }
        
        # Store results
        self.evaluation_results[method_name] = evaluation_results
        
        # Generate visualizations
        if self.generate_visualizations:
            self._generate_evaluation_visualizations(evaluation_results)
        
        # Save detailed results
        if self.save_detailed_results:
            self._save_detailed_results(evaluation_results)
        
        self.logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        return evaluation_results
    
    def _compute_basic_metrics(self, predictions: List[np.ndarray], 
                             ground_truth: List[np.ndarray]) -> Dict[str, Any]:
        """Compute basic evaluation metrics."""
        errors = []
        
        for pred, gt in zip(predictions, ground_truth):
            error = self.landmark_evaluator.compute_point_to_point_error(pred, gt)
            errors.append(error)
        
        errors = np.array(errors)
        
        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'q25_error': float(np.percentile(errors, 25)),
            'q75_error': float(np.percentile(errors, 75)),
            'errors': errors.tolist()
        }
    
    def _compute_per_landmark_statistics(self, predictions: List[np.ndarray], 
                                       ground_truth: List[np.ndarray]) -> Dict[str, Any]:
        """Compute per-landmark statistics."""
        if not predictions:
            return {}
        
        n_landmarks = len(predictions[0])
        per_landmark_errors = [[] for _ in range(n_landmarks)]
        
        # Collect errors per landmark
        for pred, gt in zip(predictions, ground_truth):
            landmark_errors = np.linalg.norm(pred - gt, axis=1)
            for i, error in enumerate(landmark_errors):
                per_landmark_errors[i].append(error)
        
        # Compute statistics per landmark
        per_landmark_stats = {}
        for i in range(n_landmarks):
            errors = np.array(per_landmark_errors[i])
            per_landmark_stats[f'landmark_{i}'] = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'median_error': float(np.median(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors)),
                'errors': errors.tolist()
            }
        
        # Identify best and worst performing landmarks
        mean_errors = [stats['mean_error'] for stats in per_landmark_stats.values()]
        best_landmark_idx = np.argmin(mean_errors)
        worst_landmark_idx = np.argmax(mean_errors)
        
        return {
            'per_landmark_statistics': per_landmark_stats,
            'best_landmark': {
                'index': best_landmark_idx,
                'mean_error': mean_errors[best_landmark_idx]
            },
            'worst_landmark': {
                'index': worst_landmark_idx,
                'mean_error': mean_errors[worst_landmark_idx]
            },
            'landmark_error_range': float(np.max(mean_errors) - np.min(mean_errors))
        }
    
    def _compute_per_image_statistics(self, predictions: List[np.ndarray], 
                                    ground_truth: List[np.ndarray],
                                    image_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute per-image statistics."""
        image_errors = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            error = self.landmark_evaluator.compute_point_to_point_error(pred, gt)
            image_name = image_names[i] if image_names else f"image_{i}"
            image_errors.append({
                'image_name': image_name,
                'error': error,
                'index': i
            })
        
        # Sort by error
        image_errors.sort(key=lambda x: x['error'])
        
        # Identify best and worst cases
        best_cases = image_errors[:5]
        worst_cases = image_errors[-5:]
        
        return {
            'per_image_errors': image_errors,
            'best_cases': best_cases,
            'worst_cases': worst_cases,
            'error_distribution': {
                'mean': float(np.mean([e['error'] for e in image_errors])),
                'std': float(np.std([e['error'] for e in image_errors]))
            }
        }
    
    def _compute_statistical_analysis(self, predictions: List[np.ndarray], 
                                    ground_truth: List[np.ndarray]) -> Dict[str, Any]:
        """Compute advanced statistical analysis."""
        errors = []
        for pred, gt in zip(predictions, ground_truth):
            error = self.landmark_evaluator.compute_point_to_point_error(pred, gt)
            errors.append(error)
        
        errors = np.array(errors)
        
        # Normality test
        normality_stat, normality_p = stats.shapiro(errors)
        
        # Confidence intervals
        confidence_95 = stats.t.interval(0.95, len(errors)-1, 
                                        loc=np.mean(errors), 
                                        scale=stats.sem(errors))
        
        # Bootstrap confidence intervals
        bootstrap_means = []
        for _ in range(1000):
            bootstrap_sample = np.random.choice(errors, size=len(errors), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])
        
        return {
            'normality_test': {
                'statistic': float(normality_stat),
                'p_value': float(normality_p),
                'is_normal': normality_p > 0.05
            },
            'confidence_intervals': {
                'parametric_95': [float(confidence_95[0]), float(confidence_95[1])],
                'bootstrap_95': [float(bootstrap_ci[0]), float(bootstrap_ci[1])]
            },
            'effect_size': {
                'cohens_d': float((np.mean(errors) - self.baseline_error) / 
                                 np.sqrt((np.std(errors)**2 + self.baseline_std**2) / 2))
            }
        }
    
    def _compare_with_baseline(self, basic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results with baseline performance."""
        method_error = basic_results['mean_error']
        method_std = basic_results['std_error']
        
        # Relative performance
        relative_error = (method_error - self.baseline_error) / self.baseline_error
        
        # Statistical significance test (assuming normal distribution)
        z_score = (method_error - self.baseline_error) / np.sqrt(
            (method_std**2 / len(basic_results['errors'])) + 
            (self.baseline_std**2 / self.baseline_n_images)
        )
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'baseline_error': self.baseline_error,
            'baseline_std': self.baseline_std,
            'method_error': method_error,
            'method_std': method_std,
            'relative_error': float(relative_error),
            'improvement': float(-relative_error),  # Negative relative error is improvement
            'statistical_test': {
                'z_score': float(z_score),
                'p_value': float(p_value),
                'is_significantly_different': p_value < 0.05
            },
            'performance_category': self._categorize_performance(relative_error)
        }
    
    def _categorize_performance(self, relative_error: float) -> str:
        """Categorize performance relative to baseline."""
        if relative_error < -0.1:
            return "Significantly Better"
        elif relative_error < -0.05:
            return "Better"
        elif relative_error < 0.05:
            return "Equivalent"
        elif relative_error < 0.1:
            return "Worse"
        else:
            return "Significantly Worse"
    
    def _analyze_confidence_scores(self, predictions: List[np.ndarray], 
                                 ground_truth: List[np.ndarray],
                                 confidence_scores: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze confidence scores vs. actual performance."""
        errors = []
        confidences = []
        
        for pred, gt, conf in zip(predictions, ground_truth, confidence_scores):
            error = self.landmark_evaluator.compute_point_to_point_error(pred, gt)
            mean_confidence = np.mean(conf)
            errors.append(error)
            confidences.append(mean_confidence)
        
        # Correlation analysis
        correlation, correlation_p = stats.pearsonr(confidences, errors)
        
        # Reliability analysis (lower error should correlate with higher confidence)
        reliability_correlation, reliability_p = stats.pearsonr(confidences, [-e for e in errors])
        
        return {
            'confidence_statistics': {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences))
            },
            'confidence_error_correlation': {
                'correlation': float(correlation),
                'p_value': float(correlation_p),
                'interpretation': "Lower confidence correlates with higher error" if correlation > 0 else "Higher confidence correlates with higher error"
            },
            'reliability_analysis': {
                'correlation': float(reliability_correlation),
                'p_value': float(reliability_p),
                'is_reliable': reliability_correlation > 0.3 and reliability_p < 0.05
            }
        }
    
    def _generate_evaluation_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive evaluation visualizations."""
        self.logger.info("Generating evaluation visualizations")
        
        # Create visualizations directory
        vis_dir = self.visualizations_dir / "evaluation"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Error distribution plot
        self._plot_error_distribution(results, vis_dir)
        
        # Per-landmark analysis
        self._plot_per_landmark_analysis(results, vis_dir)
        
        # Baseline comparison
        self._plot_baseline_comparison(results, vis_dir)
        
        # CED curve
        self._plot_ced_curve(results, vis_dir)
        
        # Generate interactive HTML report
        self._generate_html_report(results, vis_dir)
    
    def _plot_error_distribution(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot error distribution."""
        errors = results['basic_metrics']['errors']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
        axes[0, 0].axvline(self.baseline_error, color='green', linestyle='--', label=f'Baseline: {self.baseline_error:.2f}')
        axes[0, 0].set_xlabel('Error (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(errors, labels=['Method'])
        axes[0, 1].set_ylabel('Error (pixels)')
        axes[0, 1].set_title('Error Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(errors, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error vs. Image Index
        axes[1, 1].plot(errors, marker='o', markersize=3, alpha=0.7)
        axes[1, 1].axhline(np.mean(errors), color='red', linestyle='--', label='Mean')
        axes[1, 1].axhline(self.baseline_error, color='green', linestyle='--', label='Baseline')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Error (pixels)')
        axes[1, 1].set_title('Error vs. Image Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_landmark_analysis(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot per-landmark analysis."""
        per_landmark_stats = results['per_landmark_analysis']['per_landmark_statistics']
        
        landmark_indices = list(range(len(per_landmark_stats)))
        mean_errors = [stats['mean_error'] for stats in per_landmark_stats.values()]
        std_errors = [stats['std_error'] for stats in per_landmark_stats.values()]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Mean error per landmark
        bars = axes[0].bar(landmark_indices, mean_errors, yerr=std_errors, 
                          capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axhline(self.baseline_error, color='red', linestyle='--', 
                       label=f'Baseline: {self.baseline_error:.2f}')
        axes[0].set_xlabel('Landmark Index')
        axes[0].set_ylabel('Mean Error (pixels)')
        axes[0].set_title('Mean Error per Landmark')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Highlight best and worst landmarks
        best_idx = results['per_landmark_analysis']['best_landmark']['index']
        worst_idx = results['per_landmark_analysis']['worst_landmark']['index']
        bars[best_idx].set_color('green')
        bars[worst_idx].set_color('red')
        
        # Error distribution per landmark (violin plot)
        all_errors = [stats['errors'] for stats in per_landmark_stats.values()]
        axes[1].violinplot(all_errors, positions=landmark_indices, showmeans=True)
        axes[1].set_xlabel('Landmark Index')
        axes[1].set_ylabel('Error (pixels)')
        axes[1].set_title('Error Distribution per Landmark')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_landmark_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_baseline_comparison(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot baseline comparison."""
        baseline_comp = results['baseline_comparison']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error comparison
        methods = ['Baseline', 'Method']
        errors = [baseline_comp['baseline_error'], baseline_comp['method_error']]
        stds = [baseline_comp['baseline_std'], baseline_comp['method_std']]
        
        bars = axes[0].bar(methods, errors, yerr=stds, capsize=5, 
                          color=['green', 'skyblue'], alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Mean Error (pixels)')
        axes[0].set_title('Method vs. Baseline Comparison')
        axes[0].grid(True, alpha=0.3)
        
        # Add improvement percentage
        improvement = baseline_comp['improvement'] * 100
        axes[0].text(0.5, max(errors) * 0.8, f'Improvement: {improvement:.1f}%', 
                    ha='center', fontsize=12, fontweight='bold')
        
        # Performance category
        category = baseline_comp['performance_category']
        color_map = {
            'Significantly Better': 'green',
            'Better': 'lightgreen',
            'Equivalent': 'yellow',
            'Worse': 'orange',
            'Significantly Worse': 'red'
        }
        
        axes[1].bar([category], [1], color=color_map.get(category, 'gray'), alpha=0.7)
        axes[1].set_ylabel('Performance Category')
        axes[1].set_title('Performance Category vs. Baseline')
        axes[1].set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ced_curve(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot Cumulative Error Distribution curve."""
        errors = results['basic_metrics']['errors']
        
        # Compute CED
        max_error = max(errors)
        thresholds = np.linspace(0, max_error, 100)
        ced_values = []
        
        for threshold in thresholds:
            proportion = np.mean(np.array(errors) <= threshold)
            ced_values.append(proportion)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, ced_values, linewidth=2, label='Method')
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        plt.axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
        plt.xlabel('Error Threshold (pixels)')
        plt.ylabel('Proportion of Images')
        plt.title('Cumulative Error Distribution (CED)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add key statistics
        median_error = np.median(errors)
        error_95 = np.percentile(errors, 95)
        plt.axvline(median_error, color='orange', linestyle=':', alpha=0.7, label=f'Median: {median_error:.2f}')
        plt.axvline(error_95, color='purple', linestyle=':', alpha=0.7, label=f'95th percentile: {error_95:.2f}')
        plt.legend()
        
        plt.savefig(output_dir / 'ced_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate interactive HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report - {results['method_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .improvement {{ color: green; font-weight: bold; }}
                .degradation {{ color: red; font-weight: bold; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Evaluation Report: {results['method_name']}</h1>
                <p>Generated on: {results['evaluation_date']}</p>
                <p>Processing time: {results['processing_time']:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Dataset Information</h2>
                <p>Number of images: {results['dataset_info']['n_images']}</p>
                <p>Number of landmarks: {results['dataset_info']['n_landmarks']}</p>
                <p>Baseline dataset: {results['dataset_info']['baseline_dataset']}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <div class="metric">Mean Error: {results['basic_metrics']['mean_error']:.3f} ± {results['basic_metrics']['std_error']:.3f} pixels</div>
                <div class="metric">Median Error: {results['basic_metrics']['median_error']:.3f} pixels</div>
                <div class="metric">Error Range: {results['basic_metrics']['min_error']:.3f} - {results['basic_metrics']['max_error']:.3f} pixels</div>
            </div>
            
            <div class="section">
                <h2>Baseline Comparison</h2>
                <div class="metric">Baseline Error: {results['baseline_comparison']['baseline_error']:.3f} pixels</div>
                <div class="metric">Method Error: {results['baseline_comparison']['method_error']:.3f} pixels</div>
                <div class="metric {'improvement' if results['baseline_comparison']['improvement'] > 0 else 'degradation'}">
                    Performance Change: {results['baseline_comparison']['improvement']*100:.1f}%
                </div>
                <div class="metric">Category: {results['baseline_comparison']['performance_category']}</div>
            </div>
            
            <div class="section">
                <h2>Per-Landmark Analysis</h2>
                <p>Best performing landmark: {results['per_landmark_analysis']['best_landmark']['index']} 
                   (error: {results['per_landmark_analysis']['best_landmark']['mean_error']:.3f} pixels)</p>
                <p>Worst performing landmark: {results['per_landmark_analysis']['worst_landmark']['index']} 
                   (error: {results['per_landmark_analysis']['worst_landmark']['mean_error']:.3f} pixels)</p>
                <p>Error range across landmarks: {results['per_landmark_analysis']['landmark_error_range']:.3f} pixels</p>
            </div>
            
            <div class="section">
                <h2>Best and Worst Cases</h2>
                <h3>Best Cases (Lowest Error)</h3>
                <table class="table">
                    <tr><th>Rank</th><th>Image</th><th>Error (pixels)</th></tr>
        """
        
        for i, case in enumerate(results['per_image_analysis']['best_cases']):
            html_content += f"<tr><td>{i+1}</td><td>{case['image_name']}</td><td>{case['error']:.3f}</td></tr>"
        
        html_content += """
                </table>
                <h3>Worst Cases (Highest Error)</h3>
                <table class="table">
                    <tr><th>Rank</th><th>Image</th><th>Error (pixels)</th></tr>
        """
        
        for i, case in enumerate(results['per_image_analysis']['worst_cases']):
            html_content += f"<tr><td>{i+1}</td><td>{case['image_name']}</td><td>{case['error']:.3f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
        """
        
        if results['statistical_analysis']:
            stat_analysis = results['statistical_analysis']
            html_content += f"""
                <p>Normality test p-value: {stat_analysis['normality_test']['p_value']:.4f}</p>
                <p>Data is {'normally' if stat_analysis['normality_test']['is_normal'] else 'not normally'} distributed</p>
                <p>95% Confidence Interval: [{stat_analysis['confidence_intervals']['parametric_95'][0]:.3f}, {stat_analysis['confidence_intervals']['parametric_95'][1]:.3f}]</p>
                <p>Effect size (Cohen's d): {stat_analysis['effect_size']['cohens_d']:.3f}</p>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>The following visualizations have been generated:</p>
                <ul>
                    <li><a href="error_distribution.png">Error Distribution</a></li>
                    <li><a href="per_landmark_analysis.png">Per-Landmark Analysis</a></li>
                    <li><a href="baseline_comparison.png">Baseline Comparison</a></li>
                    <li><a href="ced_curve.png">CED Curve</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / 'evaluation_report.html', 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_dir / 'evaluation_report.html'}")
    
    def _save_detailed_results(self, results: Dict[str, Any]) -> None:
        """Save detailed results to files."""
        # Create results directory
        results_dir = self.results_dir / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = results_dir / f"{results['method_name']}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as YAML
        yaml_path = results_dir / f"{results['method_name']}_results.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Save per-image results as CSV
        per_image_data = results['per_image_analysis']['per_image_errors']
        df = pd.DataFrame(per_image_data)
        csv_path = results_dir / f"{results['method_name']}_per_image_errors.csv"
        df.to_csv(csv_path, index=False)
        
        # Save per-landmark results as CSV
        per_landmark_stats = results['per_landmark_analysis']['per_landmark_statistics']
        landmark_data = []
        for landmark, stats in per_landmark_stats.items():
            landmark_data.append({
                'landmark': landmark,
                'mean_error': stats['mean_error'],
                'std_error': stats['std_error'],
                'median_error': stats['median_error'],
                'min_error': stats['min_error'],
                'max_error': stats['max_error']
            })
        
        df_landmarks = pd.DataFrame(landmark_data)
        csv_landmarks_path = results_dir / f"{results['method_name']}_per_landmark_stats.csv"
        df_landmarks.to_csv(csv_landmarks_path, index=False)
        
        self.logger.info(f"Detailed results saved to: {results_dir}")
    
    def compare_methods(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple methods against each other and baseline.
        
        Args:
            method_results: Dictionary of method names to evaluation results
            
        Returns:
            Dictionary containing method comparison results
        """
        self.logger.info(f"Comparing {len(method_results)} methods")
        
        comparison_results = {
            'comparison_date': datetime.now().isoformat(),
            'methods': list(method_results.keys()),
            'baseline_error': self.baseline_error,
            'method_comparison': {},
            'ranking': {},
            'statistical_tests': {}
        }
        
        # Extract errors for each method
        method_errors = {}
        for method_name, results in method_results.items():
            method_errors[method_name] = results['basic_metrics']['errors']
        
        # Pairwise comparisons
        method_names = list(method_errors.keys())
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(method_errors[method1], method_errors[method2])
                
                comparison_key = f"{method1}_vs_{method2}"
                comparison_results['statistical_tests'][comparison_key] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_significantly_different': p_value < 0.05
                }
        
        # Method ranking
        mean_errors = {name: np.mean(errors) for name, errors in method_errors.items()}
        sorted_methods = sorted(mean_errors.items(), key=lambda x: x[1])
        
        comparison_results['ranking'] = {
            'by_mean_error': [{'method': method, 'error': error} for method, error in sorted_methods],
            'best_method': sorted_methods[0][0],
            'worst_method': sorted_methods[-1][0]
        }
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> None:
        """Generate HTML comparison report for multiple methods."""
        # Create comparison visualizations directory
        vis_dir = self.visualizations_dir / "comparison"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Method Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Method Comparison Report</h1>
                <p>Generated on: {comparison_results['comparison_date']}</p>
                <p>Baseline error: {comparison_results['baseline_error']:.3f} pixels</p>
            </div>
            
            <div class="section">
                <h2>Method Ranking</h2>
                <table class="table">
                    <tr><th>Rank</th><th>Method</th><th>Mean Error (pixels)</th><th>vs Baseline</th></tr>
        """
        
        for i, method_info in enumerate(comparison_results['ranking']['by_mean_error']):
            method_name = method_info['method']
            error = method_info['error']
            vs_baseline = ((error - comparison_results['baseline_error']) / comparison_results['baseline_error']) * 100
            row_class = 'best' if i == 0 else 'worst' if i == len(comparison_results['ranking']['by_mean_error']) - 1 else ''
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{i+1}</td>
                    <td>{method_name}</td>
                    <td>{error:.3f}</td>
                    <td>{vs_baseline:+.1f}%</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(vis_dir / 'comparison_report.html', 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Comparison report generated: {vis_dir / 'comparison_report.html'}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        return {
            'n_evaluations': len(self.evaluation_results),
            'methods_evaluated': list(self.evaluation_results.keys()),
            'baseline_error': self.baseline_error,
            'baseline_std': self.baseline_std,
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'statistical_tests': self.statistical_tests,
                'generate_visualizations': self.generate_visualizations
            }
        }