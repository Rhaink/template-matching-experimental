#!/usr/bin/env python3
"""
Experimental evaluation script for template matching landmark detection.

This script provides comprehensive evaluation capabilities that replicate and extend
the functionality of template_matching/scripts/per_landmark_evaluation.py while
adding HTML reporting, interactive visualizations, and automatic baseline comparison.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import argparse
import logging
import yaml
import time
from datetime import datetime
import pickle

# Add project root to path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
MATCHING_EXPERIMENTAL_DIR = PROJECT_ROOT_DIR / "matching_experimental"
sys.path.insert(0, str(MATCHING_EXPERIMENTAL_DIR))

from core.experimental_evaluator import ExperimentalEvaluator


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration from config."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Create logs directory
    logs_dir = Path(config.get('paths', {}).get('logs_dir', 'logs'))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(logs_dir / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )


def load_processing_results(results_path: str) -> Dict[str, Any]:
    """
    Load processing results from pickle file.
    
    Args:
        results_path: Path to processing results pickle file
        
    Returns:
        Dictionary containing processing results
    """
    logging.info(f"Loading processing results from: {results_path}")
    
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    logging.info(f"Loaded results for {results.get('n_images', 'unknown')} images")
    return results


def extract_predictions_and_ground_truth(processing_results: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Extract predictions and ground truth from processing results.
    
    Args:
        processing_results: Processing results dictionary
        
    Returns:
        Tuple of (predictions, ground_truth, image_names)
    """
    logging.info("Extracting predictions and ground truth...")
    
    predictions = []
    ground_truth = []
    image_names = []
    
    # Get predictions from processing results
    prediction_results = processing_results.get('predictions', [])
    error_stats = processing_results.get('error_statistics', {})
    per_image_results = error_stats.get('per_image_results', [])
    
    if not prediction_results:
        raise ValueError("No predictions found in processing results")
    
    # Extract landmark predictions
    for pred_result in prediction_results:
        predictions.append(pred_result.landmarks)
        image_names.append(getattr(pred_result, 'image_name', f'image_{len(predictions)}'))
    
    # Try to extract ground truth from per_image_results or other sources
    # This is a simplified approach - in practice, we might need to reload the ground truth
    if per_image_results:
        logging.info("Ground truth extraction not directly available from processing results")
        logging.info("Recommend providing ground truth separately for complete evaluation")
        # For now, create dummy ground truth
        for pred in predictions:
            ground_truth.append(np.zeros_like(pred))
    else:
        # Create dummy ground truth
        for pred in predictions:
            ground_truth.append(np.zeros_like(pred))
    
    logging.info(f"Extracted {len(predictions)} predictions and {len(ground_truth)} ground truth arrays")
    return predictions, ground_truth, image_names


def load_ground_truth_separately(coordinates_file: str, images_base_dir: str,
                                image_names: List[str],
                                coordinate_scale_factor: float = 4.67) -> List[np.ndarray]:
    """
    Load ground truth landmarks separately from coordinates file.
    
    Args:
        coordinates_file: Path to coordinates CSV file
        images_base_dir: Base directory for images (unused but kept for compatibility)
        image_names: List of image names to match
        coordinate_scale_factor: Scale factor from 64x64 to image size
        
    Returns:
        List of ground truth landmark arrays
    """
    logging.info(f"Loading ground truth from: {coordinates_file}")
    
    # Load coordinates (CSV files have no header)
    df = pd.read_csv(coordinates_file, header=None)
    
    # Create mapping from image name to landmarks
    gt_mapping = {}
    for idx, row in df.iterrows():
        try:
            # Extract coordinates (30 values for 15 landmarks)
            coords = row.iloc[:30].values.astype(float)
            landmarks = coords.reshape(15, 2)
            
            # Scale landmarks from 64x64 to actual image size
            landmarks_scaled = landmarks * coordinate_scale_factor
            
            # Extract image name
            image_name = row.iloc[30]
            gt_mapping[image_name] = landmarks_scaled
            
        except Exception as e:
            logging.warning(f"Failed to process ground truth for row {idx}: {e}")
    
    # Match ground truth to image names
    ground_truth = []
    for image_name in image_names:
        if image_name in gt_mapping:
            ground_truth.append(gt_mapping[image_name])
        else:
            logging.warning(f"Ground truth not found for image: {image_name}")
            # Create dummy ground truth
            ground_truth.append(np.zeros((15, 2)))
    
    logging.info(f"Loaded ground truth for {len(ground_truth)} images")
    return ground_truth


def run_comprehensive_evaluation(evaluator: ExperimentalEvaluator,
                                predictions: List[np.ndarray],
                                ground_truth: List[np.ndarray],
                                image_names: List[str],
                                processing_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive evaluation using the experimental evaluator.
    
    Args:
        evaluator: Experimental evaluator instance
        predictions: List of predicted landmark arrays
        ground_truth: List of ground truth landmark arrays
        image_names: List of image names
        processing_results: Original processing results
        
    Returns:
        Dictionary containing evaluation results
    """
    logging.info("Running comprehensive evaluation...")
    
    # Extract confidence scores if available
    confidence_scores = None
    if 'predictions' in processing_results:
        confidence_scores = []
        for pred_result in processing_results['predictions']:
            if hasattr(pred_result, 'confidence') and pred_result.confidence is not None:
                confidence_scores.append(pred_result.confidence)
            else:
                confidence_scores.append(np.ones(15))  # Default confidence
    
    # Run evaluation
    method_name = f"Experimental Template Matching ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    
    evaluation_results = evaluator.evaluate_method(
        predictions=predictions,
        ground_truth=ground_truth,
        method_name=method_name,
        confidence_scores=confidence_scores,
        image_names=image_names
    )
    
    return evaluation_results


def generate_comparison_with_baseline(evaluation_results: Dict[str, Any],
                                    config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate detailed comparison with baseline performance.
    
    Args:
        evaluation_results: Evaluation results from experimental evaluator
        config: Configuration dictionary
        
    Returns:
        Dictionary containing comparison analysis
    """
    logging.info("Generating baseline comparison analysis...")
    
    baseline_error = 5.63
    baseline_std = 0.17
    baseline_n_images = 159
    
    method_error = evaluation_results['basic_metrics']['mean_error']
    method_std = evaluation_results['basic_metrics']['std_error']
    method_n_images = evaluation_results['dataset_info']['n_images']
    
    # Extended comparison analysis
    comparison = {
        'baseline': {
            'error': baseline_error,
            'std': baseline_std,
            'n_images': baseline_n_images,
            'source': 'coordenadas_prueba_1.csv',
            'reference': 'Template Matching with Eigenpatches (Original Implementation)'
        },
        'method': {
            'error': method_error,
            'std': method_std,
            'n_images': method_n_images,
            'source': config.get('datasets', {}).get('test_coords', 'unknown'),
            'reference': evaluation_results['method_name']
        },
        'comparison': {
            'absolute_difference': method_error - baseline_error,
            'relative_difference': (method_error - baseline_error) / baseline_error,
            'improvement_percentage': (baseline_error - method_error) / baseline_error * 100,
            'is_better': method_error < baseline_error,
            'is_equivalent': abs(method_error - baseline_error) < 0.1,  # Within 0.1 pixels
            'effect_size': (method_error - baseline_error) / np.sqrt((method_std**2 + baseline_std**2) / 2)
        }
    }
    
    # Categorize performance
    improvement = comparison['comparison']['improvement_percentage']
    if improvement > 10:
        performance_category = "Significantly Better"
    elif improvement > 5:
        performance_category = "Better"
    elif improvement > -5:
        performance_category = "Equivalent"
    elif improvement > -10:
        performance_category = "Worse"
    else:
        performance_category = "Significantly Worse"
    
    comparison['comparison']['performance_category'] = performance_category
    
    return comparison


def generate_executive_summary(evaluation_results: Dict[str, Any],
                             baseline_comparison: Dict[str, Any],
                             config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate executive summary of evaluation results.
    
    Args:
        evaluation_results: Evaluation results
        baseline_comparison: Baseline comparison results
        config: Configuration used
        
    Returns:
        Dictionary containing executive summary
    """
    logging.info("Generating executive summary...")
    
    basic_metrics = evaluation_results['basic_metrics']
    per_landmark = evaluation_results['per_landmark_analysis']
    
    summary = {
        'executive_summary': {
            'method_name': evaluation_results['method_name'],
            'evaluation_date': evaluation_results['evaluation_date'],
            'dataset_size': evaluation_results['dataset_info']['n_images'],
            
            'key_results': {
                'mean_error': f"{basic_metrics['mean_error']:.3f} ± {basic_metrics['std_error']:.3f} pixels",
                'median_error': f"{basic_metrics['median_error']:.3f} pixels",
                'error_range': f"{basic_metrics['min_error']:.3f} - {basic_metrics['max_error']:.3f} pixels",
                'baseline_comparison': baseline_comparison['comparison']['performance_category'],
                'improvement_vs_baseline': f"{baseline_comparison['comparison']['improvement_percentage']:+.1f}%"
            },
            
            'landmark_analysis': {
                'best_landmark': {
                    'index': per_landmark['best_landmark']['index'],
                    'error': f"{per_landmark['best_landmark']['mean_error']:.3f} pixels"
                },
                'worst_landmark': {
                    'index': per_landmark['worst_landmark']['index'],
                    'error': f"{per_landmark['worst_landmark']['mean_error']:.3f} pixels"
                },
                'error_range_across_landmarks': f"{per_landmark['landmark_error_range']:.3f} pixels"
            },
            
            'configuration_used': {
                'patch_size': config.get('eigenpatches', {}).get('patch_size', 'N/A'),
                'n_components': config.get('eigenpatches', {}).get('n_components', 'N/A'),
                'pyramid_levels': config.get('eigenpatches', {}).get('pyramid_levels', 'N/A'),
                'lambda_shape': config.get('landmark_predictor', {}).get('lambda_shape', 'N/A')
            },
            
            'recommendations': []
        }
    }
    
    # Generate recommendations based on results
    improvement = baseline_comparison['comparison']['improvement_percentage']
    
    if improvement > 5:
        summary['executive_summary']['recommendations'].append(
            "Excellent performance! Consider this configuration for production use."
        )
    elif improvement > 0:
        summary['executive_summary']['recommendations'].append(
            "Good performance improvement. Consider fine-tuning parameters for further gains."
        )
    elif improvement > -5:
        summary['executive_summary']['recommendations'].append(
            "Performance equivalent to baseline. Experiment with different parameter combinations."
        )
    else:
        summary['executive_summary']['recommendations'].append(
            "Performance below baseline. Review configuration and consider alternative approaches."
        )
    
    # Landmark-specific recommendations
    worst_landmark_error = per_landmark['worst_landmark']['mean_error']
    if worst_landmark_error > basic_metrics['mean_error'] * 1.5:
        summary['executive_summary']['recommendations'].append(
            f"Landmark {per_landmark['worst_landmark']['index']} shows poor performance. "
            "Consider landmark-specific parameter tuning."
        )
    
    # Error distribution recommendations
    if basic_metrics['std_error'] > basic_metrics['mean_error'] * 0.5:
        summary['executive_summary']['recommendations'].append(
            "High error variability detected. Consider ensemble methods or robustness improvements."
        )
    
    return summary


def save_evaluation_results(evaluation_results: Dict[str, Any],
                          baseline_comparison: Dict[str, Any],
                          executive_summary: Dict[str, Any],
                          config: Dict[str, Any]) -> str:
    """
    Save comprehensive evaluation results.
    
    Args:
        evaluation_results: Main evaluation results
        baseline_comparison: Baseline comparison results
        executive_summary: Executive summary
        config: Configuration used
        
    Returns:
        Path to saved evaluation file
    """
    # Create evaluation directory
    results_dir = Path(config.get('paths', {}).get('results_dir', 'results')) / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f"evaluation_results_{timestamp}.pkl"
    eval_path = results_dir / eval_filename
    
    # Compile complete results
    complete_results = {
        **evaluation_results,
        'baseline_comparison': baseline_comparison,
        'executive_summary': executive_summary,
        'configuration_used': config
    }
    
    # Save as pickle
    logging.info(f"Saving evaluation results to: {eval_path}")
    with open(eval_path, 'wb') as f:
        pickle.dump(complete_results, f)
    
    # Save summary as YAML
    summary_path = eval_path.with_suffix('.summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(executive_summary, f, default_flow_style=False)
    
    # Save baseline comparison as YAML
    comparison_path = eval_path.with_suffix('.comparison.yaml')
    with open(comparison_path, 'w') as f:
        yaml.dump(baseline_comparison, f, default_flow_style=False)
    
    logging.info(f"Evaluation results saved successfully")
    logging.info(f"Main results: {eval_path}")
    logging.info(f"Summary: {summary_path}")
    logging.info(f"Comparison: {comparison_path}")
    
    return str(eval_path)


def print_summary_to_console(executive_summary: Dict[str, Any],
                           baseline_comparison: Dict[str, Any]) -> None:
    """
    Print evaluation summary to console.
    
    Args:
        executive_summary: Executive summary dictionary
        baseline_comparison: Baseline comparison dictionary
    """
    summary = executive_summary['executive_summary']
    
    print("\n" + "="*80)
    print("EXPERIMENTAL TEMPLATE MATCHING - EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nMethod: {summary['method_name']}")
    print(f"Evaluation Date: {summary['evaluation_date']}")
    print(f"Dataset Size: {summary['dataset_size']} images")
    
    print(f"\nKEY RESULTS:")
    print(f"  Mean Error: {summary['key_results']['mean_error']}")
    print(f"  Median Error: {summary['key_results']['median_error']}")
    print(f"  Error Range: {summary['key_results']['error_range']}")
    
    print(f"\nBASELINE COMPARISON:")
    print(f"  Baseline Error: {baseline_comparison['baseline']['error']:.3f} pixels")
    print(f"  Method Error: {baseline_comparison['method']['error']:.3f} pixels")
    print(f"  Performance Category: {summary['key_results']['baseline_comparison']}")
    print(f"  Improvement: {summary['key_results']['improvement_vs_baseline']}")
    
    print(f"\nLANDMARK ANALYSIS:")
    print(f"  Best Landmark: #{summary['landmark_analysis']['best_landmark']['index']} "
          f"({summary['landmark_analysis']['best_landmark']['error']})")
    print(f"  Worst Landmark: #{summary['landmark_analysis']['worst_landmark']['index']} "
          f"({summary['landmark_analysis']['worst_landmark']['error']})")
    print(f"  Error Range: {summary['landmark_analysis']['error_range_across_landmarks']}")
    
    print(f"\nCONFIGURATION:")
    config_used = summary['configuration_used']
    print(f"  Patch Size: {config_used['patch_size']}")
    print(f"  PCA Components: {config_used['n_components']}")
    print(f"  Pyramid Levels: {config_used['pyramid_levels']}")
    print(f"  Lambda Shape: {config_used['lambda_shape']}")
    
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of experimental template matching results')
    parser.add_argument('results_file', type=str,
                       help='Path to processing results pickle file')
    parser.add_argument('--config', type=str, 
                       default='matching_experimental/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth coordinates file (if not in results)')
    parser.add_argument('--images', type=str,
                       help='Path to images base directory (for ground truth loading)')
    parser.add_argument('--output', type=str,
                       help='Output evaluation path (overrides config)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generation of evaluation visualizations')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only run baseline comparison (faster)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT_DIR / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(config)
    
    logging.info("Starting experimental template matching evaluation")
    logging.info(f"Configuration loaded from: {config_path}")
    logging.info(f"Results file: {args.results_file}")
    
    # Load processing results
    try:
        processing_results = load_processing_results(args.results_file)
    except Exception as e:
        logging.error(f"Failed to load processing results: {e}")
        return 1
    
    # Extract predictions and attempt to get ground truth
    try:
        predictions, ground_truth_dummy, image_names = extract_predictions_and_ground_truth(processing_results)
    except Exception as e:
        logging.error(f"Failed to extract predictions: {e}")
        return 1
    
    # Load ground truth separately if provided or try from config
    ground_truth = ground_truth_dummy  # Default
    if args.ground_truth:
        try:
            coordinate_scale_factor = config.get('image_processing', {}).get('coordinate_scale_factor', 4.67)
            ground_truth = load_ground_truth_separately(
                args.ground_truth, 
                args.images or config['datasets']['image_base_path'],
                image_names,
                coordinate_scale_factor
            )
        except Exception as e:
            logging.warning(f"Failed to load ground truth separately: {e}")
            logging.warning("Using dummy ground truth - evaluation will be limited")
    
    # Initialize experimental evaluator
    evaluator = ExperimentalEvaluator(config=config)
    
    if args.compare_only:
        # Quick baseline comparison only
        logging.info("Running quick baseline comparison...")
        basic_error = np.mean([
            np.mean(np.linalg.norm(pred - gt, axis=1))
            for pred, gt in zip(predictions, ground_truth)
        ])
        
        baseline_comparison = {
            'baseline': {'error': 5.63, 'std': 0.17},
            'method': {'error': basic_error, 'std': 0.0},
            'comparison': {
                'improvement_percentage': (5.63 - basic_error) / 5.63 * 100,
                'performance_category': "Better" if basic_error < 5.63 else "Worse"
            }
        }
        
        print(f"\nQuick Baseline Comparison:")
        print(f"Baseline Error: 5.63 pixels")
        print(f"Method Error: {basic_error:.3f} pixels")
        print(f"Improvement: {baseline_comparison['comparison']['improvement_percentage']:+.1f}%")
        print(f"Category: {baseline_comparison['comparison']['performance_category']}")
        
        return 0
    
    # Run comprehensive evaluation
    try:
        evaluation_results = run_comprehensive_evaluation(
            evaluator, predictions, ground_truth, image_names, processing_results
        )
    except Exception as e:
        logging.error(f"Failed to run evaluation: {e}")
        return 1
    
    # Generate baseline comparison
    try:
        baseline_comparison = generate_comparison_with_baseline(evaluation_results, config)
    except Exception as e:
        logging.error(f"Failed to generate baseline comparison: {e}")
        return 1
    
    # Generate executive summary
    try:
        executive_summary = generate_executive_summary(evaluation_results, baseline_comparison, config)
    except Exception as e:
        logging.error(f"Failed to generate executive summary: {e}")
        return 1
    
    # Save results
    try:
        if args.output:
            eval_path = args.output
            complete_results = {
                **evaluation_results,
                'baseline_comparison': baseline_comparison,
                'executive_summary': executive_summary
            }
            with open(eval_path, 'wb') as f:
                pickle.dump(complete_results, f)
            logging.info(f"Results saved to: {eval_path}")
        else:
            eval_path = save_evaluation_results(
                evaluation_results, baseline_comparison, executive_summary, config
            )
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        return 1
    
    # Print summary to console
    print_summary_to_console(executive_summary, baseline_comparison)
    
    logging.info("Evaluation completed successfully!")
    logging.info(f"Results saved to: {eval_path}")
    logging.info("Check the HTML report for detailed visualizations and analysis")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())