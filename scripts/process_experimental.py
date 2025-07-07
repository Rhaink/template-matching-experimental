#!/usr/bin/env python3
"""
Experimental processing script for template matching landmark detection.

This script provides an enhanced processing pipeline that replicates the exact
behavior of template_matching/scripts/process_all_images.py while adding
configuration management, detailed progress tracking, and experimental features.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple, Optional, Dict, Any
import argparse
import logging
import yaml
import time
from datetime import datetime
import pickle
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
MATCHING_EXPERIMENTAL_DIR = PROJECT_ROOT_DIR / "matching_experimental"
sys.path.insert(0, str(MATCHING_EXPERIMENTAL_DIR))

from core.experimental_predictor import ExperimentalLandmarkPredictor, PredictionResult


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
            logging.FileHandler(logs_dir / 'processing.log'),
            logging.StreamHandler()
        ]
    )


def load_test_dataset(coordinates_file: str, images_base_dir: str, 
                     num_landmarks: int = 15,
                     coordinate_scale_factor: float = 4.67) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load test dataset from coordinate CSV file and images.
    
    Args:
        coordinates_file: CSV file containing coordinates and image names
        images_base_dir: Base directory containing image subdirectories
        num_landmarks: Number of landmarks per shape
        coordinate_scale_factor: Scale factor from 64x64 to image size
        
    Returns:
        Tuple of (images, ground_truth_landmarks, image_names)
    """
    logging.info("Loading test dataset...")
    
    # Load coordinates (CSV files have no header)
    df = pd.read_csv(coordinates_file, header=None)
    logging.info(f"Loaded {len(df)} test samples")
    
    images = []
    ground_truth_landmarks = []
    image_names = []
    
    # Image directories to search
    image_dirs = [
        Path(images_base_dir) / "COVID" / "images",
        Path(images_base_dir) / "Normal" / "images", 
        Path(images_base_dir) / "Viral Pneumonia" / "images"
    ]
    
    successful_loads = 0
    failed_loads = 0
    
    for idx, row in df.iterrows():
        try:
            # Extract coordinates (30 values for 15 landmarks)
            coords = row.iloc[:30].values.astype(float)
            landmarks = coords.reshape(num_landmarks, 2)
            
            # Scale landmarks from 64x64 to actual image size
            landmarks_scaled = landmarks * coordinate_scale_factor
            
            # Extract image name
            image_name = row.iloc[30]
            
            # Find image file
            image_path = None
            for img_dir in image_dirs:
                potential_path = img_dir / image_name
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
                logging.warning(f"Image not found: {image_name}")
                failed_loads += 1
                continue
            
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                failed_loads += 1
                continue
            
            # Validate image size and landmarks
            h, w = image.shape
            if np.any(landmarks_scaled < 0) or np.any(landmarks_scaled[:, 0] >= w) or np.any(landmarks_scaled[:, 1] >= h):
                logging.warning(f"Invalid landmarks for image {image_name}: landmarks out of bounds")
                failed_loads += 1
                continue
            
            images.append(image)
            ground_truth_landmarks.append(landmarks_scaled)
            image_names.append(image_name)
            successful_loads += 1
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            failed_loads += 1
    
    logging.info(f"Successfully loaded {successful_loads} images, failed to load {failed_loads} images")
    
    if successful_loads == 0:
        raise RuntimeError("No images loaded successfully")
    
    return images, ground_truth_landmarks, image_names


def load_trained_model(model_path: str, config: Dict[str, Any]) -> ExperimentalLandmarkPredictor:
    """
    Load trained model from file.
    
    Args:
        model_path: Path to saved model
        config: Configuration dictionary
        
    Returns:
        Loaded ExperimentalLandmarkPredictor
    """
    logging.info(f"Loading trained model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize predictor with configuration
    predictor = ExperimentalLandmarkPredictor(config=config)
    
    # Load the trained model
    predictor.load(model_path)
    
    logging.info("Model loaded successfully")
    return predictor


def process_images(predictor: ExperimentalLandmarkPredictor, 
                  images: List[np.ndarray],
                  image_names: List[str],
                  config: Dict[str, Any]) -> List[PredictionResult]:
    """
    Process all images and generate predictions.
    
    Args:
        predictor: Trained landmark predictor
        images: List of test images
        image_names: List of image names
        config: Configuration dictionary
        
    Returns:
        List of prediction results
    """
    logging.info(f"Processing {len(images)} images...")
    
    results = []
    processing_times = []
    
    # Performance configuration
    perf_config = config.get('performance', {})
    batch_size = perf_config.get('batch_size', 1)  # Currently processing one by one
    
    # Progress bar
    progress_bar = tqdm(total=len(images), desc="Processing images")
    
    start_time = time.time()
    
    for i, (image, image_name) in enumerate(zip(images, image_names)):
        try:
            # Process single image
            image_start_time = time.time()
            
            result = predictor.predict_landmarks(
                image, 
                initial_landmarks=None,
                return_detailed=True
            )
            
            image_processing_time = time.time() - image_start_time
            processing_times.append(image_processing_time)
            
            # Add image metadata to result
            result.image_name = image_name
            result.image_index = i
            
            results.append(result)
            
            # Update progress
            progress_bar.update(1)
            
            # Log progress periodically
            if (i + 1) % 50 == 0 or i == len(images) - 1:
                avg_time = np.mean(processing_times[-50:])
                remaining = len(images) - (i + 1)
                eta = remaining * avg_time
                logging.info(f"Processed {i + 1}/{len(images)} images. "
                           f"Avg time: {avg_time:.3f}s. ETA: {eta:.1f}s")
        
        except Exception as e:
            logging.error(f"Failed to process image {image_name}: {e}")
            # Create empty result for failed processing
            empty_result = PredictionResult(
                landmarks=np.zeros((15, 2)),
                confidence=None,
                processing_time=0.0
            )
            empty_result.image_name = image_name
            empty_result.image_index = i
            results.append(empty_result)
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    total_time = time.time() - start_time
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    logging.info(f"Processing completed in {total_time:.2f} seconds")
    logging.info(f"Average processing time per image: {avg_processing_time:.3f} seconds")
    logging.info(f"Images per second: {len(images) / total_time:.2f}")
    
    return results


def compute_errors(predictions: List[PredictionResult], 
                  ground_truth: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compute prediction errors against ground truth.
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth landmark arrays
        
    Returns:
        Dictionary containing error statistics
    """
    logging.info("Computing prediction errors...")
    
    errors = []
    per_landmark_errors = [[] for _ in range(15)]  # Assuming 15 landmarks
    per_image_errors = []
    
    for pred_result, gt_landmarks in zip(predictions, ground_truth):
        # Compute point-to-point error
        pred_landmarks = pred_result.landmarks
        
        # Euclidean distances per landmark
        landmark_distances = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)
        
        # Mean error for this image
        mean_error = np.mean(landmark_distances)
        errors.append(mean_error)
        
        # Store per-landmark errors
        for i, error in enumerate(landmark_distances):
            per_landmark_errors[i].append(error)
        
        # Store per-image data
        per_image_errors.append({
            'image_name': getattr(pred_result, 'image_name', f'image_{len(per_image_errors)}'),
            'mean_error': mean_error,
            'landmark_errors': landmark_distances.tolist(),
            'processing_time': pred_result.processing_time,
            'confidence': pred_result.confidence.tolist() if pred_result.confidence is not None else None
        })
    
    # Compute overall statistics
    errors = np.array(errors)
    
    error_stats = {
        'overall_statistics': {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'q25_error': float(np.percentile(errors, 25)),
            'q75_error': float(np.percentile(errors, 75))
        },
        'per_landmark_statistics': {},
        'per_image_results': per_image_errors,
        'raw_errors': errors.tolist()
    }
    
    # Compute per-landmark statistics
    for i, landmark_errors in enumerate(per_landmark_errors):
        landmark_errors = np.array(landmark_errors)
        error_stats['per_landmark_statistics'][f'landmark_{i}'] = {
            'mean_error': float(np.mean(landmark_errors)),
            'std_error': float(np.std(landmark_errors)),
            'median_error': float(np.median(landmark_errors)),
            'min_error': float(np.min(landmark_errors)),
            'max_error': float(np.max(landmark_errors))
        }
    
    # Identify best and worst cases
    sorted_results = sorted(per_image_errors, key=lambda x: x['mean_error'])
    error_stats['best_cases'] = sorted_results[:5]
    error_stats['worst_cases'] = sorted_results[-5:]
    
    logging.info(f"Error computation completed. Mean error: {error_stats['overall_statistics']['mean_error']:.3f} pixels")
    
    return error_stats


def save_results(predictions: List[PredictionResult],
                error_stats: Dict[str, Any],
                config: Dict[str, Any],
                model_path: str) -> str:
    """
    Save processing results to files.
    
    Args:
        predictions: List of prediction results
        error_stats: Error statistics
        config: Configuration used
        model_path: Path to model used
        
    Returns:
        Path to saved results file
    """
    # Create results directory
    results_dir = Path(config.get('paths', {}).get('results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate results filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"processing_results_{timestamp}.pkl"
    results_path = results_dir / results_filename
    
    # Prepare results for saving
    results_data = {
        'processing_date': datetime.now().isoformat(),
        'model_path': model_path,
        'configuration': config,
        'predictions': predictions,
        'error_statistics': error_stats,
        'n_images': len(predictions),
        'baseline_comparison': {
            'baseline_error': 5.63,
            'baseline_std': 0.17,
            'method_error': error_stats['overall_statistics']['mean_error'],
            'method_std': error_stats['overall_statistics']['std_error'],
            'improvement': (5.63 - error_stats['overall_statistics']['mean_error']) / 5.63 * 100
        }
    }
    
    # Save as pickle
    logging.info(f"Saving results to: {results_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    # Save summary as YAML
    summary_path = results_path.with_suffix('.summary.yaml')
    summary_data = {
        'processing_date': results_data['processing_date'],
        'model_path': model_path,
        'n_images': len(predictions),
        'overall_statistics': error_stats['overall_statistics'],
        'baseline_comparison': results_data['baseline_comparison'],
        'best_case': error_stats['best_cases'][0] if error_stats['best_cases'] else None,
        'worst_case': error_stats['worst_cases'][0] if error_stats['worst_cases'] else None
    }
    
    with open(summary_path, 'w') as f:
        yaml.dump(summary_data, f, default_flow_style=False)
    
    # Save detailed results as CSV
    csv_path = results_path.with_suffix('.csv')
    df_data = []
    for result in error_stats['per_image_results']:
        df_data.append({
            'image_name': result['image_name'],
            'mean_error': result['mean_error'],
            'processing_time': result['processing_time']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    
    logging.info(f"Results saved successfully")
    logging.info(f"Pickle file: {results_path}")
    logging.info(f"Summary: {summary_path}")
    logging.info(f"CSV: {csv_path}")
    
    return str(results_path)


def generate_quick_visualizations(predictions: List[PredictionResult],
                                ground_truth: List[np.ndarray],
                                images: List[np.ndarray],
                                image_names: List[str],
                                config: Dict[str, Any],
                                max_visualizations: int = 10) -> None:
    """
    Generate quick visualizations for best and worst cases.
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth landmarks
        images: List of images
        image_names: List of image names
        config: Configuration
        max_visualizations: Maximum number of visualizations to generate
    """
    import matplotlib.pyplot as plt
    
    logging.info(f"Generating up to {max_visualizations} quick visualizations...")
    
    # Create visualizations directory
    vis_dir = Path(config.get('paths', {}).get('visualizations_dir', 'visualizations')) / "processing"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute errors for sorting
    errors = []
    for pred_result, gt_landmarks in zip(predictions, ground_truth):
        mean_error = np.mean(np.linalg.norm(pred_result.landmarks - gt_landmarks, axis=1))
        errors.append((mean_error, pred_result, gt_landmarks))
    
    # Sort by error
    errors.sort(key=lambda x: x[0])
    
    # Generate visualizations for best and worst cases
    n_best = min(max_visualizations // 2, len(errors))
    n_worst = min(max_visualizations - n_best, len(errors))
    
    cases_to_visualize = errors[:n_best] + errors[-n_worst:]
    
    for i, (error, pred_result, gt_landmarks) in enumerate(cases_to_visualize):
        try:
            # Find corresponding image
            image_idx = getattr(pred_result, 'image_index', i)
            if image_idx < len(images):
                image = images[image_idx]
                image_name = image_names[image_idx] if image_idx < len(image_names) else f"image_{image_idx}"
            else:
                continue
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(image, cmap='gray')
            
            # Plot ground truth (green)
            gt_x, gt_y = gt_landmarks[:, 0], gt_landmarks[:, 1]
            plt.scatter(gt_x, gt_y, c='green', s=50, alpha=0.7, label='Ground Truth')
            
            # Plot predictions (red)
            pred_x, pred_y = pred_result.landmarks[:, 0], pred_result.landmarks[:, 1]
            plt.scatter(pred_x, pred_y, c='red', s=50, alpha=0.7, label='Predicted')
            
            # Connect corresponding points
            for j in range(len(gt_landmarks)):
                plt.plot([gt_x[j], pred_x[j]], [gt_y[j], pred_y[j]], 'b-', alpha=0.3, linewidth=1)
            
            plt.title(f"{image_name}\nError: {error:.3f} pixels")
            plt.legend()
            plt.axis('off')
            
            # Save
            case_type = "best" if error == errors[0][0] else "worst" if error == errors[-1][0] else "case"
            filename = f"{case_type}_{i:02d}_{image_name.replace('.', '_')}_error_{error:.3f}.png"
            plt.savefig(vis_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.warning(f"Failed to generate visualization {i}: {e}")
    
    logging.info(f"Quick visualizations saved to: {vis_dir}")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process images with experimental template matching model')
    parser.add_argument('--config', type=str, 
                       default='matching_experimental/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--coordinates', type=str,
                       help='Path to test coordinates CSV file (overrides config)')
    parser.add_argument('--images', type=str,
                       help='Path to images base directory (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output results path (overrides config)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generation of quick visualizations')
    parser.add_argument('--max-vis', type=int, default=10,
                       help='Maximum number of visualizations to generate')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT_DIR / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.coordinates:
        config['datasets']['test_coords'] = args.coordinates
    if args.images:
        config['datasets']['image_base_path'] = args.images
    
    # Setup logging
    setup_logging(config)
    
    logging.info("Starting experimental template matching processing")
    logging.info(f"Configuration loaded from: {config_path}")
    logging.info(f"Model path: {args.model}")
    
    # Resolve paths
    project_root = Path(config.get('paths', {}).get('project_root', PROJECT_ROOT_DIR))
    coordinates_file = project_root / config['datasets']['test_coords']
    images_base_dir = project_root / config['datasets']['image_base_path']
    
    # Load test dataset
    try:
        coordinate_scale_factor = config.get('image_processing', {}).get('coordinate_scale_factor', 4.67)
        images, ground_truth_landmarks, image_names = load_test_dataset(
            str(coordinates_file), str(images_base_dir),
            coordinate_scale_factor=coordinate_scale_factor
        )
    except Exception as e:
        logging.error(f"Failed to load test dataset: {e}")
        return 1
    
    # Load trained model
    try:
        predictor = load_trained_model(args.model, config)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return 1
    
    # Process images
    try:
        predictions = process_images(predictor, images, image_names, config)
    except Exception as e:
        logging.error(f"Failed to process images: {e}")
        return 1
    
    # Compute errors
    try:
        error_stats = compute_errors(predictions, ground_truth_landmarks)
    except Exception as e:
        logging.error(f"Failed to compute errors: {e}")
        return 1
    
    # Save results
    try:
        if args.output:
            results_path = args.output
            # Save with custom path logic
            results_data = {
                'processing_date': datetime.now().isoformat(),
                'model_path': args.model,
                'configuration': config,
                'predictions': predictions,
                'error_statistics': error_stats,
                'n_images': len(predictions)
            }
            with open(results_path, 'wb') as f:
                pickle.dump(results_data, f)
            logging.info(f"Results saved to: {results_path}")
        else:
            results_path = save_results(predictions, error_stats, config, args.model)
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        return 1
    
    # Generate quick visualizations
    if not args.no_visualizations:
        try:
            generate_quick_visualizations(
                predictions, ground_truth_landmarks, images, 
                image_names, config, args.max_vis
            )
        except Exception as e:
            logging.warning(f"Failed to generate visualizations: {e}")
    
    # Print summary
    overall_stats = error_stats['overall_statistics']
    baseline_error = 5.63
    improvement = (baseline_error - overall_stats['mean_error']) / baseline_error * 100
    
    logging.info("Processing completed successfully!")
    logging.info(f"Results saved to: {results_path}")
    logging.info("=== PERFORMANCE SUMMARY ===")
    logging.info(f"Images processed: {len(predictions)}")
    logging.info(f"Mean error: {overall_stats['mean_error']:.3f} ± {overall_stats['std_error']:.3f} pixels")
    logging.info(f"Median error: {overall_stats['median_error']:.3f} pixels")
    logging.info(f"Error range: {overall_stats['min_error']:.3f} - {overall_stats['max_error']:.3f} pixels")
    logging.info(f"Baseline comparison: {baseline_error:.2f} pixels")
    logging.info(f"Improvement vs baseline: {improvement:+.1f}%")
    logging.info("Next steps:")
    logging.info("1. Run evaluate_experimental.py for comprehensive analysis")
    logging.info("2. Compare with other methods using comparison tools")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())