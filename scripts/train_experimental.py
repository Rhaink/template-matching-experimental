#!/usr/bin/env python3
"""
Experimental training script for template matching landmark detection.

This script provides an enhanced training pipeline with YAML configuration,
detailed logging, and experimental features while maintaining full compatibility
with the original template_matching implementation.
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

# Add project root to path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
MATCHING_EXPERIMENTAL_DIR = PROJECT_ROOT_DIR / "matching_experimental"
sys.path.insert(0, str(MATCHING_EXPERIMENTAL_DIR))

from core.experimental_predictor import ExperimentalLandmarkPredictor


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
            logging.FileHandler(logs_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def load_dataset(coordinates_file: str, images_base_dir: str, 
                num_landmarks: int = 15, 
                coordinate_scale_factor: float = 4.67) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load training dataset from coordinate CSV file and images.
    
    Args:
        coordinates_file: CSV file containing coordinates and image names
        images_base_dir: Base directory containing image subdirectories
        num_landmarks: Number of landmarks per shape
        coordinate_scale_factor: Scale factor from 64x64 to image size
        
    Returns:
        Tuple of (images, landmarks_list)
    """
    logging.info("Loading dataset...")
    
    # Load coordinates (CSV files have no header)
    df = pd.read_csv(coordinates_file, header=None)
    logging.info(f"Loaded {len(df)} training samples")
    
    images = []
    landmarks_list = []
    
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
            landmarks_list.append(landmarks_scaled)
            successful_loads += 1
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            failed_loads += 1
    
    logging.info(f"Successfully loaded {successful_loads} images, failed to load {failed_loads} images")
    
    if successful_loads == 0:
        raise RuntimeError("No images loaded successfully")
    
    return images, landmarks_list


def validate_dataset(images: List[np.ndarray], landmarks_list: List[np.ndarray]) -> Dict[str, Any]:
    """
    Validate dataset and compute statistics.
    
    Args:
        images: List of images
        landmarks_list: List of landmark arrays
        
    Returns:
        Dictionary containing dataset statistics
    """
    logging.info("Validating dataset...")
    
    # Basic validation
    assert len(images) == len(landmarks_list), "Number of images and landmarks must match"
    
    # Compute statistics
    image_shapes = [img.shape for img in images]
    unique_shapes = set(image_shapes)
    
    # Landmark statistics
    all_landmarks = np.array(landmarks_list)
    landmark_means = np.mean(all_landmarks, axis=0)
    landmark_stds = np.std(all_landmarks, axis=0)
    
    # Intensity statistics
    intensities = [img.mean() for img in images]
    
    stats = {
        'n_images': len(images),
        'n_landmarks': len(landmarks_list[0]) if landmarks_list else 0,
        'image_shapes': list(unique_shapes),
        'landmark_statistics': {
            'mean_coordinates': landmark_means.tolist(),
            'std_coordinates': landmark_stds.tolist(),
            'coordinate_ranges': {
                'x_min': float(np.min(all_landmarks[:, :, 0])),
                'x_max': float(np.max(all_landmarks[:, :, 0])),
                'y_min': float(np.min(all_landmarks[:, :, 1])),
                'y_max': float(np.max(all_landmarks[:, :, 1]))
            }
        },
        'intensity_statistics': {
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities))
        }
    }
    
    logging.info(f"Dataset validation complete: {stats['n_images']} images, {stats['n_landmarks']} landmarks")
    return stats


def train_model(config: Dict[str, Any], images: List[np.ndarray], 
                landmarks_list: List[np.ndarray]) -> ExperimentalLandmarkPredictor:
    """
    Train the experimental landmark predictor.
    
    Args:
        config: Configuration dictionary
        images: List of training images
        landmarks_list: List of landmark arrays
        
    Returns:
        Trained ExperimentalLandmarkPredictor
    """
    logging.info("Initializing experimental landmark predictor...")
    
    # Initialize predictor with configuration
    predictor = ExperimentalLandmarkPredictor(config=config)
    
    # Train the model
    logging.info("Starting model training...")
    start_time = time.time()
    
    predictor.train(images, landmarks_list)
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    return predictor


def save_model(predictor: ExperimentalLandmarkPredictor, config: Dict[str, Any], 
               dataset_stats: Dict[str, Any]) -> str:
    """
    Save the trained model with metadata.
    
    Args:
        predictor: Trained predictor
        config: Configuration used for training
        dataset_stats: Dataset statistics
        
    Returns:
        Path to saved model
    """
    # Create models directory
    models_dir = Path(config.get('paths', {}).get('models_dir', 'models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"experimental_predictor_{timestamp}.pkl"
    model_path = models_dir / model_filename
    
    # Save model
    logging.info(f"Saving model to: {model_path}")
    predictor.save(str(model_path))
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_path': str(model_path),
        'configuration': config,
        'dataset_statistics': dataset_stats,
        'training_statistics': predictor.get_prediction_statistics()
    }
    
    metadata_path = model_path.with_suffix('.training_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logging.info(f"Model and metadata saved successfully")
    return str(model_path)


def generate_training_report(model_path: str, config: Dict[str, Any], 
                           dataset_stats: Dict[str, Any], 
                           training_time: float) -> None:
    """
    Generate comprehensive training report.
    
    Args:
        model_path: Path to saved model
        config: Configuration used
        dataset_stats: Dataset statistics
        training_time: Training duration in seconds
    """
    # Create reports directory
    reports_dir = Path(config.get('paths', {}).get('results_dir', 'results')) / "training_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report_path = reports_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report - Experimental Template Matching</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
            .table {{ width: 100%; border-collapse: collapse; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Training Report - Experimental Template Matching</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Model saved to: {model_path}</p>
        </div>
        
        <div class="section">
            <h2>Training Configuration</h2>
            <div class="metric">Patch Size: {config.get('eigenpatches', {}).get('patch_size', 'N/A')}</div>
            <div class="metric">PCA Components: {config.get('eigenpatches', {}).get('n_components', 'N/A')}</div>
            <div class="metric">Pyramid Levels: {config.get('eigenpatches', {}).get('pyramid_levels', 'N/A')}</div>
            <div class="metric">Lambda Shape: {config.get('landmark_predictor', {}).get('lambda_shape', 'N/A')}</div>
            <div class="metric">Max Iterations: {config.get('landmark_predictor', {}).get('max_iterations', 'N/A')}</div>
        </div>
        
        <div class="section">
            <h2>Dataset Statistics</h2>
            <div class="metric">Number of Images: {dataset_stats.get('n_images', 'N/A')}</div>
            <div class="metric">Number of Landmarks: {dataset_stats.get('n_landmarks', 'N/A')}</div>
            <div class="metric">Image Shapes: {dataset_stats.get('image_shapes', 'N/A')}</div>
            <div class="metric">Mean Intensity: {dataset_stats.get('intensity_statistics', {}).get('mean_intensity', 'N/A'):.2f}</div>
            <div class="metric">Std Intensity: {dataset_stats.get('intensity_statistics', {}).get('std_intensity', 'N/A'):.2f}</div>
        </div>
        
        <div class="section">
            <h2>Training Performance</h2>
            <div class="metric">Training Time: {training_time:.2f} seconds</div>
            <div class="metric">Images per Second: {dataset_stats.get('n_images', 0) / training_time:.2f}</div>
        </div>
        
        <div class="section">
            <h2>Next Steps</h2>
            <ul>
                <li>Run processing script to test the model on test dataset</li>
                <li>Use evaluation script to generate comprehensive analysis</li>
                <li>Compare with baseline performance (5.63±0.17 pixels)</li>
                <li>Experiment with different configurations</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Training report generated: {report_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train experimental template matching model')
    parser.add_argument('--config', type=str, 
                       default='matching_experimental/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--coordinates', type=str,
                       help='Path to coordinates CSV file (overrides config)')
    parser.add_argument('--images', type=str,
                       help='Path to images base directory (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output model path (overrides config)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset without training')
    
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
        config['datasets']['training_coords'] = args.coordinates
    if args.images:
        config['datasets']['image_base_path'] = args.images
    
    # Setup logging
    setup_logging(config)
    
    logging.info("Starting experimental template matching training")
    logging.info(f"Configuration loaded from: {config_path}")
    
    # Resolve paths
    project_root = Path(config.get('paths', {}).get('project_root', PROJECT_ROOT_DIR))
    coordinates_file = project_root / config['datasets']['training_coords']
    images_base_dir = project_root / config['datasets']['image_base_path']
    
    # Load dataset
    try:
        coordinate_scale_factor = config.get('image_processing', {}).get('coordinate_scale_factor', 4.67)
        images, landmarks_list = load_dataset(
            str(coordinates_file), str(images_base_dir), 
            coordinate_scale_factor=coordinate_scale_factor
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return 1
    
    # Validate dataset
    try:
        dataset_stats = validate_dataset(images, landmarks_list)
    except Exception as e:
        logging.error(f"Dataset validation failed: {e}")
        return 1
    
    if args.validate_only:
        logging.info("Dataset validation completed. Exiting without training.")
        return 0
    
    # Train model
    try:
        start_time = time.time()
        predictor = train_model(config, images, landmarks_list)
        training_time = time.time() - start_time
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1
    
    # Save model
    try:
        model_path = save_model(predictor, config, dataset_stats)
        if args.output:
            # Also save to specified path
            predictor.save(args.output)
            logging.info(f"Model also saved to: {args.output}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        return 1
    
    # Generate training report
    try:
        generate_training_report(model_path, config, dataset_stats, training_time)
    except Exception as e:
        logging.warning(f"Failed to generate training report: {e}")
    
    logging.info("Training completed successfully!")
    logging.info(f"Model saved to: {model_path}")
    logging.info("Next steps:")
    logging.info("1. Run process_experimental.py to test the model")
    logging.info("2. Run evaluate_experimental.py to generate comprehensive analysis")
    logging.info("3. Compare with baseline performance (5.63±0.17 pixels)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())