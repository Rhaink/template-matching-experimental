#!/usr/bin/env python3
"""
Generate visualizations exactly like the original template_matching system.

This script creates visualizations with identical naming, format, and style 
to template_matching/scripts/visualize_all_test_images.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import pickle
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import logging

# Add project paths
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
EXPERIMENTAL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from core.visualization_utils import (
    get_lung_connections, determine_pathology, get_pathology_colors,
    load_image_by_name
)


def setup_logging():
    """Setup logging for the exact generation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('exact_visualization_generation.log'),
            logging.StreamHandler()
        ]
    )


class ExactTemplateMatchingVisualizer:
    """Generate visualizations exactly like the original template_matching system."""
    
    def __init__(self, results_path: str, config_path: str, output_dir: str):
        """Initialize the exact visualizer."""
        self.results_path = Path(results_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Load results
        with open(self.results_path, 'rb') as f:
            self.results = pickle.load(f)
        
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get images base directory from config
        project_root = Path(config['paths']['project_root'])
        self.images_base_dir = project_root / config['datasets']['image_base_path']
        
        # Setup output directories
        self.landmarks_dir = self.output_dir / 'landmark_predictions'
        self.contours_dir = self.output_dir / 'lung_contours'
        self.comparison_dir = self.output_dir / 'side_by_side'
        
        for dir_path in [self.landmarks_dir, self.contours_dir, self.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Pre-compute anatomical connections
        self.contour_connections, self.mediastinal_connections = get_lung_connections()
        self.colors = get_pathology_colors()
        
        self.logger.info(f"Initialized exact visualizer for {len(self.results['predictions'])} images")
    
    def generate_exact_landmark_visualization(self, idx: int, image: np.ndarray, 
                                            landmarks: np.ndarray, image_name: str,
                                            error: float) -> plt.Figure:
        """Generate landmark visualization exactly like original template_matching."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display image
        ax.imshow(image, cmap='gray', aspect='equal')
        
        # Determine pathology and color
        pathology = determine_pathology(image_name)
        landmark_color = self.colors.get(pathology, self.colors['default'])
        
        # Draw anatomical connections exactly like original
        for start_idx, end_idx in self.contour_connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=landmark_color, linewidth=2, alpha=0.8)
        
        # Draw mediastinal connections
        for start_idx, end_idx in self.mediastinal_connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=landmark_color, linewidth=2, alpha=0.8, linestyle='--')
        
        # Draw landmark points
        ax.scatter(landmarks[:, 0], landmarks[:, 1], 
                  c=landmark_color, s=100, alpha=0.9, edgecolors='black', linewidth=1)
        
        # Add landmark numbers like original
        for i, (x, y) in enumerate(landmarks):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Set title exactly like original
        title = f"{pathology} - {image_name} - Error: {error:.3f} px"
        ax.set_title(title, fontsize=12, weight='bold')
        
        # Formatting exactly like original
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)  # Invert y-axis
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate_exact_contour_visualization(self, idx: int, image: np.ndarray,
                                           landmarks: np.ndarray, image_name: str,
                                           error: float) -> plt.Figure:
        """Generate contour visualization exactly like original template_matching."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display image
        ax.imshow(image, cmap='gray', aspect='equal')
        
        # Determine pathology and color
        pathology = determine_pathology(image_name)
        contour_color = self.colors.get(pathology, self.colors['default'])
        
        # Draw contour lines only (no individual points)
        for start_idx, end_idx in self.contour_connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=contour_color, linewidth=3, alpha=0.9)
        
        # Draw mediastinal lines
        for start_idx, end_idx in self.mediastinal_connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=contour_color, linewidth=3, alpha=0.9, linestyle='--')
        
        # Set title
        title = f"{pathology} - {image_name} - Lung Contours"
        ax.set_title(title, fontsize=12, weight='bold')
        
        # Formatting
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate_exact_side_by_side_visualization(self, idx: int, image: np.ndarray,
                                                ground_truth: np.ndarray,
                                                prediction: np.ndarray,
                                                image_name: str, error: float) -> plt.Figure:
        """Generate side-by-side visualization exactly like original template_matching."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        pathology = determine_pathology(image_name)
        
        # Ground truth plot (left)
        ax1.imshow(image, cmap='gray', aspect='equal')
        
        # Draw ground truth connections in green
        for start_idx, end_idx in self.contour_connections:
            start_point = ground_truth[start_idx]
            end_point = ground_truth[end_idx]
            ax1.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    color='green', linewidth=2, alpha=0.8)
        
        for start_idx, end_idx in self.mediastinal_connections:
            start_point = ground_truth[start_idx]
            end_point = ground_truth[end_idx]
            ax1.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    color='green', linewidth=2, alpha=0.8, linestyle='--')
        
        ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], 
                   c='green', s=100, alpha=0.9, edgecolors='black', linewidth=1)
        ax1.set_title(f'Ground Truth - {pathology}', fontsize=14, weight='bold')
        ax1.set_xlim(0, image.shape[1])
        ax1.set_ylim(image.shape[0], 0)
        ax1.axis('off')
        
        # Prediction plot (right)
        ax2.imshow(image, cmap='gray', aspect='equal')
        
        # Draw prediction connections in red
        for start_idx, end_idx in self.contour_connections:
            start_point = prediction[start_idx]
            end_point = prediction[end_idx]
            ax2.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    color='red', linewidth=2, alpha=0.8)
        
        for start_idx, end_idx in self.mediastinal_connections:
            start_point = prediction[start_idx]
            end_point = prediction[end_idx]
            ax2.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    color='red', linewidth=2, alpha=0.8, linestyle='--')
        
        ax2.scatter(prediction[:, 0], prediction[:, 1], 
                   c='red', s=100, alpha=0.9, edgecolors='black', linewidth=1)
        ax2.set_title(f'Prediction - Error: {error:.3f} px', fontsize=14, weight='bold')
        ax2.set_xlim(0, image.shape[1])
        ax2.set_ylim(image.shape[0], 0)
        ax2.axis('off')
        
        # Overall title
        fig.suptitle(f'{image_name}', fontsize=16, weight='bold')
        plt.tight_layout()
        return fig
    
    def generate_exact_filename(self, idx: int, image_name: str, viz_type: str) -> str:
        """Generate filename exactly like original template_matching."""
        # Original format: {idx:03d}_{image_name}_{type}.png
        return f"{idx:03d}_{image_name}_{viz_type}.png"
    
    def generate_all_exact_visualizations(self) -> Dict[str, int]:
        """Generate all 159 visualizations exactly like original template_matching."""
        num_images = len(self.results['image_names'])
        self.logger.info(f"Generating {num_images} exact visualizations...")
        
        # Clear existing files
        for dir_path in [self.landmarks_dir, self.contours_dir, self.comparison_dir]:
            for file in dir_path.glob('*.png'):
                file.unlink()
        
        results = {
            'landmarks_success': 0,
            'landmarks_failed': 0,
            'contours_success': 0,
            'contours_failed': 0,
            'comparisons_success': 0,
            'comparisons_failed': 0
        }
        
        # Process each image with progress bar
        for idx in tqdm(range(num_images), desc="Generating exact visualizations"):
            try:
                image_name = self.results['image_names'][idx]
                prediction = self.results['predictions'][idx]
                ground_truth = self.results['ground_truth'][idx]
                
                # Calculate error
                error = np.mean(np.sqrt(np.sum((prediction - ground_truth)**2, axis=1)))
                
                # Load image
                image = load_image_by_name(image_name, str(self.images_base_dir))
                if image is None:
                    self.logger.warning(f"Could not load image: {image_name}")
                    results['landmarks_failed'] += 1
                    results['contours_failed'] += 1
                    results['comparisons_failed'] += 1
                    continue
                
                # Generate exact filenames like original
                landmark_filename = self.generate_exact_filename(idx, image_name, 'landmarks')
                contour_filename = self.generate_exact_filename(idx, image_name, 'contour')
                comparison_filename = self.generate_exact_filename(idx, image_name, 'comparison')
                
                # Generate landmark visualization
                try:
                    fig = self.generate_exact_landmark_visualization(idx, image, prediction, 
                                                                   image_name, error)
                    output_path = self.landmarks_dir / landmark_filename
                    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                    results['landmarks_success'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to generate landmark viz for {image_name}: {e}")
                    results['landmarks_failed'] += 1
                
                # Generate contour visualization
                try:
                    fig = self.generate_exact_contour_visualization(idx, image, prediction,
                                                                  image_name, error)
                    output_path = self.contours_dir / contour_filename
                    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                    results['contours_success'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to generate contour viz for {image_name}: {e}")
                    results['contours_failed'] += 1
                
                # Generate side-by-side comparison
                try:
                    fig = self.generate_exact_side_by_side_visualization(idx, image, ground_truth,
                                                                        prediction, image_name, error)
                    output_path = self.comparison_dir / comparison_filename
                    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    plt.close(fig)
                    results['comparisons_success'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to generate comparison viz for {image_name}: {e}")
                    results['comparisons_failed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing image {idx}: {e}")
                results['landmarks_failed'] += 1
                results['contours_failed'] += 1
                results['comparisons_failed'] += 1
        
        return results


def main():
    """Main execution function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Paths
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / 'data' / 'results_coordenadas_prueba_1.pkl'
    config_path = script_dir.parent / 'configs' / 'default_config.yaml'
    output_dir = script_dir.parent / 'visualizations'
    
    try:
        logger.info("Starting exact template matching visualization generation...")
        
        # Initialize visualizer
        visualizer = ExactTemplateMatchingVisualizer(
            str(results_path),
            str(config_path),
            str(output_dir)
        )
        
        # Generate all visualizations
        results = visualizer.generate_all_exact_visualizations()
        
        # Print summary
        print("\n" + "="*80)
        print("EXACT TEMPLATE MATCHING VISUALIZATION GENERATION COMPLETE")
        print("="*80)
        print(f"Landmark Predictions: {results['landmarks_success']} success, {results['landmarks_failed']} failed")
        print(f"Lung Contours: {results['contours_success']} success, {results['contours_failed']} failed")
        print(f"Side-by-Side Comparisons: {results['comparisons_success']} success, {results['comparisons_failed']} failed")
        print("="*80)
        print(f"Output directory: {output_dir}")
        
        # Validate exact count
        landmark_count = len(list((output_dir / 'landmark_predictions').glob('*.png')))
        contour_count = len(list((output_dir / 'lung_contours').glob('*.png')))
        comparison_count = len(list((output_dir / 'side_by_side').glob('*.png')))
        
        print(f"\nFinal file counts:")
        print(f"  Landmarks: {landmark_count} files")
        print(f"  Contours: {contour_count} files")
        print(f"  Comparisons: {comparison_count} files")
        
        if landmark_count == 159 and contour_count == 159 and comparison_count == 159:
            print("\n✅ SUCCESS: All 159 visualizations generated with exact template_matching format!")
        else:
            print(f"\n⚠️ Warning: Expected 159 files each, got {landmark_count}, {contour_count}, {comparison_count}")
        
        logger.info("Exact visualization generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Exact visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())