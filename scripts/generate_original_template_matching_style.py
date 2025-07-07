#!/usr/bin/env python3
"""
Generate visualizations exactly like the original template_matching system.

This script replicates the exact visual format from:
/home/donrobot/Projects/Tesiscopia/template_matching/scripts/visualize_all_test_images.py

Visual format:
- Ground truth: lime green circles with dark green edges
- Predictions: red X markers with thick lines
- Error lines: yellow connecting lines between GT and predictions
- Landmark numbers: white text with black background boxes
- Better visual presentation than experimental platform
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
TESISCOPIA_DIR = PROJECT_ROOT_DIR / "Tesiscopia"
sys.path.insert(0, str(EXPERIMENTAL_DIR))
sys.path.insert(0, str(TESISCOPIA_DIR / "pulmones" / "src"))

from core.visualization_utils import load_image_by_name, determine_pathology
from utils import asm_utils


def setup_logging():
    """Setup logging for the original style generation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('original_template_matching_style.log'),
            logging.StreamHandler()
        ]
    )


class OriginalTemplateMatchingVisualizer:
    """Generate visualizations exactly like the original template_matching system."""
    
    def __init__(self, results_path: str, config_path: str, output_dir: str):
        """Initialize the original style visualizer."""
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
        
        # Setup output directories (using original template_matching structure)
        self.landmarks_dir = self.output_dir / 'landmark_predictions'
        self.contours_dir = self.output_dir / 'lung_contours'
        self.comparison_dir = self.output_dir / 'side_by_side'
        
        for dir_path in [self.landmarks_dir, self.contours_dir, self.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Define anatomical connections exactly like original template_matching
        self.contour_connections = [
            (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
            (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
        ]
        self.midline_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
        
        self.logger.info(f"Initialized original style visualizer for {len(self.results['predictions'])} images")
    
    def generate_original_landmark_visualization(self, idx: int, image: np.ndarray, 
                                              pred_landmarks: np.ndarray, 
                                              gt_landmarks: np.ndarray,
                                              image_name: str, error: float) -> plt.Figure:
        """Generate landmark visualization exactly like original template_matching."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display image
        ax.imshow(image, cmap='gray')
        
        # Ground truth (lime green circles with dark green edges) - EXACTLY like original
        ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='lime', s=120, alpha=0.9,
                  label='Ground Truth', marker='o', edgecolors='darkgreen', linewidth=2)
        
        # Predictions (red X markers with thick lines) - EXACTLY like original
        ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=120, alpha=0.9,
                  label='Predicted', marker='x', linewidth=3)
        
        # Error lines (yellow connecting lines) - EXACTLY like original
        for j, (gt_pt, pred_pt) in enumerate(zip(gt_landmarks, pred_landmarks)):
            ax.plot([gt_pt[0], pred_pt[0]], [gt_pt[1], pred_pt[1]], 
                   'yellow', alpha=0.5, linewidth=1.5)
        
        # Add landmark numbers with black background - EXACTLY like original
        for j, (x, y) in enumerate(gt_landmarks):
            ax.annotate(f'{j}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Title and formatting exactly like original
        ax.set_title(f'{image_name} - Error: {error:.2f} px', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate_original_contour_visualization(self, idx: int, image: np.ndarray,
                                              pred_landmarks: np.ndarray, 
                                              image_name: str) -> plt.Figure:
        """Generate contour visualization exactly like original template_matching."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display image
        ax.imshow(image, cmap='gray')
        
        # Draw predicted contour - EXACTLY like original
        for start_idx, end_idx in self.contour_connections:
            start_pt = pred_landmarks[start_idx]
            end_pt = pred_landmarks[end_idx]
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                   'cyan', alpha=0.8, linewidth=3)
        
        # Draw midline connections - EXACTLY like original
        for start_idx, end_idx in self.midline_connections:
            start_pt = pred_landmarks[start_idx]
            end_pt = pred_landmarks[end_idx]
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                   'yellow', alpha=0.8, linewidth=2, linestyle='--')
        
        # Plot landmarks - EXACTLY like original
        ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=150, 
                  alpha=0.9, edgecolors='white', linewidth=2)
        
        # Title and formatting exactly like original
        ax.set_title(f'{image_name} - Lung Contour', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate_original_filename(self, idx: int, image_name: str, viz_type: str) -> str:
        """Generate filename exactly like original template_matching."""
        # Clean image name for filename (replace problematic characters)
        clean_name = image_name.replace('/', '_').replace(' ', '_')
        # Original format: {idx:03d}_{clean_name}_{type}.png
        return f"{idx:03d}_{clean_name}_{viz_type}.png"
    
    def generate_all_original_visualizations(self) -> Dict[str, int]:
        """Generate all 159 visualizations exactly like original template_matching."""
        num_images = len(self.results['image_names'])
        self.logger.info(f"Generating {num_images} original template_matching style visualizations...")
        
        # Clear existing files
        for dir_path in [self.landmarks_dir, self.contours_dir]:
            for file in dir_path.glob('*.png'):
                file.unlink()
        
        # Calculate errors for each image
        image_errors = []
        for pred, gt in zip(self.results['predictions'], self.results['ground_truth']):
            error_per_landmark = np.linalg.norm(pred - gt, axis=1)
            image_errors.append(np.mean(error_per_landmark))
        
        results = {
            'landmarks_success': 0,
            'landmarks_failed': 0,
            'contours_success': 0,
            'contours_failed': 0
        }
        
        # Process each image with progress bar
        for idx in tqdm(range(num_images), desc="Generating original template_matching style"):
            try:
                image_name = self.results['image_names'][idx]
                pred_landmarks = self.results['predictions'][idx]
                gt_landmarks = self.results['ground_truth'][idx]
                error = image_errors[idx]
                
                # Load image using original ASM utils
                img_path = asm_utils.get_image_path(image_name, None, str(self.images_base_dir))
                if not img_path or not os.path.exists(img_path):
                    self.logger.warning(f"Could not find image path for: {image_name}")
                    results['landmarks_failed'] += 1
                    results['contours_failed'] += 1
                    continue
                
                image = asm_utils.load_image_grayscale(img_path)
                if image is None:
                    self.logger.warning(f"Could not load image: {image_name}")
                    results['landmarks_failed'] += 1
                    results['contours_failed'] += 1
                    continue
                
                # Generate exact filenames like original
                landmark_filename = self.generate_original_filename(idx, image_name, 'landmarks')
                contour_filename = self.generate_original_filename(idx, image_name, 'contour')
                
                # Generate landmark visualization (with GT vs predictions)
                try:
                    fig = self.generate_original_landmark_visualization(
                        idx, image, pred_landmarks, gt_landmarks, image_name, error
                    )
                    output_path = self.landmarks_dir / landmark_filename
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    results['landmarks_success'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to generate landmark viz for {image_name}: {e}")
                    results['landmarks_failed'] += 1
                
                # Generate contour visualization
                try:
                    fig = self.generate_original_contour_visualization(
                        idx, image, pred_landmarks, image_name
                    )
                    output_path = self.contours_dir / contour_filename
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    results['contours_success'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to generate contour viz for {image_name}: {e}")
                    results['contours_failed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing image {idx}: {e}")
                results['landmarks_failed'] += 1
                results['contours_failed'] += 1
        
        # Generate summary statistics exactly like original
        self.generate_summary_report(image_errors)
        
        return results
    
    def generate_summary_report(self, image_errors: List[float]):
        """Generate summary report exactly like original template_matching."""
        summary_path = self.output_dir / 'summary_all_images.txt'
        
        with open(summary_path, 'w') as f:
            f.write("VISUALIZACIONES DE TODAS LAS IMÁGENES DE PRUEBA\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total de imágenes: {len(self.results['predictions'])}\n")
            f.write(f"Error promedio: {np.mean(image_errors):.2f} ± {np.std(image_errors):.2f} px\n")
            f.write(f"Error mediano: {np.median(image_errors):.2f} px\n")
            f.write(f"Error mínimo: {np.min(image_errors):.2f} px\n")
            f.write(f"Error máximo: {np.max(image_errors):.2f} px\n\n")
            
            # List all images with errors
            f.write("DETALLE POR IMAGEN:\n")
            f.write("-" * 50 + "\n")
            sorted_indices = np.argsort(image_errors)
            for i, idx in enumerate(sorted_indices):
                f.write(f"{i+1:3d}. {self.results['image_names'][idx]:30s} | {image_errors[idx]:6.2f} px\n")
        
        self.logger.info(f"Summary report saved to: {summary_path}")


def main():
    """Main execution function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Paths
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / 'data' / 'results_coordenadas_prueba_1.pkl'
    config_path = script_dir.parent / 'configs' / 'default_config.yaml'
    output_dir = script_dir.parent / 'visualizations' / 'original_template_matching_style'
    
    try:
        logger.info("Starting original template_matching style visualization generation...")
        
        # Initialize visualizer
        visualizer = OriginalTemplateMatchingVisualizer(
            str(results_path),
            str(config_path),
            str(output_dir)
        )
        
        # Generate all visualizations
        results = visualizer.generate_all_original_visualizations()
        
        # Print summary
        print("\n" + "="*80)
        print("ORIGINAL TEMPLATE MATCHING STYLE VISUALIZATION GENERATION COMPLETE")
        print("="*80)
        print(f"Landmark Predictions: {results['landmarks_success']} success, {results['landmarks_failed']} failed")
        print(f"Lung Contours: {results['contours_success']} success, {results['contours_failed']} failed")
        print("="*80)
        print(f"Output directory: {output_dir}")
        
        # Validate exact count
        landmark_count = len(list((output_dir / 'landmark_predictions').glob('*.png')))
        contour_count = len(list((output_dir / 'lung_contours').glob('*.png')))
        
        print(f"\nFinal file counts:")
        print(f"  Landmarks: {landmark_count} files")
        print(f"  Contours: {contour_count} files")
        
        if landmark_count == 159 and contour_count == 159:
            print("\n✅ SUCCESS: All 159 visualizations generated with EXACT original template_matching style!")
            print("   • Ground truth: lime green circles with dark green edges")
            print("   • Predictions: red X markers with thick lines")  
            print("   • Error lines: yellow connecting lines")
            print("   • Landmark numbers: white text with black background")
        else:
            print(f"\n⚠️ Warning: Expected 159 files each, got {landmark_count}, {contour_count}")
        
        logger.info("Original template_matching style visualization generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Original style visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())