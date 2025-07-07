#!/usr/bin/env python3
"""
Generate visualizations with the EXACT visual format from original template_matching.

This script creates visualizations in our experimental repository with the superior 
visual format that the user requested:
- Ground truth: lime green circles with dark green edges
- Predictions: red X markers with thick lines  
- Error lines: yellow connecting lines between GT and predictions
- Landmark numbers: white text with black background boxes

User request: "un nuevo scripts que haga las visualizaciones exactamente como lo usaba 
template_matching ya que tenia un mejor formato visual de mostrar los resultados"
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
EXPERIMENTAL_DIR = Path(__file__).parent.parent
PROJECT_ROOT_DIR = EXPERIMENTAL_DIR.parent / "Tesiscopia"
sys.path.insert(0, str(EXPERIMENTAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT_DIR / "pulmones" / "src"))

# Import ASM utils from Tesiscopia
try:
    from utils import asm_utils
    HAVE_ASM_UTILS = True
except ImportError:
    HAVE_ASM_UTILS = False
    print("Warning: ASM utils not available, using basic image loading")


def load_image_by_name(image_name: str, images_base_dir: str) -> np.ndarray:
    """Load image using available methods."""
    if HAVE_ASM_UTILS:
        # Use original ASM utils
        img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
        if img_path and os.path.exists(img_path):
            return asm_utils.load_image_grayscale(img_path)
    else:
        # Fallback: construct path manually
        # Handle different pathology folder formats
        pathology_map = {
            'normal': 'Normal',
            'covid': 'COVID',
            'viral pneumonia': 'Viral Pneumonia'
        }
        
        # Extract pathology from image name
        image_lower = image_name.lower()
        folder_name = None
        for key, value in pathology_map.items():
            if key in image_lower:
                folder_name = value
                break
        
        if folder_name:
            # Try different possible paths
            possible_paths = [
                os.path.join(images_base_dir, folder_name, 'images', f"{image_name}.png"),
                os.path.join(images_base_dir, folder_name, f"{image_name}.png"),
                os.path.join(images_base_dir, f"{image_name}.png")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    return None


def determine_pathology(image_name: str) -> str:
    """Determine pathology from image name."""
    image_lower = image_name.lower()
    if 'normal' in image_lower:
        return 'Normal'
    elif 'covid' in image_lower:
        return 'COVID'
    elif 'viral pneumonia' in image_lower or 'viral' in image_lower:
        return 'Viral Pneumonia'
    else:
        return 'Unknown'


def setup_logging():
    """Setup logging for original format generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(EXPERIMENTAL_DIR / 'original_format_generation.log'),
            logging.StreamHandler()
        ]
    )


class OriginalFormatVisualizer:
    """Generate visualizations with exact original template_matching format."""
    
    def __init__(self):
        """Initialize the original format visualizer."""
        self.experimental_dir = EXPERIMENTAL_DIR
        self.logger = logging.getLogger(__name__)
        
        # Load results from experimental repository
        self.results_path = self.experimental_dir / 'data' / 'results_coordenadas_prueba_1.pkl'
        with open(self.results_path, 'rb') as f:
            self.results = pickle.load(f)
        
        # Load config
        self.config_path = self.experimental_dir / 'configs' / 'default_config.yaml'
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get images base directory 
        project_root = Path(config['paths']['project_root'])
        self.images_base_dir = project_root / config['datasets']['image_base_path']
        
        # Setup output directories in experimental repository
        self.output_dir = self.experimental_dir / 'visualizations'
        self.landmarks_dir = self.output_dir / 'landmark_predictions'
        self.contours_dir = self.output_dir / 'lung_contours' 
        
        # Create directories
        for dir_path in [self.landmarks_dir, self.contours_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Original template_matching anatomical connections
        self.contour_connections = [
            (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
            (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
        ]
        self.midline_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
        
        self.logger.info(f"Initialized original format visualizer for {len(self.results['predictions'])} images")
    
    def generate_original_landmark_visualization(self, idx: int, image: np.ndarray, 
                                              pred_landmarks: np.ndarray, 
                                              gt_landmarks: np.ndarray,
                                              image_name: str, error: float) -> plt.Figure:
        """Generate landmark visualization with EXACT original template_matching format."""
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
        """Generate contour visualization with EXACT original template_matching format."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display image
        ax.imshow(image, cmap='gray')
        
        # Draw predicted contour exactly like original
        for start_idx, end_idx in self.contour_connections:
            start_pt = pred_landmarks[start_idx]
            end_pt = pred_landmarks[end_idx]
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                   'cyan', alpha=0.8, linewidth=3)
        
        # Draw midline connections exactly like original
        for start_idx, end_idx in self.midline_connections:
            start_pt = pred_landmarks[start_idx]
            end_pt = pred_landmarks[end_idx]
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                   'yellow', alpha=0.8, linewidth=2, linestyle='--')
        
        # Plot landmarks exactly like original
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
    
    def generate_all_original_format_visualizations(self) -> Dict[str, int]:
        """Generate all 159 visualizations with EXACT original template_matching format."""
        num_images = len(self.results['image_names'])
        self.logger.info(f"Generating {num_images} original template_matching format visualizations...")
        
        print("🎨 GENERANDO VISUALIZACIONES CON FORMATO ORIGINAL TEMPLATE MATCHING")
        print("=" * 70)
        print("Reemplazando visualizaciones en repositorio experimental con formato exacto:")
        print("• Ground Truth: círculos verde lima con bordes verde oscuro")
        print("• Predicciones: marcadores X rojos con líneas gruesas")
        print("• Líneas de error: líneas amarillas conectando GT y predicciones")
        print("• Números de landmarks: texto blanco con fondo negro")
        print("=" * 70)
        
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
        for idx in tqdm(range(num_images), desc="Generando formato original template_matching"):
            try:
                image_name = self.results['image_names'][idx]
                pred_landmarks = self.results['predictions'][idx]
                gt_landmarks = self.results['ground_truth'][idx]
                error = image_errors[idx]
                
                # Load image using experimental repository's loader
                image = load_image_by_name(image_name, str(self.images_base_dir))
                if image is None:
                    self.logger.warning(f"Could not load image: {image_name}")
                    results['landmarks_failed'] += 1
                    results['contours_failed'] += 1
                    continue
                
                # Generate exact filenames like original
                landmark_filename = self.generate_original_filename(idx, image_name, 'landmarks')
                contour_filename = self.generate_original_filename(idx, image_name, 'contour')
                
                # Generate landmark visualization with ORIGINAL FORMAT
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
                
                # Generate contour visualization with ORIGINAL FORMAT
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
        
        # Generate summary report
        self.generate_summary_report(image_errors)
        
        return results
    
    def generate_summary_report(self, image_errors: List[float]):
        """Generate summary report with original template_matching format."""
        summary_path = self.output_dir / 'summary_original_template_matching_format.txt'
        
        with open(summary_path, 'w') as f:
            f.write("VISUALIZACIONES CON FORMATO ORIGINAL TEMPLATE MATCHING\n")
            f.write("=" * 60 + "\n")
            f.write("Generadas en repositorio experimental con formato visual exacto:\n")
            f.write("• Ground Truth: círculos verde lima con bordes verde oscuro\n")
            f.write("• Predicciones: marcadores X rojos con líneas gruesas\n")
            f.write("• Líneas de error: líneas amarillas conectando GT y predicciones\n")
            f.write("• Números de landmarks: texto blanco con fondo negro\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total de imágenes: {len(self.results['predictions'])}\n")
            f.write(f"Error promedio: {np.mean(image_errors):.2f} ± {np.std(image_errors):.2f} px\n")
            f.write(f"Error mediano: {np.median(image_errors):.2f} px\n")
            f.write(f"Error mínimo: {np.min(image_errors):.2f} px\n")
            f.write(f"Error máximo: {np.max(image_errors):.2f} px\n\n")
            
            # List all images with errors
            f.write("DETALLE POR IMAGEN:\n")
            f.write("-" * 60 + "\n")
            sorted_indices = np.argsort(image_errors)
            for i, idx in enumerate(sorted_indices):
                if idx < len(self.results['image_names']):
                    f.write(f"{i+1:3d}. {self.results['image_names'][idx]:30s} | {image_errors[idx]:6.2f} px\n")
        
        self.logger.info(f"Summary report saved to: {summary_path}")


def main():
    """Main execution function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting original template_matching format visualization generation...")
        
        # Initialize visualizer
        visualizer = OriginalFormatVisualizer()
        
        # Generate all visualizations with original format
        results = visualizer.generate_all_original_format_visualizations()
        
        # Print summary
        print("\n" + "="*80)
        print("GENERACIÓN DE FORMATO ORIGINAL TEMPLATE MATCHING COMPLETADA")
        print("="*80)
        print(f"Landmark Predictions: {results['landmarks_success']} exitosos, {results['landmarks_failed']} fallidos")
        print(f"Lung Contours: {results['contours_success']} exitosos, {results['contours_failed']} fallidos")
        print("="*80)
        print(f"Directorio de salida: {EXPERIMENTAL_DIR / 'visualizations'}")
        
        # Validate exact count
        landmark_count = len(list((EXPERIMENTAL_DIR / 'visualizations' / 'landmark_predictions').glob('*.png')))
        contour_count = len(list((EXPERIMENTAL_DIR / 'visualizations' / 'lung_contours').glob('*.png')))
        
        print(f"\nArchivos finales:")
        print(f"  Landmarks: {landmark_count} archivos")
        print(f"  Contours: {contour_count} archivos")
        
        if landmark_count == 159 and contour_count == 159:
            print("\n🎉 ¡ÉXITO! 159 visualizaciones generadas con formato EXACTO template_matching original!")
            print("   ✅ Ground truth: círculos verde lima con bordes verde oscuro")
            print("   ✅ Predicciones: marcadores X rojos con líneas gruesas")  
            print("   ✅ Líneas de error: líneas amarillas conectando GT y predicciones")
            print("   ✅ Números de landmarks: texto blanco con fondo negro")
            print("   ✅ Generado en repositorio experimental correcto")
        else:
            print(f"\n⚠️ Advertencia: Se esperaban 159 archivos cada uno, se obtuvieron {landmark_count}, {contour_count}")
        
        logger.info("Original template_matching format visualization generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Original format visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())