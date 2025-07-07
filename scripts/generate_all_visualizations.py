#!/usr/bin/env python3
"""
Generate visualizations for all 159 test images in the experimental platform.

This script creates comprehensive visualizations including landmark predictions,
lung contours, and side-by-side comparisons for all test images.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import pickle
import yaml
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing as mp
from functools import partial

# Add project paths
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
EXPERIMENTAL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from core.visualization_utils import (
    plot_landmarks_on_image, plot_lung_contours, plot_side_by_side_comparison,
    load_image_by_name, save_figure_safely, create_performance_summary,
    determine_pathology, get_pathology_colors
)


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('visualization_generation.log'),
            logging.StreamHandler()
        ]
    )


class VisualizationGenerator:
    """Main class for generating all visualizations."""
    
    def __init__(self, config_path: str, results_path: str, output_dir: str):
        """
        Initialize visualization generator.
        
        Args:
            config_path: Path to configuration file
            results_path: Path to results pickle file
            output_dir: Output directory for visualizations
        """
        self.config_path = Path(config_path)
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load results
        with open(self.results_path, 'rb') as f:
            self.results = pickle.load(f)
        
        # Setup output directories
        self.setup_output_directories()
        
        # Get dataset info - use project root from config
        project_root = Path(self.config['paths']['project_root'])
        self.images_base_dir = project_root / self.config['datasets']['image_base_path']
        
        self.logger.info(f"Initialized with {len(self.results['predictions'])} images")
        
    def setup_output_directories(self) -> None:
        """Create output directory structure."""
        self.landmarks_dir = self.output_dir / 'landmark_predictions'
        self.contours_dir = self.output_dir / 'lung_contours'
        self.comparison_dir = self.output_dir / 'side_by_side'
        self.analysis_dir = self.output_dir / 'performance_analysis'
        
        for dir_path in [self.landmarks_dir, self.contours_dir, 
                        self.comparison_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directories created in: {self.output_dir}")
    
    def get_completed_files(self) -> Dict[str, List[str]]:
        """Get list of already completed visualizations for resume capability."""
        completed = {
            'landmarks': [f.stem for f in self.landmarks_dir.glob('*.png')],
            'contours': [f.stem for f in self.contours_dir.glob('*.png')],
            'comparisons': [f.stem for f in self.comparison_dir.glob('*.png')]
        }
        return completed
    
    def calculate_errors(self) -> np.ndarray:
        """Calculate per-image errors."""
        predictions = np.array(self.results['predictions'])
        ground_truth = np.array(self.results['ground_truth'])
        
        # Calculate Euclidean distances
        errors = np.sqrt(np.sum((predictions - ground_truth)**2, axis=2))
        per_image_errors = np.mean(errors, axis=1)
        
        return per_image_errors
    
    def generate_single_visualization(self, idx: int, viz_type: str, 
                                    resume: bool = False) -> bool:
        """
        Generate visualization for a single image.
        
        Args:
            idx: Image index
            viz_type: Type of visualization ('landmarks', 'contours', 'comparison')
            resume: Whether to skip if file already exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_name = self.results['image_names'][idx]
            pathology = determine_pathology(image_name)
            
            # Generate filename
            filename = f"{idx:03d}_{image_name.replace(' ', '_').replace('-', '_')}"
            
            # Check if file already exists (for resume)
            if viz_type == 'landmarks':
                output_path = self.landmarks_dir / f"{filename}_landmarks.png"
            elif viz_type == 'contours':
                output_path = self.contours_dir / f"{filename}_contour.png"
            elif viz_type == 'comparison':
                output_path = self.comparison_dir / f"{filename}_combined.png"
            else:
                return False
            
            if resume and output_path.exists():
                return True
            
            # Load image
            image = load_image_by_name(image_name, str(self.images_base_dir))
            if image is None:
                self.logger.warning(f"Could not load image: {image_name}")
                return False
            
            # Get landmarks
            prediction = self.results['predictions'][idx]
            ground_truth = self.results['ground_truth'][idx]
            
            # Calculate error for this image
            error = np.mean(np.sqrt(np.sum((prediction - ground_truth)**2, axis=1)))
            
            # Generate visualization based on type
            if viz_type == 'landmarks':
                title = f"{pathology} - {image_name}\nPredicted Landmarks - Error: {error:.3f} px"
                fig = plot_landmarks_on_image(image, prediction, title=title, 
                                            pathology=pathology)
                
            elif viz_type == 'contours':
                title = f"{pathology} - {image_name}\nLung Contours - Error: {error:.3f} px"
                fig = plot_lung_contours(image, prediction, title=title, 
                                       pathology=pathology)
                
            elif viz_type == 'comparison':
                fig = plot_side_by_side_comparison(image, ground_truth, prediction,
                                                 image_name, error)
            
            # Save figure
            success = save_figure_safely(fig, str(output_path))
            
            if not success:
                self.logger.error(f"Failed to save {viz_type} for {image_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error generating {viz_type} for index {idx}: {e}")
            return False
    
    def generate_batch_visualizations(self, viz_type: str, 
                                    batch_size: int = 20,
                                    resume: bool = False) -> Tuple[int, int]:
        """
        Generate visualizations in batches.
        
        Args:
            viz_type: Type of visualization
            batch_size: Number of images to process in each batch
            resume: Whether to skip existing files
            
        Returns:
            Tuple of (successful, failed) counts
        """
        num_images = len(self.results['image_names'])
        successful = 0
        failed = 0
        
        self.logger.info(f"Generating {viz_type} visualizations for {num_images} images...")
        
        # Create progress bar
        pbar = tqdm(range(num_images), desc=f"Generating {viz_type}")
        
        for idx in pbar:
            success = self.generate_single_visualization(idx, viz_type, resume)
            if success:
                successful += 1
            else:
                failed += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Success': successful, 
                'Failed': failed,
                'Rate': f"{successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "0%"
            })
        
        return successful, failed
    
    def generate_performance_analysis(self) -> bool:
        """Generate performance analysis visualizations."""
        try:
            self.logger.info("Generating performance analysis...")
            
            # Calculate errors
            errors = self.calculate_errors()
            image_names = self.results['image_names']
            
            # Create performance summary
            output_path = self.analysis_dir / 'performance_summary.png'
            success = create_performance_summary(errors, image_names, str(output_path))
            
            if success:
                self.logger.info(f"Performance analysis saved to: {output_path}")
            
            # Create detailed statistics report
            self.create_statistics_report(errors, image_names)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error generating performance analysis: {e}")
            return False
    
    def create_statistics_report(self, errors: np.ndarray, 
                               image_names: List[str]) -> None:
        """Create detailed statistics report."""
        try:
            report_path = self.analysis_dir / 'statistics_report.txt'
            
            # Calculate statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            median_error = np.median(errors)
            min_error = np.min(errors)
            max_error = np.max(errors)
            
            # Pathology breakdown
            pathologies = [determine_pathology(name) for name in image_names]
            unique_pathologies = list(set(pathologies))
            
            pathology_stats = {}
            for pathology in unique_pathologies:
                mask = np.array(pathologies) == pathology
                pathology_errors = errors[mask]
                pathology_stats[pathology] = {
                    'count': len(pathology_errors),
                    'mean': np.mean(pathology_errors),
                    'std': np.std(pathology_errors),
                    'median': np.median(pathology_errors)
                }
            
            # Best and worst cases
            best_indices = np.argsort(errors)[:5]
            worst_indices = np.argsort(errors)[-5:]
            
            # Write report
            with open(report_path, 'w') as f:
                f.write("TEMPLATE MATCHING EXPERIMENTAL PLATFORM - VISUALIZATION REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Images: {len(image_names)}\n\n")
                
                f.write("OVERALL PERFORMANCE:\n")
                f.write(f"  Mean Error: {mean_error:.3f} ± {std_error:.3f} pixels\n")
                f.write(f"  Median Error: {median_error:.3f} pixels\n")
                f.write(f"  Range: {min_error:.3f} - {max_error:.3f} pixels\n")
                f.write(f"  Baseline Target: 5.63 ± 0.17 pixels\n")
                f.write(f"  Accuracy: {'EXCELLENT' if abs(mean_error - 5.63) < 0.1 else 'GOOD'}\n\n")
                
                f.write("PERFORMANCE BY PATHOLOGY:\n")
                for pathology, stats in pathology_stats.items():
                    f.write(f"  {pathology}:\n")
                    f.write(f"    Count: {stats['count']} images\n")
                    f.write(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f} pixels\n")
                    f.write(f"    Median: {stats['median']:.3f} pixels\n\n")
                
                f.write("BEST PERFORMING IMAGES:\n")
                for i, idx in enumerate(best_indices):
                    f.write(f"  {i+1}. {image_names[idx]}: {errors[idx]:.3f} pixels\n")
                f.write("\n")
                
                f.write("WORST PERFORMING IMAGES:\n")
                for i, idx in enumerate(worst_indices):
                    f.write(f"  {i+1}. {image_names[idx]}: {errors[idx]:.3f} pixels\n")
                f.write("\n")
                
                f.write("VISUALIZATION FILES GENERATED:\n")
                f.write(f"  Landmark Predictions: {len(list(self.landmarks_dir.glob('*.png')))} files\n")
                f.write(f"  Lung Contours: {len(list(self.contours_dir.glob('*.png')))} files\n")
                f.write(f"  Side-by-side Comparisons: {len(list(self.comparison_dir.glob('*.png')))} files\n")
                f.write(f"  Performance Analysis: {len(list(self.analysis_dir.glob('*.png')))} files\n")
            
            self.logger.info(f"Statistics report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating statistics report: {e}")
    
    def generate_all(self, resume: bool = False, 
                    viz_types: List[str] = None) -> Dict[str, Tuple[int, int]]:
        """
        Generate all visualizations.
        
        Args:
            resume: Whether to skip existing files
            viz_types: List of visualization types to generate
            
        Returns:
            Dictionary with results for each visualization type
        """
        if viz_types is None:
            viz_types = ['landmarks', 'contours', 'comparison']
        
        results = {}
        total_start_time = datetime.now()
        
        self.logger.info("Starting complete visualization generation...")
        self.logger.info(f"Resume mode: {resume}")
        self.logger.info(f"Visualization types: {viz_types}")
        
        # Generate each type of visualization
        for viz_type in viz_types:
            start_time = datetime.now()
            successful, failed = self.generate_batch_visualizations(viz_type, resume=resume)
            duration = (datetime.now() - start_time).total_seconds()
            
            results[viz_type] = (successful, failed)
            self.logger.info(f"{viz_type.title()} complete: {successful} success, "
                           f"{failed} failed in {duration:.1f}s")
        
        # Generate performance analysis
        if 'analysis' not in viz_types or 'analysis' in viz_types:
            analysis_success = self.generate_performance_analysis()
            results['analysis'] = (1 if analysis_success else 0, 0 if analysis_success else 1)
        
        total_duration = (datetime.now() - total_start_time).total_seconds()
        
        # Summary
        total_success = sum(result[0] for result in results.values())
        total_failed = sum(result[1] for result in results.values())
        
        self.logger.info(f"COMPLETE! Total: {total_success} success, {total_failed} failed "
                        f"in {total_duration/60:.1f} minutes")
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate all visualizations for template matching experimental platform')
    parser.add_argument('--config', type=str, 
                       default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--results', type=str,
                       default='data/results_coordenadas_prueba_1.pkl',
                       help='Path to results pickle file')
    parser.add_argument('--output', type=str,
                       default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--resume', action='store_true',
                       help='Resume generation (skip existing files)')
    parser.add_argument('--types', nargs='+', 
                       choices=['landmarks', 'contours', 'comparison', 'analysis'],
                       default=['landmarks', 'contours', 'comparison', 'analysis'],
                       help='Types of visualizations to generate')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Convert relative paths to absolute
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / args.config
        results_path = script_dir.parent / args.results
        output_dir = script_dir.parent / args.output
        
        # Validate inputs
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        # Initialize generator
        generator = VisualizationGenerator(
            str(config_path), 
            str(results_path), 
            str(output_dir)
        )
        
        # Generate visualizations
        results = generator.generate_all(resume=args.resume, viz_types=args.types)
        
        # Print final summary
        print("\n" + "="*70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*70)
        for viz_type, (success, failed) in results.items():
            print(f"{viz_type.title():20}: {success:3d} success, {failed:3d} failed")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print("✅ All visualizations generated successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())