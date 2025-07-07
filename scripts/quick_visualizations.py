#!/usr/bin/env python3
"""
Quick visualization script for rapid generation of all 159 visualizations.

This script is optimized for speed and generates visualizations with minimal overhead.
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
from typing import List, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for speed
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
EXPERIMENTAL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from core.visualization_utils import (
    get_lung_connections, determine_pathology, get_pathology_colors,
    load_image_by_name
)


class QuickVisualizationGenerator:
    """Optimized visualization generator for rapid processing."""
    
    def __init__(self, results_path: str, output_dir: str, config_path: str = None):
        """Initialize quick generator."""
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        
        # Load results
        with open(self.results_path, 'rb') as f:
            self.results = pickle.load(f)
        
        # Load minimal config if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Use project root from config
            project_root = Path(config['paths']['project_root'])
            self.images_base_dir = project_root / config['datasets']['image_base_path']
        else:
            self.images_base_dir = PROJECT_ROOT_DIR / "COVID-19_Radiography_Dataset"
        
        # Setup output directories
        self.landmarks_dir = self.output_dir / 'landmark_predictions'
        self.contours_dir = self.output_dir / 'lung_contours'
        
        for dir_path in [self.landmarks_dir, self.contours_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Pre-compute connections for speed
        self.contour_connections, self.mediastinal_connections = get_lung_connections()
        self.colors = get_pathology_colors()
        
        print(f"✅ Quick generator initialized for {len(self.results['predictions'])} images")
    
    def quick_landmark_plot(self, image: np.ndarray, landmarks: np.ndarray, 
                           pathology: str, title: str, output_path: str) -> bool:
        """Generate landmark plot with minimal overhead."""
        try:
            # Create figure with fixed size for speed
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            
            # Display image
            ax.imshow(image, cmap='gray', aspect='equal')
            
            # Get color
            landmark_color = self.colors.get(pathology, self.colors['default'])
            
            # Draw connections (vectorized for speed)
            for start_idx, end_idx in self.contour_connections:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       color=landmark_color, linewidth=2, alpha=0.8)
            
            for start_idx, end_idx in self.mediastinal_connections:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       color=landmark_color, linewidth=2, alpha=0.8, linestyle='--')
            
            # Draw landmarks
            ax.scatter(landmarks[:, 0], landmarks[:, 1], 
                      c=landmark_color, s=60, alpha=0.9, edgecolors='black', linewidth=1)
            
            # Minimal formatting for speed
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.set_title(title, fontsize=10, weight='bold')
            ax.axis('off')
            
            # Save and close immediately
            plt.tight_layout()
            fig.savefig(output_path, dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            plt.close('all')  # Ensure cleanup
            return False
    
    def quick_contour_plot(self, image: np.ndarray, landmarks: np.ndarray,
                          pathology: str, title: str, output_path: str) -> bool:
        """Generate contour plot with minimal overhead."""
        try:
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            
            # Display image
            ax.imshow(image, cmap='gray', aspect='equal')
            
            # Get color
            contour_color = self.colors.get(pathology, self.colors['default'])
            
            # Draw contour lines only
            for start_idx, end_idx in self.contour_connections:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       color=contour_color, linewidth=3, alpha=0.9)
            
            for start_idx, end_idx in self.mediastinal_connections:
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       color=contour_color, linewidth=3, alpha=0.9, linestyle='--')
            
            # Minimal formatting
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.set_title(title, fontsize=10, weight='bold')
            ax.axis('off')
            
            # Save and close
            plt.tight_layout()
            fig.savefig(output_path, dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return True
            
        except Exception as e:
            plt.close('all')
            return False
    
    def process_single_image(self, idx: int, generate_landmarks: bool = True,
                           generate_contours: bool = True) -> Tuple[bool, bool]:
        """Process single image for both landmark and contour visualizations."""
        try:
            image_name = self.results['image_names'][idx]
            prediction = self.results['predictions'][idx]
            ground_truth = self.results['ground_truth'][idx]
            
            # Calculate error
            error = np.mean(np.sqrt(np.sum((prediction - ground_truth)**2, axis=1)))
            pathology = determine_pathology(image_name)
            
            # Load image
            image = load_image_by_name(image_name, str(self.images_base_dir))
            if image is None:
                return False, False
            
            # Generate filename
            filename = f"{idx:03d}_{image_name.replace(' ', '_').replace('-', '_')}"
            
            landmark_success = True
            contour_success = True
            
            # Generate landmark visualization
            if generate_landmarks:
                title = f"{pathology} - Error: {error:.3f} px"
                output_path = self.landmarks_dir / f"{filename}_landmarks.png"
                landmark_success = self.quick_landmark_plot(
                    image, prediction, pathology, title, str(output_path)
                )
            
            # Generate contour visualization  
            if generate_contours:
                title = f"{pathology} - Contours"
                output_path = self.contours_dir / f"{filename}_contour.png"
                contour_success = self.quick_contour_plot(
                    image, prediction, pathology, title, str(output_path)
                )
            
            return landmark_success, contour_success
            
        except Exception as e:
            return False, False
    
    def generate_all_quick(self, resume: bool = False, 
                          landmarks: bool = True, 
                          contours: bool = True) -> Dict[str, int]:
        """Generate all visualizations with optimized speed."""
        num_images = len(self.results['image_names'])
        
        # Check existing files if resume mode
        existing_landmarks = set()
        existing_contours = set()
        
        if resume:
            existing_landmarks = {f.stem.replace('_landmarks', '') 
                                for f in self.landmarks_dir.glob('*_landmarks.png')}
            existing_contours = {f.stem.replace('_contour', '') 
                               for f in self.contours_dir.glob('*_contour.png')}
        
        # Counters
        landmark_success = 0
        landmark_failed = 0
        contour_success = 0
        contour_failed = 0
        skipped = 0
        
        print(f"🚀 Quick visualization generation for {num_images} images")
        print(f"   Landmarks: {'✅' if landmarks else '❌'}")
        print(f"   Contours: {'✅' if contours else '❌'}")
        print(f"   Resume mode: {'✅' if resume else '❌'}")
        
        # Process with progress bar
        start_time = datetime.now()
        
        for idx in tqdm(range(num_images), desc="Generating", ncols=80):
            image_name = self.results['image_names'][idx]
            filename = f"{idx:03d}_{image_name.replace(' ', '_').replace('-', '_')}"
            
            # Check if we should skip
            skip_landmarks = resume and landmarks and filename in existing_landmarks
            skip_contours = resume and contours and filename in existing_contours
            
            if skip_landmarks and skip_contours:
                skipped += 1
                continue
            
            # Process image
            landmark_ok, contour_ok = self.process_single_image(
                idx, 
                generate_landmarks=landmarks and not skip_landmarks,
                generate_contours=contours and not skip_contours
            )
            
            # Update counters
            if landmarks and not skip_landmarks:
                if landmark_ok:
                    landmark_success += 1
                else:
                    landmark_failed += 1
            
            if contours and not skip_contours:
                if contour_ok:
                    contour_success += 1
                else:
                    contour_failed += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Results summary
        results = {
            'landmark_success': landmark_success,
            'landmark_failed': landmark_failed,
            'contour_success': contour_success, 
            'contour_failed': contour_failed,
            'skipped': skipped,
            'duration': duration,
            'rate': num_images / duration if duration > 0 else 0
        }
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Quick visualization generation')
    parser.add_argument('--results', type=str,
                       default='data/results_coordenadas_prueba_1.pkl',
                       help='Path to results pickle file')
    parser.add_argument('--output', type=str,
                       default='visualizations',
                       help='Output directory')
    parser.add_argument('--config', type=str,
                       default='configs/default_config.yaml',
                       help='Configuration file (optional)')
    parser.add_argument('--resume', action='store_true',
                       help='Skip existing files')
    parser.add_argument('--landmarks-only', action='store_true',
                       help='Generate only landmark visualizations')
    parser.add_argument('--contours-only', action='store_true',
                       help='Generate only contour visualizations')
    
    args = parser.parse_args()
    
    try:
        # Convert paths
        script_dir = Path(__file__).parent
        results_path = script_dir.parent / args.results
        output_dir = script_dir.parent / args.output
        config_path = script_dir.parent / args.config if args.config else None
        
        # Validate inputs
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        # Determine what to generate
        generate_landmarks = not args.contours_only
        generate_contours = not args.landmarks_only
        
        # Initialize generator
        generator = QuickVisualizationGenerator(
            str(results_path),
            str(output_dir),
            str(config_path) if config_path else None
        )
        
        # Generate visualizations
        print(f"\nStarting quick generation...")
        results = generator.generate_all_quick(
            resume=args.resume,
            landmarks=generate_landmarks,
            contours=generate_contours
        )
        
        # Print results
        print("\n" + "="*60)
        print("QUICK VISUALIZATION GENERATION COMPLETE")
        print("="*60)
        if generate_landmarks:
            print(f"Landmarks: {results['landmark_success']:3d} success, "
                  f"{results['landmark_failed']:3d} failed")
        if generate_contours:
            print(f"Contours:  {results['contour_success']:3d} success, "
                  f"{results['contour_failed']:3d} failed")
        print(f"Skipped:   {results['skipped']:3d} (resume mode)")
        print(f"Duration:  {results['duration']:.1f} seconds")
        print(f"Rate:      {results['rate']:.1f} images/second")
        print("="*60)
        print(f"Output: {output_dir}")
        print("🎉 Quick generation complete!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Quick generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())