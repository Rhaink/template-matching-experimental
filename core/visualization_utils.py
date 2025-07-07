#!/usr/bin/env python3
"""
Visualization utilities for template matching experimental platform.

This module provides functions for creating landmark and contour visualizations
that replicate the functionality of the original template_matching system.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_lung_connections() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Get anatomically correct lung landmark connections.
    
    Returns:
        Tuple of (contour_connections, mediastinal_connections)
    """
    # Anatomical contour connections (outer lung boundary)
    contour_connections = [
        (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
        (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
    ]
    
    # Mediastinal connections (center line)
    mediastinal_connections = [
        (0, 8), (8, 9), (9, 10), (10, 1)
    ]
    
    return contour_connections, mediastinal_connections


def get_pathology_colors() -> Dict[str, str]:
    """Get color mapping for different pathologies."""
    return {
        'Normal': '#00FF00',        # Green
        'COVID': '#FF0000',         # Red  
        'Viral Pneumonia': '#0000FF',  # Blue
        'default': '#FFFF00'        # Yellow
    }


def determine_pathology(image_name: str) -> str:
    """
    Determine pathology from image name.
    
    Args:
        image_name: Name of the image file
        
    Returns:
        Pathology type ('Normal', 'COVID', 'Viral Pneumonia', or 'Unknown')
    """
    if 'Normal' in image_name:
        return 'Normal'
    elif 'COVID' in image_name:
        return 'COVID'
    elif 'Viral' in image_name or 'Pneumonia' in image_name:
        return 'Viral Pneumonia'
    else:
        return 'Unknown'


def plot_landmarks_on_image(image: np.ndarray, 
                           landmarks: np.ndarray,
                           title: str = "",
                           show_connections: bool = True,
                           show_numbers: bool = True,
                           pathology: str = "Unknown") -> plt.Figure:
    """
    Plot landmarks on image with anatomical connections.
    
    Args:
        image: Grayscale image (H, W)
        landmarks: Landmark coordinates (15, 2)
        title: Plot title
        show_connections: Whether to show anatomical connections
        show_numbers: Whether to show landmark numbers
        pathology: Pathology type for color coding
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    ax.imshow(image, cmap='gray', aspect='equal')
    
    # Get connections and colors
    contour_connections, mediastinal_connections = get_lung_connections()
    colors = get_pathology_colors()
    landmark_color = colors.get(pathology, colors['default'])
    
    # Draw connections
    if show_connections:
        # Draw contour connections
        for start_idx, end_idx in contour_connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=landmark_color, linewidth=2, alpha=0.8)
        
        # Draw mediastinal connections  
        for start_idx, end_idx in mediastinal_connections:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            ax.plot([start_point[0], end_point[0]], 
                   [start_point[1], end_point[1]], 
                   color=landmark_color, linewidth=2, alpha=0.8, linestyle='--')
    
    # Draw landmark points
    ax.scatter(landmarks[:, 0], landmarks[:, 1], 
              c=landmark_color, s=100, alpha=0.9, edgecolors='black', linewidth=1)
    
    # Add landmark numbers
    if show_numbers:
        for i, (x, y) in enumerate(landmarks):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Formatting
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_lung_contours(image: np.ndarray,
                      landmarks: np.ndarray, 
                      title: str = "",
                      pathology: str = "Unknown") -> plt.Figure:
    """
    Plot lung contours without individual landmark points.
    
    Args:
        image: Grayscale image (H, W)
        landmarks: Landmark coordinates (15, 2)
        title: Plot title
        pathology: Pathology type for color coding
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    ax.imshow(image, cmap='gray', aspect='equal')
    
    # Get connections and colors
    contour_connections, mediastinal_connections = get_lung_connections()
    colors = get_pathology_colors()
    contour_color = colors.get(pathology, colors['default'])
    
    # Draw contour lines
    for start_idx, end_idx in contour_connections:
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        ax.plot([start_point[0], end_point[0]], 
               [start_point[1], end_point[1]], 
               color=contour_color, linewidth=3, alpha=0.9)
    
    # Draw mediastinal lines
    for start_idx, end_idx in mediastinal_connections:
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        ax.plot([start_point[0], end_point[0]], 
               [start_point[1], end_point[1]], 
               color=contour_color, linewidth=3, alpha=0.9, linestyle='--')
    
    # Formatting
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_side_by_side_comparison(image: np.ndarray,
                                ground_truth: np.ndarray,
                                prediction: np.ndarray,
                                image_name: str,
                                error: float) -> plt.Figure:
    """
    Create side-by-side comparison of ground truth vs prediction.
    
    Args:
        image: Grayscale image (H, W)
        ground_truth: Ground truth landmarks (15, 2)
        prediction: Predicted landmarks (15, 2)
        image_name: Name of the image
        error: Mean error in pixels
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    pathology = determine_pathology(image_name)
    
    # Ground truth plot
    ax1.imshow(image, cmap='gray', aspect='equal')
    contour_connections, mediastinal_connections = get_lung_connections()
    
    # Draw ground truth connections
    for start_idx, end_idx in contour_connections:
        start_point = ground_truth[start_idx]
        end_point = ground_truth[end_idx]
        ax1.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                color='green', linewidth=2, alpha=0.8)
    
    for start_idx, end_idx in mediastinal_connections:
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
    
    # Prediction plot
    ax2.imshow(image, cmap='gray', aspect='equal')
    
    # Draw prediction connections
    for start_idx, end_idx in contour_connections:
        start_point = prediction[start_idx]
        end_point = prediction[end_idx]
        ax2.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                color='red', linewidth=2, alpha=0.8)
    
    for start_idx, end_idx in mediastinal_connections:
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


def load_image_by_name(image_name: str, images_base_dir: str) -> Optional[np.ndarray]:
    """
    Load image by name from the dataset directories.
    
    Args:
        image_name: Name of the image file (with or without extension)
        images_base_dir: Base directory containing image subdirectories
        
    Returns:
        Loaded image or None if not found
    """
    # Image directories to search
    image_dirs = [
        Path(images_base_dir) / "COVID" / "images",
        Path(images_base_dir) / "Normal" / "images", 
        Path(images_base_dir) / "Viral Pneumonia" / "images"
    ]
    
    # Common image extensions to try
    extensions = ['', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    for img_dir in image_dirs:
        for ext in extensions:
            potential_path = img_dir / f"{image_name}{ext}"
            if potential_path.exists():
                image = cv2.imread(str(potential_path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    return image
                else:
                    logger.warning(f"Could not load image: {potential_path}")
    
    logger.error(f"Image not found in any directory: {image_name}")
    return None


def save_figure_safely(fig: plt.Figure, output_path: str, dpi: int = 150) -> bool:
    """
    Safely save figure to file with error handling.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Output file path
        dpi: Resolution for saving
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)  # Free memory
        return True
        
    except Exception as e:
        logger.error(f"Failed to save figure to {output_path}: {e}")
        plt.close(fig)  # Still close to free memory
        return False


def create_performance_summary(errors: np.ndarray, 
                             image_names: List[str],
                             output_path: str) -> bool:
    """
    Create performance summary visualization.
    
    Args:
        errors: Array of per-image errors
        image_names: List of image names
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error distribution histogram
        ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Error (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.3f}')
        ax1.legend()
        
        # Box plot by pathology
        pathologies = [determine_pathology(name) for name in image_names]
        unique_pathologies = list(set(pathologies))
        pathology_errors = [errors[np.array(pathologies) == p] for p in unique_pathologies]
        
        ax2.boxplot(pathology_errors, labels=unique_pathologies)
        ax2.set_ylabel('Error (pixels)')
        ax2.set_title('Error by Pathology')
        ax2.tick_params(axis='x', rotation=45)
        
        # Error over image index
        ax3.plot(errors, alpha=0.7, color='blue')
        ax3.set_xlabel('Image Index')
        ax3.set_ylabel('Error (pixels)')
        ax3.set_title('Error by Image Index')
        ax3.axhline(np.mean(errors), color='red', linestyle='--', alpha=0.7)
        
        # Top 10 worst errors
        worst_indices = np.argsort(errors)[-10:]
        worst_names = [image_names[i] for i in worst_indices]
        worst_errors = errors[worst_indices]
        
        ax4.barh(range(len(worst_names)), worst_errors, color='lightcoral')
        ax4.set_yticks(range(len(worst_names)))
        ax4.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                           for name in worst_names], fontsize=8)
        ax4.set_xlabel('Error (pixels)')
        ax4.set_title('Top 10 Worst Cases')
        
        plt.tight_layout()
        return save_figure_safely(fig, output_path)
        
    except Exception as e:
        logger.error(f"Failed to create performance summary: {e}")
        return False