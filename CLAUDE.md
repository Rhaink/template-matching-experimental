# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **Template Matching Experimental Platform** - a comprehensive research platform for template matching landmark detection using eigenpatches, geometric constraints, and AI-assisted research capabilities. The platform replicates and enhances the baseline Template Matching with Eigenpatches algorithm that achieves 5.63±0.17 pixels mean error on 159 test images.

## Key Architecture

The codebase is organized into several key components:

- **`core/`** - Experimental adaptations of original algorithms with enhanced configuration and logging
- **`scripts/`** - Production-ready CLI tools for training, processing, and evaluation
- **`configs/`** - YAML configuration management for systematic experimentation
- **`tests/`** - Comprehensive test suite following TDD approach
- **`notebooks/`** - Interactive analysis and experimentation interfaces
- **`docs/`** - Mathematical foundations and research documentation

## Essential Commands

### Development Setup
```bash
# Install dependencies and set up development environment
make install-dev

# Validate installation
make validate
```

### Testing
```bash
# Run full test suite
make test

# Run fast tests (excluding slow integration tests)
make test-fast

# Run tests with coverage
make test-cov
```

### Code Quality
```bash
# Check code quality (linting, formatting, tests)
make check

# Format code
make format

# Run linting
make lint
```

### Pipeline Operations
```bash
# Run complete experimental pipeline
make pipeline

# Run full pipeline with comprehensive analysis
make pipeline-full

# Validate baseline replication (5.63±0.17 px target)
make baseline
```

### Visualization Generation
```bash
# Generate all 159 visualizations (comprehensive)
make visualize

# Quick generation of all visualizations
make visualize-quick

# Generate only landmark visualizations
make visualize-landmarks

# Generate only contour visualizations
make visualize-contours
```

### Individual Script Usage
```bash
# Train model with baseline configuration
python scripts/train_experimental.py --config configs/default_config.yaml

# Process test images
python scripts/process_experimental.py --config configs/default_config.yaml

# Evaluate results
python scripts/evaluate_experimental.py results/processing_results_*.pkl

# Run complete automated pipeline
python scripts/run_full_pipeline.py --config configs/default_config.yaml
```

## Configuration System

The platform uses YAML-based configuration management:

- **`configs/default_config.yaml`** - Baseline replication parameters that achieve 5.63±0.17 px error
- **`configs/experimental_configs.yaml`** - Parameter variations for systematic experimentation

Critical configuration parameters:
- `eigenpatches.patch_size: 21` - Size of patches extracted around landmarks
- `eigenpatches.n_components: 20` - Number of PCA components
- `landmark_predictor.lambda_shape: 0.1` - Shape model constraint weight
- `image_processing.coordinate_scale_factor: 4.67` - Essential scaling from 64x64 to 299x299 coordinates

## Core Architecture Details

### Key Classes
- **`ExperimentalEigenpatches`** (`core/experimental_eigenpatches.py`) - Enhanced eigenpatches model with configuration support
- **`ExperimentalLandmarkPredictor`** (`core/experimental_predictor.py`) - Template landmark predictor with convergence analysis
- **`ExperimentalEvaluator`** (`core/experimental_evaluator.py`) - Comprehensive evaluation with HTML reports

### Critical Algorithm Components
1. **Eigenpatches (PCA-based Template Matching)** - Uses PCA reconstruction error for landmark detection
2. **Statistical Shape Models** - Geometric constraints with 3σ bounds on shape parameters
3. **Multi-scale Optimization** - Coarse-to-fine search using Gaussian pyramids
4. **Coordinate Scaling** - Essential 4.67x scaling factor from reference to target resolution

## Performance Expectations

The platform should replicate these baseline results:
- **Mean Error**: 5.63±0.17 pixels on 159 test images
- **Processing Speed**: ~0.2 seconds per image
- **Training Time**: ~60 seconds for 640 training images
- **Visualization Generation**: ~2.7 images/second (57 seconds total for 159 images)

## Testing Strategy

The codebase follows Test-Driven Development (TDD):
- **Unit Tests** - `test_eigenpatches.py`, `test_predictor.py`, `test_evaluator.py`
- **Integration Tests** - `test_integration.py` for end-to-end pipeline validation
- **Fixtures** - Synthetic test data generators in `tests/fixtures/`

## Common Development Tasks

### Adding New Experiments
1. Create new configuration in `configs/experimental_configs.yaml`
2. Use `scripts/run_full_pipeline.py` with custom config
3. Analyze results using notebooks in `notebooks/`

### Debugging Pipeline Issues
1. Check configuration validity with `make validate`
2. Run with debug logging: set `logging.level: "DEBUG"` in config
3. Use `make test-fast` to validate core functionality

### Performance Optimization
```bash
# Profile pipeline performance
make profile

# Memory profiling
make memory-profile
```

## File Structure Context

- **Models** are saved in `models/` directory as `.pkl` files
- **Results** are stored in `results/` with detailed per-image analysis
- **Visualizations** are generated in `visualizations/` with subdirectories for different types
- **Logs** are written to `logs/` when file logging is enabled

## Development Environment

- **Python 3.8+** required
- **Key Dependencies**: numpy, opencv-python, scikit-learn, matplotlib, PyYAML
- **Development Tools**: pytest, black, flake8, mypy, pre-commit
- **Optional**: CUDA support for GPU acceleration

## Critical Notes

1. **Coordinate Scaling**: The 4.67x scaling factor is essential - incorrect scaling breaks the entire pipeline
2. **Configuration Compatibility**: Always validate configuration changes against baseline performance
3. **Test Coverage**: Maintain comprehensive test coverage for all core functionality
4. **Documentation**: Mathematical foundations are documented in `docs/Mathematical_Foundations.md`

## Integration Points

The experimental platform integrates with the original template matching codebase through:
- **Path Management**: Automatic detection of `template_matching/src` directory
- **API Compatibility**: Drop-in replacement for original classes
- **Adapter Pattern**: Experimental classes wrap original implementations