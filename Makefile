# Makefile for Matching Experimental Platform
# Provides automation for common development and research tasks

.PHONY: help install test clean lint format docs pipeline check validate

# Default target
help:
	@echo "Matching Experimental Platform - Development Automation"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install the package and dependencies"
	@echo "  install-dev - Install with development dependencies"
	@echo "  test        - Run the full test suite"
	@echo "  test-fast   - Run tests without slow integration tests"
	@echo "  test-cov    - Run tests with coverage report"
	@echo "  lint        - Run code linting (flake8, mypy)"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean build artifacts and cache"
	@echo "  docs        - Build documentation"
	@echo "  pipeline    - Run complete experimental pipeline"
	@echo "  validate    - Validate installation and configuration"
	@echo "  check       - Run all checks (lint, test, validate)"
	@echo ""
	@echo "Research workflow:"
	@echo "  make install-dev  # One-time setup"
	@echo "  make validate     # Verify installation"
	@echo "  make pipeline     # Run baseline experiments"
	@echo "  make test         # Validate results"

# Installation targets
install:
	@echo "Installing Matching Experimental Platform..."
	pip install -e .

install-dev:
	@echo "Installing with development dependencies..."
	pip install -e .[dev,notebooks]
	pre-commit install

# Testing targets
test:
	@echo "Running full test suite..."
	python -m pytest tests/ -v --tb=short

test-fast:
	@echo "Running fast tests (excluding integration)..."
	python -m pytest tests/ -v -m "not integration" --tb=short

test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=core --cov=scripts --cov-report=html --cov-report=term

test-parallel:
	@echo "Running tests in parallel..."
	python -m pytest tests/ -v -n auto --tb=short

# Code quality targets
lint:
	@echo "Running code linting..."
	flake8 core/ scripts/ tests/
	mypy core/ scripts/

format:
	@echo "Formatting code with black..."
	black core/ scripts/ tests/

format-check:
	@echo "Checking code formatting..."
	black --check core/ scripts/ tests/

# Documentation targets
docs:
	@echo "Building documentation..."
	cd docs && make html

docs-clean:
	@echo "Cleaning documentation build..."
	cd docs && make clean

# Research and experimental targets
pipeline:
	@echo "Running complete experimental pipeline..."
	@echo "Step 1: Training baseline model..."
	python scripts/train_experimental.py --config configs/default_config.yaml
	@echo "Step 2: Processing test images..."
	python scripts/process_experimental.py --config configs/default_config.yaml
	@echo "Step 3: Evaluating results..."
	python scripts/evaluate_experimental.py results/processing_results_*.pkl
	@echo "Pipeline completed successfully!"

pipeline-full:
	@echo "Running full pipeline with all configurations..."
	python scripts/run_full_pipeline.py --config configs/default_config.yaml

baseline:
	@echo "Validating baseline replication (5.63±0.17 px)..."
	python scripts/run_full_pipeline.py \
		--config configs/default_config.yaml \
		--output-dir results/baseline_validation \
		--validate-baseline

# Validation and checking targets
validate:
	@echo "Validating installation and configuration..."
	@echo "Checking Python version..."
	python --version
	@echo "Checking package installation..."
	python -c "import core.experimental_eigenpatches; print('✅ Core modules imported successfully')"
	@echo "Checking configuration files..."
	python -c "import yaml; yaml.safe_load(open('configs/default_config.yaml')); print('✅ Configuration files valid')"
	@echo "Checking data access..."
	python -c "import os; assert os.path.exists('../coordenadas/coordenadas_prueba_1.csv'), 'Missing test data'; print('✅ Test data accessible')"
	@echo "Validation completed successfully!"

check: format-check lint test-fast validate
	@echo "All checks passed! ✅"

# Cleaning targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-results:
	@echo "Cleaning experimental results..."
	rm -rf results/
	rm -rf models/
	rm -rf logs/

# Performance and profiling targets
profile:
	@echo "Profiling experimental pipeline..."
	python -m cProfile -o profile.stats scripts/run_full_pipeline.py --config configs/default_config.yaml
	python -c "import pstats; stats = pstats.Stats('profile.stats'); stats.sort_stats('cumulative').print_stats(20)"

memory-profile:
	@echo "Memory profiling experimental pipeline..."
	python -m memory_profiler scripts/run_full_pipeline.py --config configs/default_config.yaml

# Experimental targets
experiment-params:
	@echo "Running parameter sensitivity experiments..."
	python scripts/run_full_pipeline.py --config configs/experimental_configs.yaml --grid-search

experiment-custom:
	@echo "Running custom experiment configuration..."
	@read -p "Enter experiment name: " exp_name; \
	python scripts/run_full_pipeline.py \
		--config configs/experimental_configs.yaml \
		--experiment-name "$$exp_name" \
		--output-dir "results/$$exp_name"

# Notebook targets
notebooks:
	@echo "Starting Jupyter notebook server..."
	jupyter notebook notebooks/

notebooks-lab:
	@echo "Starting JupyterLab server..."
	jupyter lab notebooks/

notebooks-test:
	@echo "Testing all notebooks..."
	python scripts/test_notebooks.py notebooks/

# Distribution targets
package:
	@echo "Building distribution package..."
	python setup.py sdist bdist_wheel

package-test:
	@echo "Testing package installation..."
	python -m pip install --quiet --user --force-reinstall dist/*.whl

# Development convenience targets
dev-setup: install-dev validate
	@echo "Development environment setup completed!"
	@echo "Next steps:"
	@echo "  1. Run 'make pipeline' to test the baseline"
	@echo "  2. Open 'notebooks/00_QuickStart.ipynb' for interactive tutorial"
	@echo "  3. Review 'docs/Mathematical_Foundations.md' for theory"

dev-reset: clean clean-results
	@echo "Development environment reset completed!"

# Continuous integration targets
ci-test: format-check lint test-cov validate
	@echo "CI test suite completed!"

# Docker targets (if using containerization)
docker-build:
	@echo "Building Docker image..."
	docker build -t matching-experimental:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -it --rm -v $(PWD):/workspace matching-experimental:latest

# Help for specific targets
help-pipeline:
	@echo "Pipeline targets:"
	@echo "  pipeline      - Quick baseline validation"
	@echo "  pipeline-full - Complete experimental pipeline"
	@echo "  baseline      - Strict baseline replication validation"
	@echo ""
	@echo "Expected baseline performance: 5.63±0.17 pixels mean error"

help-experiments:
	@echo "Experimental targets:"
	@echo "  experiment-params - Parameter sensitivity analysis"
	@echo "  experiment-custom - Custom experiment with user input"
	@echo ""
	@echo "Configuration files:"
	@echo "  configs/default_config.yaml      - Baseline replication"
	@echo "  configs/experimental_configs.yaml - Parameter variations"

help-dev:
	@echo "Development targets:"
	@echo "  dev-setup  - One-time development environment setup"
	@echo "  dev-reset  - Clean slate for development"
	@echo "  check      - Run all quality checks"
	@echo ""
	@echo "Quality assurance:"
	@echo "  format     - Auto-format code"
	@echo "  lint       - Check code quality"
	@echo "  test       - Run test suite"
	@echo "  validate   - Verify installation"
