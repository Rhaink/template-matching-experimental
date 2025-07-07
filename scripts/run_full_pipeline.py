#!/usr/bin/env python3
"""
Master pipeline script for experimental template matching system.

This script executes the complete experimental pipeline: training → processing → evaluation
with comprehensive error handling, checkpointing, and reporting capabilities.
"""

import sys
import os
import subprocess
from pathlib import Path
import yaml
import argparse
import logging
from datetime import datetime
import time
import json
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
MATCHING_EXPERIMENTAL_DIR = PROJECT_ROOT_DIR / "matching_experimental"
sys.path.insert(0, str(MATCHING_EXPERIMENTAL_DIR))


class PipelineExecutor:
    """
    Manages the complete experimental pipeline execution.
    """
    
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """
        Initialize pipeline executor.
        
        Args:
            config_path: Path to configuration file
            output_dir: Optional output directory override
        """
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT_DIR / self.config_path
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.output_dir = Path(output_dir) if output_dir else Path("pipeline_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'steps_completed': [],
            'steps_failed': [],
            'current_step': None,
            'artifacts': {}
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for pipeline execution."""
        # Create logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_file = logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline logging initialized: {log_file}")
    
    def save_checkpoint(self):
        """Save pipeline state checkpoint."""
        checkpoint_file = self.output_dir / "pipeline_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
        self.logger.debug(f"Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self) -> bool:
        """
        Load pipeline state from checkpoint.
        
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        checkpoint_file = self.output_dir / "pipeline_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                self.pipeline_state = json.load(f)
            self.logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return True
        return False
    
    def run_step(self, step_name: str, command: list, 
                 required_artifacts: list = None,
                 output_artifacts: list = None) -> bool:
        """
        Execute a single pipeline step.
        
        Args:
            step_name: Name of the step
            command: Command to execute as list
            required_artifacts: List of required artifact paths
            output_artifacts: List of expected output artifact paths
            
        Returns:
            True if step succeeded, False otherwise
        """
        self.logger.info(f"Starting step: {step_name}")
        self.pipeline_state['current_step'] = step_name
        self.save_checkpoint()
        
        # Check required artifacts
        if required_artifacts:
            for artifact in required_artifacts:
                if not Path(artifact).exists():
                    self.logger.error(f"Required artifact missing: {artifact}")
                    self.pipeline_state['steps_failed'].append(step_name)
                    return False
        
        # Execute command
        start_time = time.time()
        try:
            self.logger.info(f"Executing: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT_DIR,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"Step '{step_name}' completed successfully in {duration:.2f}s")
                
                # Verify output artifacts
                if output_artifacts:
                    missing_artifacts = []
                    for artifact in output_artifacts:
                        if not Path(artifact).exists():
                            missing_artifacts.append(artifact)
                    
                    if missing_artifacts:
                        self.logger.warning(f"Expected artifacts missing: {missing_artifacts}")
                    else:
                        self.logger.info(f"All expected artifacts created: {output_artifacts}")
                        # Store successful artifacts
                        for artifact in output_artifacts:
                            self.pipeline_state['artifacts'][step_name] = {
                                'outputs': output_artifacts,
                                'duration': duration,
                                'timestamp': datetime.now().isoformat()
                            }
                
                self.pipeline_state['steps_completed'].append(step_name)
                return True
            else:
                self.logger.error(f"Step '{step_name}' failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                self.pipeline_state['steps_failed'].append(step_name)
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Step '{step_name}' timed out after 1 hour")
            self.pipeline_state['steps_failed'].append(step_name)
            return False
        except Exception as e:
            self.logger.error(f"Step '{step_name}' failed with exception: {e}")
            self.pipeline_state['steps_failed'].append(step_name)
            return False
        finally:
            self.save_checkpoint()
    
    def estimate_total_time(self) -> Dict[str, float]:
        """
        Estimate total pipeline execution time.
        
        Returns:
            Dictionary with time estimates
        """
        # Based on typical dataset sizes and processing times
        dataset_size = self.config.get('dataset_info', {}).get('estimated_training_images', 640)
        test_size = self.config.get('dataset_info', {}).get('estimated_test_images', 159)
        
        estimates = {
            'training': max(60, dataset_size * 0.1),  # ~0.1 seconds per training image, min 1 minute
            'processing': max(30, test_size * 0.2),   # ~0.2 seconds per test image, min 30 seconds
            'evaluation': max(20, test_size * 0.1),   # ~0.1 seconds per test image, min 20 seconds
            'overhead': 30  # File I/O, initialization, etc.
        }
        
        estimates['total'] = sum(estimates.values())
        return estimates
    
    def run_training_step(self) -> bool:
        """Execute training step."""
        model_output = self.output_dir / "trained_model.pkl"
        
        command = [
            "python", "matching_experimental/scripts/train_experimental.py",
            "--config", str(self.config_path),
            "--output", str(model_output)
        ]
        
        return self.run_step(
            "training",
            command,
            required_artifacts=[],
            output_artifacts=[str(model_output)]
        )
    
    def run_processing_step(self) -> bool:
        """Execute processing step."""
        model_path = self.output_dir / "trained_model.pkl"
        results_output = self.output_dir / "processing_results.pkl"
        
        command = [
            "python", "matching_experimental/scripts/process_experimental.py",
            "--config", str(self.config_path),
            "--model", str(model_path),
            "--output", str(results_output)
        ]
        
        return self.run_step(
            "processing",
            command,
            required_artifacts=[str(model_path)],
            output_artifacts=[str(results_output)]
        )
    
    def run_evaluation_step(self) -> bool:
        """Execute evaluation step."""
        results_path = self.output_dir / "processing_results.pkl"
        eval_output = self.output_dir / "evaluation_results.pkl"
        
        # Add ground truth if available
        command = [
            "python", "matching_experimental/scripts/evaluate_experimental.py",
            str(results_path),
            "--config", str(self.config_path),
            "--output", str(eval_output)
        ]
        
        # Try to provide ground truth coordinates
        gt_coords = self.config.get('datasets', {}).get('test_coords')
        if gt_coords:
            gt_path = PROJECT_ROOT_DIR / gt_coords
            if gt_path.exists():
                command.extend(["--ground-truth", str(gt_path)])
        
        return self.run_step(
            "evaluation",
            command,
            required_artifacts=[str(results_path)],
            output_artifacts=[str(eval_output)]
        )
    
    def generate_final_report(self) -> bool:
        """Generate final pipeline report."""
        self.logger.info("Generating final pipeline report...")
        
        try:
            # Load evaluation results if available
            eval_results = None
            eval_path = self.output_dir / "evaluation_results.pkl"
            if eval_path.exists():
                import pickle
                with open(eval_path, 'rb') as f:
                    eval_results = pickle.load(f)
            
            # Generate HTML report
            report_path = self.output_dir / "pipeline_report.html"
            self._generate_html_report(eval_results, report_path)
            
            # Generate summary YAML
            summary_path = self.output_dir / "pipeline_summary.yaml"
            self._generate_summary_yaml(eval_results, summary_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
            return False
    
    def _generate_html_report(self, eval_results: Optional[Dict], output_path: Path):
        """Generate HTML pipeline report."""
        
        # Calculate total pipeline time
        if self.pipeline_state['start_time'] and self.pipeline_state['end_time']:
            total_time = (datetime.fromisoformat(self.pipeline_state['end_time']) - 
                         datetime.fromisoformat(self.pipeline_state['start_time'])).total_seconds()
        else:
            total_time = 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experimental Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; font-weight: bold; }}
                .metric {{ margin: 10px 0; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experimental Template Matching Pipeline Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Configuration: {self.config_path.name}</p>
                <p>Total Execution Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)</p>
            </div>
            
            <div class="section">
                <h2>Pipeline Execution Summary</h2>
                <div class="metric">Steps Completed: <span class="success">{len(self.pipeline_state['steps_completed'])}</span></div>
                <div class="metric">Steps Failed: <span class="failure">{len(self.pipeline_state['steps_failed'])}</span></div>
                <div class="metric">Start Time: {self.pipeline_state.get('start_time', 'N/A')}</div>
                <div class="metric">End Time: {self.pipeline_state.get('end_time', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>Step Details</h2>
                <table class="table">
                    <tr><th>Step</th><th>Status</th><th>Duration</th><th>Artifacts</th></tr>
        """
        
        all_steps = ['training', 'processing', 'evaluation']
        for step in all_steps:
            if step in self.pipeline_state['steps_completed']:
                status = '<span class="success">SUCCESS</span>'
                duration = self.pipeline_state['artifacts'].get(step, {}).get('duration', 'N/A')
                artifacts = len(self.pipeline_state['artifacts'].get(step, {}).get('outputs', []))
            elif step in self.pipeline_state['steps_failed']:
                status = '<span class="failure">FAILED</span>'
                duration = 'N/A'
                artifacts = 0
            else:
                status = 'NOT RUN'
                duration = 'N/A'
                artifacts = 0
            
            html_content += f"<tr><td>{step.title()}</td><td>{status}</td><td>{duration}</td><td>{artifacts}</td></tr>"
        
        html_content += """
                </table>
            </div>
        """
        
        # Add evaluation results if available
        if eval_results:
            exec_summary = eval_results.get('executive_summary', {}).get('executive_summary', {})
            baseline_comp = eval_results.get('baseline_comparison', {})
            
            html_content += f"""
            <div class="section">
                <h2>Performance Results</h2>
                <div class="metric">Method: {exec_summary.get('method_name', 'N/A')}</div>
                <div class="metric">Dataset Size: {exec_summary.get('dataset_size', 'N/A')} images</div>
                <div class="metric">Mean Error: {exec_summary.get('key_results', {}).get('mean_error', 'N/A')}</div>
                <div class="metric">Baseline Comparison: {exec_summary.get('key_results', {}).get('baseline_comparison', 'N/A')}</div>
                <div class="metric">Improvement: {exec_summary.get('key_results', {}).get('improvement_vs_baseline', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>Configuration Used</h2>
                <div class="metric">Patch Size: {exec_summary.get('configuration_used', {}).get('patch_size', 'N/A')}</div>
                <div class="metric">PCA Components: {exec_summary.get('configuration_used', {}).get('n_components', 'N/A')}</div>
                <div class="metric">Pyramid Levels: {exec_summary.get('configuration_used', {}).get('pyramid_levels', 'N/A')}</div>
                <div class="metric">Lambda Shape: {exec_summary.get('configuration_used', {}).get('lambda_shape', 'N/A')}</div>
            </div>
            """
        
        html_content += """
            <div class="section">
                <h2>Output Files</h2>
                <ul>
                    <li><strong>Trained Model:</strong> trained_model.pkl</li>
                    <li><strong>Processing Results:</strong> processing_results.pkl</li>
                    <li><strong>Evaluation Results:</strong> evaluation_results.pkl</li>
                    <li><strong>Pipeline Log:</strong> logs/pipeline_*.log</li>
                    <li><strong>Summary:</strong> pipeline_summary.yaml</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Next Steps</h2>
                <ul>
                    <li>Review evaluation results for detailed performance analysis</li>
                    <li>Compare with other experimental configurations</li>
                    <li>Consider parameter optimization based on results</li>
                    <li>Run additional experiments with modified configurations</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")
    
    def _generate_summary_yaml(self, eval_results: Optional[Dict], output_path: Path):
        """Generate YAML summary of pipeline execution."""
        
        summary = {
            'pipeline_execution': {
                'start_time': self.pipeline_state.get('start_time'),
                'end_time': self.pipeline_state.get('end_time'),
                'total_duration_seconds': self.pipeline_state.get('total_duration'),
                'configuration_file': str(self.config_path),
                'output_directory': str(self.output_dir),
                'steps_completed': self.pipeline_state['steps_completed'],
                'steps_failed': self.pipeline_state['steps_failed']
            },
            'configuration_summary': {
                'patch_size': self.config.get('eigenpatches', {}).get('patch_size'),
                'n_components': self.config.get('eigenpatches', {}).get('n_components'),
                'pyramid_levels': self.config.get('eigenpatches', {}).get('pyramid_levels'),
                'lambda_shape': self.config.get('landmark_predictor', {}).get('lambda_shape'),
                'max_iterations': self.config.get('landmark_predictor', {}).get('max_iterations')
            }
        }
        
        # Add performance results if available
        if eval_results:
            exec_summary = eval_results.get('executive_summary', {}).get('executive_summary', {})
            summary['performance_results'] = {
                'dataset_size': exec_summary.get('dataset_size'),
                'mean_error': exec_summary.get('key_results', {}).get('mean_error'),
                'baseline_comparison': exec_summary.get('key_results', {}).get('baseline_comparison'),
                'improvement_vs_baseline': exec_summary.get('key_results', {}).get('improvement_vs_baseline'),
                'best_landmark': exec_summary.get('landmark_analysis', {}).get('best_landmark'),
                'worst_landmark': exec_summary.get('landmark_analysis', {}).get('worst_landmark')
            }
        
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        self.logger.info(f"Summary YAML generated: {output_path}")
    
    def run_full_pipeline(self, skip_completed: bool = True) -> bool:
        """
        Execute the complete pipeline.
        
        Args:
            skip_completed: Whether to skip already completed steps
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        self.logger.info("Starting full experimental pipeline execution")
        self.pipeline_state['start_time'] = datetime.now().isoformat()
        
        # Load checkpoint if exists
        if skip_completed:
            checkpoint_loaded = self.load_checkpoint()
            if checkpoint_loaded:
                self.logger.info(f"Resuming from checkpoint. Completed steps: {self.pipeline_state['steps_completed']}")
        
        # Estimate execution time
        time_estimates = self.estimate_total_time()
        self.logger.info(f"Estimated total execution time: {time_estimates['total']:.0f} seconds ({time_estimates['total']/60:.1f} minutes)")
        
        # Execute pipeline steps
        pipeline_success = True
        
        # Step 1: Training
        if skip_completed and 'training' in self.pipeline_state['steps_completed']:
            self.logger.info("Skipping training step (already completed)")
        else:
            if not self.run_training_step():
                self.logger.error("Training step failed")
                pipeline_success = False
        
        # Step 2: Processing (only if training succeeded)
        if pipeline_success:
            if skip_completed and 'processing' in self.pipeline_state['steps_completed']:
                self.logger.info("Skipping processing step (already completed)")
            else:
                if not self.run_processing_step():
                    self.logger.error("Processing step failed")
                    pipeline_success = False
        
        # Step 3: Evaluation (only if processing succeeded)
        if pipeline_success:
            if skip_completed and 'evaluation' in self.pipeline_state['steps_completed']:
                self.logger.info("Skipping evaluation step (already completed)")
            else:
                if not self.run_evaluation_step():
                    self.logger.error("Evaluation step failed")
                    pipeline_success = False
        
        # Finalization
        self.pipeline_state['end_time'] = datetime.now().isoformat()
        
        if self.pipeline_state['start_time'] and self.pipeline_state['end_time']:
            start_dt = datetime.fromisoformat(self.pipeline_state['start_time'])
            end_dt = datetime.fromisoformat(self.pipeline_state['end_time'])
            self.pipeline_state['total_duration'] = (end_dt - start_dt).total_seconds()
        
        self.save_checkpoint()
        
        # Generate final report
        self.generate_final_report()
        
        if pipeline_success:
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Total execution time: {self.pipeline_state.get('total_duration', 0):.2f} seconds")
            self.logger.info(f"Results available in: {self.output_dir}")
        else:
            self.logger.error("Pipeline completed with errors")
            self.logger.error(f"Failed steps: {self.pipeline_state['steps_failed']}")
        
        return pipeline_success


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Run complete experimental template matching pipeline')
    parser.add_argument('--config', type=str, 
                       default='matching_experimental/configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for pipeline results')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint (skip completed steps)')
    parser.add_argument('--force-restart', action='store_true',
                       help='Force restart (ignore existing checkpoints)')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate execution time without running')
    
    args = parser.parse_args()
    
    # Initialize pipeline executor
    output_dir = args.output_dir or f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    executor = PipelineExecutor(args.config, output_dir)
    
    # Estimate execution time if requested
    if args.estimate_only:
        estimates = executor.estimate_total_time()
        print(f"\nPipeline Execution Time Estimates:")
        print(f"  Training: {estimates['training']:.0f} seconds ({estimates['training']/60:.1f} minutes)")
        print(f"  Processing: {estimates['processing']:.0f} seconds ({estimates['processing']/60:.1f} minutes)")
        print(f"  Evaluation: {estimates['evaluation']:.0f} seconds ({estimates['evaluation']/60:.1f} minutes)")
        print(f"  Overhead: {estimates['overhead']:.0f} seconds")
        print(f"  Total: {estimates['total']:.0f} seconds ({estimates['total']/60:.1f} minutes)")
        return 0
    
    # Execute pipeline
    skip_completed = args.resume and not args.force_restart
    success = executor.run_full_pipeline(skip_completed=skip_completed)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())