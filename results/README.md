# Results Directory

This directory contains experimental results and evaluation outputs.

## Generated Files

After running experiments, this directory will contain:

- `test_predictions.pkl` - Predictions on test dataset
- `evaluation_report.txt` - Detailed performance analysis
- `experimental_results_*.pkl` - Timestamped result files
- `comparison_analysis.html` - Interactive HTML reports (if generated)

## Running Experiments

To generate results:

```bash
# Run complete evaluation pipeline
python scripts/evaluate_experimental.py

# Process test images
python scripts/process_experimental.py --dataset test

# Generate comprehensive analysis
python scripts/run_full_pipeline.py
```

## Expected Results

The experimental platform should replicate the baseline Template Matching performance:

- **Average Error**: ~5.63 pixels
- **Standard Deviation**: ~0.17 pixels  
- **Test Coverage**: 159 images (72 Normal, 48 Viral Pneumonia, 39 COVID)
- **Landmarks**: 15 anatomical points per image

## Result Files

- `.pkl` files contain serialized prediction data and evaluation metrics
- `.txt` files contain human-readable analysis reports
- `.html` files contain interactive visualization reports (when enabled)