# Models Directory

This directory contains trained models for the Template Matching Experimental platform.

## Model Files

After training, the following model files will be generated:

- `experimental_eigenpatches.pkl` - Trained eigenpatches model with PCA components
- `experimental_predictor.pkl` - Complete landmark predictor with shape constraints
- `experimental_evaluator_config.pkl` - Evaluation configuration and baseline data

## Training Models

To train the models:

```bash
# Train all models with default configuration
python scripts/train_experimental.py

# Train with custom configuration
python scripts/train_experimental.py --config configs/custom_config.yaml

# Train specific components
python scripts/train_experimental.py --model-type eigenpatches
```

## Model Size

- Eigenpatches models: ~2-5 MB
- Complete predictor: ~3-8 MB
- Total training time: ~5-10 minutes on modern hardware

## Baseline Performance

The trained models should achieve:
- **Mean Error**: 5.63 ± 0.17 pixels
- **Test Images**: 159 images from coordenadas_prueba_1.csv
- **Landmarks**: 15 anatomical lung landmarks per image