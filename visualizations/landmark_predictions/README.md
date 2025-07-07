# Landmark Predictions Visualizations

This directory contains visualization outputs showing predicted vs ground truth landmarks.

## Generated Files

After running visualization scripts, this directory will contain:

- `XXX_ImageName_landmarks.png` - Individual landmark prediction visualizations
- Each image shows:
  - **Green circles**: Ground truth landmarks
  - **Red X marks**: Predicted landmarks  
  - **Yellow lines**: Error vectors connecting GT to predictions
  - **Numbers**: Landmark indices for anatomical reference

## Generating Visualizations

To create these visualizations:

```bash
# Generate landmark visualizations for all test images
python create_correct_visualizations.py

# Generate specific subset
python scripts/process_experimental.py --visualize --max-images 10
```

## Sample Files

A few sample visualizations are included to demonstrate the output format:
- `000_Normal-3173_landmarks.png` - Example normal case
- `001_Viral_Pneumonia-761_landmarks.png` - Example viral pneumonia case
- `002_Viral_Pneumonia-707_landmarks.png` - Example with anatomical variations

## File Naming Convention

Format: `{index:03d}_{ImageName}_landmarks.png`
- `index`: Sequential number (000-158 for full dataset)
- `ImageName`: Original image identifier from dataset
- `landmarks`: Indicates this shows landmark predictions

## Performance Visualization

These images allow visual verification of:
- Landmark position accuracy
- Error distribution across different anatomical regions
- Performance consistency across pathology types
- Quality of anatomical landmark placement