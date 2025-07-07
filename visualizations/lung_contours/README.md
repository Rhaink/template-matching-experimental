# Lung Contours Visualizations

This directory contains visualization outputs showing predicted lung contours with anatomical connections.

## Generated Files

After running visualization scripts, this directory will contain:

- `XXX_ImageName_contour.png` - Individual lung contour visualizations
- Each image shows:
  - **Cyan lines**: Lung contour connections (anatomically correct)
  - **Yellow dashed lines**: Mediastinal connections
  - **Red circles**: Predicted landmark positions
  - **Numbers**: Landmark indices

## Anatomical Connections

The visualizations use anatomically correct connections:

### Lung Contour (Cyan)
- Connects landmarks following actual lung boundary
- Sequence: 0→12→3→5→7→14→1→13→6→4→2→11→0

### Mediastinal Line (Yellow Dashed)  
- Connects landmarks along central mediastinal region
- Sequence: 0→8→9→10→1

## Generating Visualizations

To create these visualizations:

```bash
# Generate contour visualizations for all test images
python create_correct_visualizations.py

# Generate with custom anatomical connections
python scripts/process_experimental.py --contours --anatomy-file custom_connections.yaml
```

## Sample Files

A few sample visualizations are included:
- `000_Normal-3173_contour.png` - Example normal lung contour
- `001_Viral_Pneumonia-761_contour.png` - Example with pathological changes
- `002_Viral_Pneumonia-707_contour.png` - Example showing anatomical variation

## File Naming Convention

Format: `{index:03d}_{ImageName}_contour.png`
- `index`: Sequential number (000-158 for full dataset)
- `ImageName`: Original image identifier from dataset
- `contour`: Indicates this shows anatomical contour connections

## Clinical Value

These visualizations provide:
- Anatomical accuracy assessment
- Lung shape and size analysis
- Pathological pattern recognition
- Quality control for landmark placement accuracy