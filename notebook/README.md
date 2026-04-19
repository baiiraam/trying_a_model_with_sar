# SAR Water Segmentation with Vision Transformer

A complete deep learning pipeline for semantic segmentation of water bodies in SAR (Synthetic Aperture Radar) imagery using Vision Transformer (ViT) architecture.

## Features

- **Vision Transformer Architecture**: State-of-the-art transformer-based encoder with CNN decoder
- **Shapefile Integration**: Automatic conversion of vector shapefiles to raster masks
- **Dual-Polarization SAR**: Supports VV and VH polarization channels
- **Advanced Training**: Mixed precision training, gradient accumulation, learning rate warmup
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, F1-Score, and more
- **Automated Pipeline**: End-to-end training, validation, and prediction

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Colab Setup (if using Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### Configuration

Edit the `Config` class in the script to set your paths and parameters:

```python
@dataclass
class Config:
    # Paths
    data_dir: str = "/path/to/your/SAR_data"
    output_dir: str = "/path/to/output"
    
    # Image parameters
    img_size: int = 512
    in_channels: int = 2  # SAR dual-pol
    
    # Training parameters
    batch_size: int = 2
    accumulate_steps: int = 4  # Effective batch = 8
    num_epochs: int = 100
    learning_rate: float = 1e-2
```

### Data Structure

Your data directory should be organized as follows:

```
SAR_water_images_and_labels/
├── images/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
└── labels/
    ├── image1.shp
    ├── image1.shx
    ├── image1.dbf
    ├── image2.shp
    └── ...
```

**Important**: Each image file should have a corresponding shapefile with the same base name.

### Running the Pipeline

```bash
python sar_water_segmentation_vit.py
```

The script will:
1. Load and split your data (80/20 train/val)
2. Convert shapefiles to raster masks on-the-fly
3. Train the Vision Transformer model
4. Save checkpoints and metrics
5. Evaluate on validation set
6. Generate predictions

## Model Architecture

### Vision Transformer Encoder
- **Patch Embedding**: Splits images into 16x16 patches
- **Transformer Blocks**: 6 layers with multi-head self-attention
- **Embedding Dimension**: 256
- **Attention Heads**: 8

### CNN Decoder
- Progressive upsampling with convolutional layers
- Batch normalization and GELU activation
- Bilinear interpolation to original resolution

## Training Features

### Optimization
- **Loss Function**: Combined Binary Cross-Entropy + Dice Loss
- **Optimizer**: Adam with weight decay
- **Learning Rate**: Linear warmup (5 epochs) + ReduceLROnPlateau
- **Gradient Accumulation**: Simulate larger batch sizes
- **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training

### Augmentation
- Horizontal and vertical flips
- 90-degree rotation
- Affine transformations (translation, scaling, rotation)
- Gaussian noise and blur
- SAR-optimized normalization

### Regularization
- Early stopping (patience: 15 epochs)
- Learning rate scheduling
- Weight decay

## Outputs

The pipeline generates the following outputs:

```
outputs_for_ViT_2/
├── checkpoints_ViT_2/
│   └── best_model.pth          # Best model checkpoint
├── metrics_ViT_2/
│   ├── training_history.png    # Training curves
│   ├── training_history.csv    # Detailed metrics
│   └── val_evaluation.csv      # Validation results
└── predictions_ViT_2/
    ├── image1_pred.tif         # Predicted masks
    ├── image2_pred.tif
    └── ...
```

## Key Classes and Functions

### Dataset
- `SARWaterDataset`: Handles SAR images and shapefile-to-raster conversion

### Model
- `ViTSegmentation`: Main segmentation model
- `ViTEncoder`: Transformer encoder
- `PatchEmbedding`: Image to patch conversion
- `MultiHeadAttention`: Self-attention mechanism
- `TransformerBlock`: Encoder block

### Loss & Metrics
- `BCEDiceLoss`: Combined loss function
- `calculate_metrics()`: Comprehensive metric calculation

### Training
- `train_one_epoch()`: Single training epoch
- `validate()`: Validation loop
- `lr_warmup()`: Learning rate warmup

### Prediction
- `SARPredictor`: Inference class for predictions

## Performance Metrics

The model reports the following metrics:
- **IoU (Intersection over Union)**
- **Dice Coefficient**
- **Precision**
- **Recall**
- **Specificity**
- **Accuracy**
- **F1-Score**

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: Sufficient space for dataset and outputs
- **Python**: 3.8+
- **PyTorch**: 2.0+

## Tips for Best Results

1. **Data Quality**: Ensure shapefiles accurately match image extent and CRS
2. **Batch Size**: Adjust based on available GPU memory
3. **Learning Rate**: Default 1e-2 works well with warmup
4. **Augmentation**: Enable for better generalization
5. **Monitoring**: Watch training curves for overfitting

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` or `img_size`
- Enable gradient accumulation
- Use mixed precision training (`use_amp=True`)

### Poor Performance
- Check shapefile-image alignment
- Verify data quality
- Increase training epochs
- Adjust learning rate

### Shapefile Loading Issues
- Ensure all shapefile components (.shp, .shx, .dbf) are present
- Verify CRS compatibility with images
- Check for corrupt geometries

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sar_water_vit,
  title={SAR Water Segmentation with Vision Transformer},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sar-water-vit}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Vision Transformer architecture inspired by "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- SAR processing pipeline adapted for water body detection
- Built with PyTorch, Albumentations, and Rasterio

---

**Note**: This is a refactored, production-ready version of the original notebook code with improved structure, documentation, and best practices.
