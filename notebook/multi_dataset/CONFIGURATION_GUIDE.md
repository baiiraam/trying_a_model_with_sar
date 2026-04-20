# Multi-Dataset Training Configuration Guide

## Overview

This notebook supports training on two different SAR water segmentation datasets:

### Dataset 1: Shapefile-based
- **Images**: `.tif` files
- **Labels**: `.shp` files (shapefiles)
- **Structure**:
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
      └── ...
  ```

### Dataset 2: Sentinel-1 with Mask TIFs
- **Images**: `sentinel12_s1_*_img.tif`
- **Masks**: `sentinel12_s1_*_msk.tif`
- **Structure**:
  ```
  Sentinel1_Dataset/
  ├── folder1/
  │   ├── sentinel12_s1_80_img.tif
  │   ├── sentinel12_s1_80_msk.tif
  │   ├── sentinel12_s1_80_valid.tif
  │   ├── sentinel12_s2_80_img.tif      (not used)
  │   ├── sentinel12_copdem30_80_elevation.tif  (not used)
  │   └── sentinel12_copdem30_80_slope.tif      (not used)
  ├── folder2/
  │   ├── sentinel12_s1_81_img.tif
  │   ├── sentinel12_s1_81_msk.tif
  │   └── ...
  └── ...
  ```

## Configuration Steps

### 1. Update Paths in the Configuration Cell

Edit the `Config` class in **Cell 2** of the notebook:

```python
@dataclass
class Config:
    # ===== PATHS - UPDATE THESE =====
    
    # Dataset 1: Shapefile-based
    dataset1_dir: str = "/content/drive/MyDrive/YOUR_PATH_HERE"
    dataset1_images: str = "images"  # subfolder name
    dataset1_labels: str = "labels"   # subfolder name
    
    # Dataset 2: Sentinel-1 with mask TIF files
    dataset2_dir: str = "/content/drive/MyDrive/YOUR_SENTINEL_PATH_HERE"
    
    # Output directory
    output_dir: str = "/content/drive/MyDrive/SAR_water_combined_outputs"
    
    # ===== DATASET SELECTION =====
    use_dataset1: bool = True   # Set to False to skip Dataset 1
    use_dataset2: bool = True   # Set to False to skip Dataset 2
```

### 2. Example Configurations

#### Option A: Use Both Datasets (Recommended)
```python
dataset1_dir: str = "/content/drive/MyDrive/SAR_water_images_and_labels"
dataset2_dir: str = "/content/drive/MyDrive/Sentinel1_Dataset"
use_dataset1: bool = True
use_dataset2: bool = True
```

#### Option B: Use Only Shapefile Dataset
```python
dataset1_dir: str = "/content/drive/MyDrive/SAR_water_images_and_labels"
use_dataset1: bool = True
use_dataset2: bool = False
```

#### Option C: Use Only Sentinel-1 Dataset
```python
dataset2_dir: str = "/content/drive/MyDrive/Sentinel1_Dataset"
use_dataset1: bool = False
use_dataset2: bool = True
```

## Key Features

### Automatic Dataset Detection
The notebook automatically:
- Scans directories for valid data files
- Matches images with corresponding labels/masks
- Combines datasets if both are enabled
- Splits into train/validation (80/20 by default)

### Flexible Channel Handling
- Automatically reads only the first 2 bands from Sentinel-1 images
- Handles images with different numbers of channels
- Pads or crops to exactly 2 channels as needed

### File Pattern Matching

**Dataset 2** looks for files matching these patterns:
- Images: `*_s1_*_img.tif`
- Masks: `*_s1_*_msk.tif`

The notebook will:
- Ignore elevation, slope, and Sentinel-2 files
- Recursively search all subdirectories
- Match image-mask pairs by replacing `_img.tif` with `_msk.tif`

## Training Parameters

You can adjust these in the `Config` class:

```python
# Image parameters
img_size: int = 512              # Resize all images to this size
in_channels: int = 2             # Sentinel-1 dual-pol (VV, VH)

# Training parameters
batch_size: int = 2              # Adjust based on GPU memory
accumulate_steps: int = 4        # Effective batch = 8
num_epochs: int = 100
learning_rate: float = 1e-2
warmup_epochs: int = 5

# Data split
val_split: float = 0.2           # 20% validation
seed: int = 42                   # For reproducibility
```

## Troubleshooting

### Issue: "No images found"
**Solution**: Check that your `dataset1_dir` and `dataset1_images` paths are correct

### Issue: "No Sentinel-1 data found"
**Solution**: 
- Verify `dataset2_dir` points to the root directory containing your Sentinel-1 folders
- Ensure files follow the naming pattern `sentinel12_s1_*_img.tif`

### Issue: "Mask not found for image"
**For Dataset 1**: Ensure shapefile has same base name as image
- ✅ Correct: `image1.tif` → `image1.shp`
- ❌ Wrong: `image1.tif` → `image_1.shp`

**For Dataset 2**: Ensure mask follows pattern
- ✅ Correct: `sentinel12_s1_80_img.tif` → `sentinel12_s1_80_msk.tif`
- ❌ Wrong: `sentinel12_s1_80_img.tif` → `sentinel12_s1_80_mask.tif`

### Issue: Out of Memory Error
**Solution**: Reduce batch size or image size:
```python
batch_size: int = 1
img_size: int = 256
```

## Output Structure

After training, you'll find:

```
SAR_water_combined_outputs/
├── checkpoints/
│   └── best_model.pth                    # Best model weights
├── metrics/
│   ├── training_history.png              # Training curves
│   ├── training_history.csv              # Detailed metrics per epoch
│   ├── predictions_visualization.png     # Sample predictions
│   └── final_validation_metrics.csv      # Final performance
└── predictions/
    └── (saved predictions)
```

## Data Visualization

The notebook includes cells to:
1. **Visualize sample data** (Cell 7) - Shows VV, VH, and mask for samples
2. **Plot training curves** (Cell 13) - Loss, Dice, IoU, and learning rate
3. **Visualize predictions** (Cell 14) - Ground truth vs predictions

## Running the Notebook

1. **Mount Google Drive** (if using Colab)
2. **Update configuration** in Cell 2
3. **Run all cells** sequentially
4. **Monitor training** progress in real-time
5. **Review results** in output directory

## Performance Tips

### For Better Results:
- ✅ Use both datasets for more diverse training data
- ✅ Enable augmentation (`augment: True`)
- ✅ Use warmup (`warmup_epochs: 5`)
- ✅ Enable mixed precision training (`use_amp: True`)

### For Faster Training:
- Reduce `img_size` to 256 or 384
- Increase `accumulate_steps` to use larger effective batch size
- Use fewer epochs with early stopping enabled

## Advanced Options

### Custom Data Split
Change the validation split ratio:
```python
val_split: float = 0.15  # 85% train, 15% val
```

### Modify Architecture
Adjust ViT parameters:
```python
patch_size: int = 16      # Larger = faster but less detail
embed_dim: int = 256      # Larger = more capacity
depth: int = 6            # More layers = deeper model
num_heads: int = 8        # Multi-head attention
```

### Learning Rate Schedule
```python
learning_rate: float = 1e-2    # Initial LR
warmup_epochs: int = 5         # Linear warmup
lr_patience: int = 5           # Plateau patience
lr_factor: float = 0.5         # LR reduction factor
```

## Validation

The notebook automatically:
- Splits data into train/validation
- Evaluates on validation set after each epoch
- Saves best model based on validation loss
- Computes comprehensive metrics (IoU, Dice, Precision, Recall, F1)

## Next Steps After Training

1. Review training curves for signs of overfitting
2. Evaluate on test set (if available)
3. Fine-tune hyperparameters if needed
4. Deploy model for inference on new images
5. Export predictions as GeoTIFF files

---

**Need Help?**
- Check that all paths are absolute and correct
- Verify file naming patterns match expected formats
- Review error messages for specific issues
- Ensure sufficient GPU memory for chosen batch size
