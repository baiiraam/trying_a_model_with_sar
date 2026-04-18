# Water Segmentation with U-Net

A deep learning project for water body segmentation in satellite/aerial imagery using a U-Net architecture with adaptive learning rate scheduling.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Model Improvements](#model-improvements)
- [Outputs](#outputs)

## 🌊 Overview

This project implements a U-Net convolutional neural network for pixel-wise segmentation of water bodies in satellite or aerial imagery. The model learns to classify each pixel as either water or non-water, producing binary masks that delineate water features like rivers, lakes, and coastlines.

## ✨ Key Features

### 1. **Adaptive Learning Rate Scheduling**
- **ReduceLROnPlateau**: Automatically reduces learning rate when validation loss plateaus
- Prevents the model from getting stuck in local minima
- Configured with:
  - `patience=5`: Waits 5 epochs before reducing LR
  - `factor=0.5`: Reduces LR by 50% when triggered
  - `min_lr=1e-7`: Prevents LR from becoming too small

### 2. **Mixed Loss Function**
- Combines Binary Cross-Entropy (BCE) and Dice Loss
- BCE: Pixel-wise classification accuracy
- Dice Loss: Focuses on overlap between prediction and ground truth
- Default weights: 50% BCE + 50% Dice

### 3. **Data Augmentation**
- Horizontal and vertical flips
- Random rotations (90°)
- Scale, shift, and rotation transformations
- Gaussian noise and blur
- Brightness and contrast adjustments
- Ensures model generalization

### 4. **Comprehensive Metrics**
- IoU (Intersection over Union)
- Dice Score
- Precision and Recall
- Specificity and Accuracy
- F1 Score
- Per-image and aggregate statistics

### 5. **Best Model Checkpointing**
- Automatically saves the best model based on validation loss
- Saves both best and final models
- Includes optimizer and scheduler states for resuming training

### 6. **Early Stopping**
- Prevents overfitting by stopping training when validation loss stops improving
- Default patience: 15 epochs

### 7. **Mixed Precision Training**
- Uses automatic mixed precision (AMP) for faster training on compatible GPUs
- Reduces memory usage while maintaining accuracy

## 🏗️ Architecture

### U-Net Model

The U-Net architecture consists of:

**Encoder (Contracting Path)**:
- 4 downsampling blocks
- Each block: 2 conv layers (3x3) + BatchNorm + ReLU + MaxPool (2x2)
- Feature maps: 64 → 128 → 256 → 512 → 1024

**Bottleneck**:
- Double convolution at the deepest level
- Captures the most abstract features

**Decoder (Expanding Path)**:
- 4 upsampling blocks
- Each block: Upsampling (2x2) + Concatenation with encoder features + 2 conv layers
- Feature maps: 1024 → 512 → 256 → 128 → 64

**Output**:
- 1x1 convolution to produce final single-channel mask
- Sigmoid activation for binary classification

**Key Design Choices**:
- **Skip connections**: Connect encoder to decoder at each level
- **Batch normalization**: Stabilizes training
- **ReLU activation**: Introduces non-linearity
- **Parameter count**: ~31M parameters (default with 64 initial features)

## 📦 Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
rasterio>=1.3.0
geopandas>=0.13.0
albumentations>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
scikit-learn>=1.3.0

# For Google Colab
google-colab
```

Install via:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
water_segmentation/
│
├── data/
│   ├── train/
│   │   ├── images/          # Training images (.tif, .png, .jpg)
│   │   └── masks/           # Training masks (.tif)
│   ├── val/
│   │   ├── images/          # Validation images
│   │   └── masks/           # Validation masks
│   └── test/
│       ├── images/          # Test images
│       └── masks/           # Test masks (ground truth)
│
├── output/
│   ├── checkpoints/
│   │   ├── best_model.pth   # Best model based on validation loss
│   │   └── final_model.pth  # Final model at end of training
│   ├── metrics/
│   │   ├── training_history.png
│   │   ├── training_history.csv
│   │   ├── test_results.csv
│   │   └── test_summary.csv
│   └── predictions/
│       └── [prediction visualizations]
│
└── trial_refactored.ipynb   # Main training notebook
```

## ⚙️ Configuration

The `Config` class in the notebook contains all hyperparameters:

```python
class Config:
    # Paths
    DATA_DIR = "/path/to/data"
    OUTPUT_DIR = "/path/to/output"
    
    # Model parameters
    IMG_SIZE = 256              # Input image size
    IN_CHANNELS = 3             # RGB images
    OUT_CHANNELS = 1            # Binary mask
    INIT_FEATURES = 64          # Initial feature maps
    
    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Learning rate scheduler
    LR_PATIENCE = 5             # Epochs before reducing LR
    LR_FACTOR = 0.5             # LR reduction factor
    LR_MIN = 1e-7               # Minimum LR
    
    # Early stopping
    EARLY_STOP_PATIENCE = 15    # Epochs before stopping
    
    # Other
    USE_AMP = True              # Mixed precision training
    AUGMENT = True              # Data augmentation
```

## 🎯 Training Process

### 1. Data Loading
- Images and masks are loaded from disk using rasterio
- Images are normalized to [0, 1] range
- Masks are binarized (0 or 1)
- Augmentations are applied during training

### 2. Training Loop
For each epoch:
1. **Training Phase**:
   - Model processes batches of images
   - Loss is computed (BCE + Dice)
   - Gradients are backpropagated
   - Optimizer updates weights
   - Metrics are calculated

2. **Validation Phase**:
   - Model evaluates on validation set (no gradient computation)
   - Validation loss and metrics are calculated

3. **Learning Rate Adjustment**:
   - ReduceLROnPlateau monitors validation loss
   - If loss doesn't improve for `LR_PATIENCE` epochs, LR is reduced
   - Process continues until `LR_MIN` is reached

4. **Checkpointing**:
   - If validation loss improves, save as best model
   - Track epochs without improvement

5. **Early Stopping**:
   - If no improvement for `EARLY_STOP_PATIENCE` epochs, stop training

### 3. Evaluation
- Load best and final models
- Predict on test images
- Calculate metrics against ground truth
- Generate visualizations

## 📊 Learning Rate Scheduling

### Why ReduceLROnPlateau?

The ReduceLROnPlateau scheduler addresses a common problem in training: **loss plateaus**.

**The Problem**:
- Initially, the model learns quickly with a high learning rate
- As training progresses, the learning rate that was optimal becomes too large
- The model oscillates around a minimum without converging
- Loss stops decreasing (plateau)

**The Solution**:
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',              # Minimize validation loss
    factor=0.5,              # Reduce LR by 50%
    patience=5,              # Wait 5 epochs before reducing
    verbose=True,            # Print when LR changes
    min_lr=1e-7              # Don't go below this value
)
```

**How It Works**:
1. After each epoch, the scheduler checks the validation loss
2. If the loss hasn't improved for 5 consecutive epochs
3. The learning rate is multiplied by 0.5 (halved)
4. Training continues with the new, smaller learning rate
5. This process repeats until the minimum LR is reached

**Example Timeline**:
```
Epoch 1-10:  LR = 1e-3,  Loss decreasing
Epoch 11-15: LR = 1e-3,  Loss plateau
Epoch 16:    LR = 5e-4,  (reduced!)  Loss starts decreasing again
Epoch 17-25: LR = 5e-4,  Loss decreasing
Epoch 26-30: LR = 5e-4,  Loss plateau
Epoch 31:    LR = 2.5e-4 (reduced!)  Loss decreasing again
...
```

**Benefits**:
- **Automatic**: No manual LR tuning required
- **Adaptive**: Responds to actual training dynamics
- **Effective**: Helps model converge to better solutions
- **Prevents underfitting**: Keeps LR high when the model is still learning
- **Prevents overfitting**: Reduces LR when model needs fine-tuning

## 📈 Evaluation Metrics

### Primary Metrics

1. **IoU (Intersection over Union)**
   ```
   IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)
   ```
   - Measures overlap between prediction and truth
   - Range: [0, 1], higher is better
   - Standard metric for segmentation tasks

2. **Dice Score**
   ```
   Dice = 2 × (Prediction ∩ Ground Truth) / (|Prediction| + |Ground Truth|)
   ```
   - Similar to IoU but more sensitive to small objects
   - Range: [0, 1], higher is better
   - Commonly used in medical imaging

3. **Precision**
   ```
   Precision = True Positives / (True Positives + False Positives)
   ```
   - How many predicted water pixels are actually water?
   - High precision → few false alarms

4. **Recall (Sensitivity)**
   ```
   Recall = True Positives / (True Positives + False Negatives)
   ```
   - How many actual water pixels were detected?
   - High recall → few missed detections

5. **F1 Score**
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```
   - Harmonic mean of precision and recall
   - Balanced measure of model performance

### Loss Functions

1. **Binary Cross-Entropy (BCE)**
   - Measures pixel-wise classification error
   - Good for balanced datasets

2. **Dice Loss**
   - 1 - Dice Score
   - Handles class imbalance better than BCE
   - Focuses on overlap

3. **Combined Loss**
   - 0.5 × BCE + 0.5 × Dice
   - Leverages strengths of both

## 🚀 Usage

### Training

1. **Prepare your data**:
   - Organize images and masks in the structure shown above
   - Ensure masks are binary (0 for non-water, 255 or 1 for water)

2. **Update configuration**:
   ```python
   config.DATA_DIR = "/your/data/path"
   config.OUTPUT_DIR = "/your/output/path"
   ```

3. **Run the notebook**:
   - Execute cells sequentially in Google Colab or Jupyter
   - Monitor training progress through progress bars
   - View real-time metrics and plots

### Inference

```python
# Load the best model
predictor = WaterSegmentationPredictor("path/to/best_model.pth")

# Predict on a new image
mask, probability, profile = predictor.predict("path/to/image.tif")

# mask: Binary segmentation (0 or 1)
# probability: Continuous probability map (0 to 1)
# profile: Rasterio profile for saving results
```

### Saving Predictions

```python
import rasterio

# Update profile for output
profile.update(
    count=1,
    dtype=rasterio.uint8,
    compress='lzw'
)

# Save binary mask
with rasterio.open("output_mask.tif", 'w', **profile) as dst:
    dst.write(mask.astype(np.uint8), 1)

# Save probability map
profile.update(dtype=rasterio.float32)
with rasterio.open("output_prob.tif", 'w', **profile) as dst:
    dst.write(prob.astype(np.float32), 1)
```

## 🔧 Model Improvements

### Refactored from Original

1. **Learning Rate Scheduler**: Added ReduceLROnPlateau for adaptive LR
2. **Code Organization**: Separated into clear sections with docstrings
3. **Configuration Class**: Centralized all parameters
4. **Metric Calculation**: Improved numerical stability with smoothing
5. **Error Handling**: Better handling of edge cases
6. **Documentation**: Comprehensive comments and explanations
7. **Best Practices**:
   - Used `weights_only=False` for torch.load (required for full checkpoint)
   - Proper device management
   - Gradient scaler for mixed precision
   - Consistent naming conventions
   - Type hints in docstrings

### Removed Issues

1. **Hard-coded paths**: Now use Config class
2. **Magic numbers**: All hyperparameters are named constants
3. **Repetitive code**: Extracted common operations into functions
4. **Missing validation**: Added input validation where needed
5. **Poor commenting**: Added comprehensive documentation

## 📊 Outputs

### Training Outputs

1. **Checkpoints**:
   - `best_model.pth`: Best model based on validation loss
   - `final_model.pth`: Model at end of training
   - Includes model weights, optimizer state, scheduler state

2. **Metrics**:
   - `training_history.csv`: Loss and metrics per epoch
   - `training_history.png`: Visualization of training curves
   - Shows loss, Dice score, IoU, and learning rate over time

3. **Test Results**:
   - `test_results.csv`: Per-image metrics on test set
   - `test_summary.csv`: Aggregate statistics (mean, std, min, max)

4. **Predictions**:
   - Visualization of predictions vs ground truth
   - Includes original image, probability map, binary mask

### Typical Training Output

```
Epoch 1/100
Learning Rate: 1.00e-03
Training: 100%|████████| Loss: 0.3456, Dice: 0.7123
Validation: 100%|████████| Loss: 0.3123, Dice: 0.7456
✅ New best model saved! Val Loss: 0.3123 | Val Dice: 0.7456

...

Epoch 15/100
Learning Rate: 1.00e-03
⏳ No improvement for 5 epoch(s)

Epoch 16/100
Learning Rate: 5.00e-04  (Learning rate reduced!)
✅ New best model saved! Val Loss: 0.2567 | Val Dice: 0.8123

...

🛑 Early stopping triggered after 45 epochs
Best Val Loss: 0.2234 | Best Val Dice: 0.8567
```

## 🤝 Contributing

Suggestions for improvement:
- Experiment with different architectures (DeepLabV3, SegFormer)
- Try different loss functions (Focal Loss, Tversky Loss)
- Implement learning rate warm-up
- Add test-time augmentation
- Implement ensemble methods
- Add uncertainty estimation

## 📝 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- U-Net architecture from Ronneberger et al. (2015)
- PyTorch team for the excellent deep learning framework
- Albumentations for data augmentation library

---

**Happy Segmenting! 🌊**
