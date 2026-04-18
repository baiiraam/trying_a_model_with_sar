# SAR Water Segmentation with Vision Transformer - Code Explanation

## Table of Contents
1. [Overview](#overview)
2. [Configuration Block](#configuration-block)
3. [Vision Transformer Components](#vision-transformer-components)
4. [Model Architecture](#model-architecture)
5. [Dataset Class](#dataset-class)
6. [Loss Functions](#loss-functions)
7. [Trainer Class](#trainer-class)
8. [Predictor Class](#predictor-class)
9. [Data Utilities](#data-utilities)
10. [Main Pipeline](#main-pipeline)

---

## Overview

This implementation uses a **Vision Transformer (ViT)** for semantic segmentation of water bodies from Synthetic Aperture Radar (SAR) imagery. The key components are:

- **Vision Transformer Encoder**: Processes image patches using self-attention
- **Segmentation Decoder**: Upsamples features to pixel-level predictions
- **Custom Dataset**: Handles 2-band SAR images and shapefile labels
- **Augmentation Pipeline**: Comprehensive data augmentation for training
- **Training System**: Complete training loop with learning rate scheduling

---

## Configuration Block

```python
class Config:
    """Central configuration for the training pipeline"""
```

**Purpose**: Centralized configuration management for all hyperparameters and paths.

**Key Parameters**:

### Path Configuration
- `BASE_PATH`: Root directory containing your data
- `IMAGES_FOLDER`: Subfolder with .tif SAR images
- `LABELS_FOLDER`: Subfolder with .shp shapefiles

### Model Architecture
- `IMAGE_SIZE = 512`: Input images are resized to 512×512
- `PATCH_SIZE = 16`: Each image is divided into 16×16 pixel patches
  - Results in (512/16)² = 1024 patches per image
- `EMBED_DIM = 768`: Dimension of patch embeddings
- `NUM_HEADS = 12`: Number of parallel attention heads
- `NUM_LAYERS = 12`: Number of transformer encoder blocks
- `MLP_RATIO = 4`: Feed-forward network expansion (768 → 3072 → 768)

### Training Parameters
- `BATCH_SIZE = 4`: Number of images per training batch
- `NUM_EPOCHS = 100`: Total training iterations
- `LEARNING_RATE = 3e-4`: Initial learning rate (0.0003)
- `WEIGHT_DECAY = 0.01`: L2 regularization strength

### Augmentation Probabilities
- `AUGMENT_PROB = 0.85`: 85% chance to augment each sample
- `FLIP_PROB = 0.5`: Horizontal/vertical flip probability
- `ROTATION_PROB = 0.7`: 90°/180°/270° rotation probability
- `BRIGHTNESS_PROB = 0.4`: Brightness adjustment probability
- `NOISE_PROB = 0.3`: Random noise addition probability

**Why These Values?**
- 512×512 is optimal for modern GPUs (balances resolution and memory)
- 16×16 patches are standard for ViT (from original paper)
- 768 embedding dimension provides rich feature representation
- 12 layers × 12 heads matches ViT-Base architecture

---

## Vision Transformer Components

### 1. Patch Embedding (`PatchEmbedding`)

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=512, patch_size=16, in_channels=2, embed_dim=768):
```

**Purpose**: Converts the input image into a sequence of patch embeddings.

**How It Works**:
1. **Input**: (Batch, 2, 512, 512) - SAR image with VV and VH bands
2. **Convolution**: Applies Conv2d with kernel_size=16, stride=16
   - This divides the image into non-overlapping 16×16 patches
   - Each patch becomes a 768-dimensional vector
3. **Output**: (Batch, 1024, 768) - Sequence of 1024 patch embeddings

**Mathematical Details**:
- Number of patches = (512/16) × (512/16) = 32 × 32 = 1024
- Each patch: 16 × 16 × 2 = 512 values → compressed to 768 dimensions
- The convolutional projection learns to extract features from each patch

**Code Breakdown**:
```python
self.projection = nn.Conv2d(
    in_channels,      # 2 (VV and VH bands)
    embed_dim,        # 768 (output embedding dimension)
    kernel_size=patch_size,  # 16×16 convolution
    stride=patch_size        # Non-overlapping patches
)
```

### 2. Multi-Head Self-Attention (`MultiHeadSelfAttention`)

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
```

**Purpose**: Allows the model to attend to different parts of the image simultaneously.

**How It Works**:
1. **Generate Q, K, V**: Linear projection creates Query, Key, Value matrices
2. **Split Heads**: 768-dim vector → 12 heads × 64 dimensions per head
3. **Attention Computation**: 
   - Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   - Each head computes attention independently
4. **Combine Heads**: Concatenate 12 heads back to 768 dimensions

**Visual Example**:
```
Input patch embeddings: [1024 patches × 768 dim]
                          ↓
Generate Q, K, V:        [1024 × 768] → 3 × [1024 × 768]
                          ↓
Split into heads:        [12 heads × 1024 patches × 64 dim]
                          ↓
Attention computation:   Each head finds relationships between patches
                          ↓
Combine heads:           [1024 patches × 768 dim]
```

**Why Multiple Heads?**
- Different heads learn different relationships
- Head 1 might learn spatial adjacency
- Head 2 might learn texture similarity
- Head 3 might learn water-land boundaries

**Code Breakdown**:
```python
# Generate Q, K, V in one operation (efficient)
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

# Compute attention scores
attn = (q @ k.transpose(-2, -1)) * self.scale  # Scale by 1/√64

# Apply softmax to get attention weights
attn = attn.softmax(dim=-1)

# Apply attention to values
x = (attn @ v)  # Weighted combination
```

### 3. Feed-Forward Network (`MLP`)

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.1):
```

**Purpose**: Processes each patch embedding independently with non-linear transformations.

**Architecture**:
```
Input: 768 dim
   ↓
Linear: 768 → 3072 (expansion by MLP_RATIO=4)
   ↓
GELU activation (smooth non-linearity)
   ↓
Dropout: 0.1
   ↓
Linear: 3072 → 768 (back to original dimension)
   ↓
Dropout: 0.1
   ↓
Output: 768 dim
```

**Why GELU?**
- GELU (Gaussian Error Linear Unit) is smoother than ReLU
- Works better for transformers (empirically proven)
- Formula: GELU(x) = x * Φ(x), where Φ is cumulative normal distribution

### 4. Transformer Block (`TransformerBlock`)

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
```

**Purpose**: Combines attention and feed-forward layers with residual connections.

**Architecture**:
```
Input patch embeddings
   ↓
Layer Norm → Multi-Head Attention → Residual Add
   ↓
Layer Norm → MLP → Residual Add
   ↓
Output patch embeddings
```

**Residual Connections**:
```python
# Attention with residual
x = x + self.attn(self.norm1(x))

# MLP with residual
x = x + self.mlp(self.norm2(x))
```

**Why Residuals?**
- Helps gradient flow during backpropagation
- Allows training very deep networks (12 layers)
- Each layer only needs to learn the "delta" (change)

### 5. ViT Encoder (`ViTEncoder`)

```python
class ViTEncoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, ...):
```

**Purpose**: Complete vision transformer encoder that processes the entire image.

**Processing Pipeline**:
```
Input: (Batch, 2, 512, 512) SAR image
   ↓
Patch Embedding: → (Batch, 1024, 768)
   ↓
Add Positional Embedding: → (Batch, 1024, 768)
   ↓
Transformer Block 1: → (Batch, 1024, 768)
   ↓
Transformer Block 2: → (Batch, 1024, 768)
   ...
   ↓
Transformer Block 12: → (Batch, 1024, 768)
   ↓
Layer Norm: → (Batch, 1024, 768)
   ↓
Output: Encoded features
```

**Positional Embeddings**:
```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
```
- **Why?** Self-attention is permutation-invariant (order doesn't matter)
- Positional embeddings tell the model where each patch is located
- Learned during training (not fixed like in original Transformer)
- Initialized with truncated normal distribution

**Key Code**:
```python
# Add positional information
x = x + self.pos_embed

# Process through all transformer blocks
for block in self.blocks:
    x = block(x)
```

---

## Model Architecture

### Segmentation Head (`SegmentationHead`)

```python
class SegmentationHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1, image_size=512, patch_size=16):
```

**Purpose**: Converts patch-level features back to pixel-level predictions.

**Upsampling Strategy**:
```
Input: (Batch, 1024, 768) - patch embeddings
   ↓
Reshape: → (Batch, 768, 32, 32) - spatial feature map
   ↓
Conv 3×3: 768 → 512 channels
   ↓
Transpose Conv (×2): 32×32 → 64×64, 512 channels
   ↓
Transpose Conv (×2): 64×64 → 128×128, 256 channels
   ↓
Transpose Conv (×2): 128×128 → 256×256, 128 channels
   ↓
Transpose Conv (×2): 256×256 → 512×512, 64 channels
   ↓
Conv 1×1: 64 → 1 channel (final prediction)
   ↓
Output: (Batch, 1, 512, 512) - pixel-level water mask
```

**Why Transpose Convolutions?**
- Also called "deconvolutions" or "upsampling convolutions"
- Learnable upsampling (better than simple interpolation)
- Each layer doubles spatial resolution

**Code Breakdown**:
```python
# First reshape patches to spatial grid
x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, 768, 32, 32)

# Progressive upsampling with learned features
nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),  # 32→64
nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 64→128
nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 128→256
nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 256→512
```

### Complete ViT Segmentation Model (`ViTSegmentationModel`)

```python
class ViTSegmentationModel(nn.Module):
    def __init__(self, image_size=512, patch_size=16, ...):
```

**Purpose**: Combines encoder and decoder for end-to-end segmentation.

**Complete Forward Pass**:
```
SAR Image (2, 512, 512)
   ↓
[ENCODER]
   ↓
Patch Embeddings (1024, 768)
   ↓
[DECODER]
   ↓
Water Mask (1, 512, 512)
```

**Forward Method**:
```python
def forward(self, x):
    features = self.encoder(x)    # (B, 1024, 768)
    output = self.decoder(features)  # (B, 1, 512, 512)
    return output
```

---

## Dataset Class

### SAR Segmentation Dataset (`SARSegmentationDataset`)

```python
class SARSegmentationDataset(Dataset):
    def __init__(self, image_paths, shapefile_paths, target_size=512, ...):
```

**Purpose**: Loads SAR images and converts shapefiles to masks, with augmentation.

### Key Methods:

#### 1. `_load_sar_image()`

**Purpose**: Load and normalize 2-band SAR image.

**Process**:
```python
# 1. Read raster file
with rasterio.open(image_path) as src:
    img = src.read()  # Shape: (bands, height, width)

# 2. Ensure exactly 2 bands (VV and VH)
if img.shape[0] > 2:
    img = img[:2]  # Take first 2 bands
elif img.shape[0] == 1:
    img = np.repeat(img, 2, axis=0)  # Duplicate if only 1 band

# 3. Normalize each band to [0, 1]
for i in range(2):
    band_min, band_max = img[i].min(), img[i].max()
    normalized[i] = (img[i] - band_min) / (band_max - band_min)
```

**Why Normalize Per-Band?**
- VV and VH have different intensity ranges
- Neural networks train better with normalized inputs [0, 1]
- Prevents one band from dominating the other

#### 2. `_shapefile_to_mask()`

**Purpose**: Convert vector shapefiles to raster masks.

**Process**:
```python
# 1. Get spatial reference from the image
with rasterio.open(reference_image_path) as src:
    transform = src.transform  # Geospatial transform matrix
    shape = (src.height, src.width)

# 2. Read shapefile polygons
gdf = gpd.read_file(shapefile_path)

# 3. Rasterize polygons to binary mask
geometries = [(geom, 1) for geom in gdf.geometry]
mask = features.rasterize(
    geometries,
    out_shape=shape,
    transform=transform,
    fill=0,  # Background = 0
    dtype=np.uint8
)
```

**What This Does**:
- Converts polygon boundaries → filled pixels
- Matches image coordinate system exactly
- Creates binary mask: 1 = water, 0 = non-water

#### 3. `_resize_image()` and `_resize_mask()`

**Purpose**: Resize to 512×512 for model input.

**Image Resizing** (Bilinear Interpolation):
```python
image = F.interpolate(
    image.unsqueeze(0),  # Add batch dimension
    size=(512, 512),
    mode='bilinear',     # Smooth interpolation
    align_corners=False
)
```

**Mask Resizing** (Nearest Neighbor):
```python
mask = F.interpolate(
    mask.unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
    size=(512, 512),
    mode='nearest'  # Preserve binary values (0 or 1)
)
```

**Why Different Methods?**
- **Images**: Bilinear preserves smooth transitions
- **Masks**: Nearest neighbor keeps crisp boundaries (no intermediate values)

#### 4. `_apply_augmentations()`

**Purpose**: Apply random augmentations to increase training data diversity.

**Augmentation Types**:

**A. Horizontal/Vertical Flips**
```python
if random.random() < Config.FLIP_PROB:  # 50% chance
    image_np = np.flip(image_np, axis=2).copy()  # Flip width
    mask_np = np.flip(mask_np, axis=1).copy()
```
- Increases dataset 4× (original, H-flip, V-flip, HV-flip)
- Water bodies look the same from different angles

**B. 90° Rotations**
```python
if random.random() < Config.ROTATION_PROB:  # 70% chance
    k = random.randint(1, 3)  # Rotate 90°, 180°, or 270°
    image_np = np.rot90(image_np, k=k, axes=(1, 2))
    mask_np = np.rot90(mask_np, k=k, axes=(0, 1))
```
- Increases dataset 4× more
- Helps model learn rotation invariance

**C. Brightness Adjustment**
```python
if random.random() < Config.BRIGHTNESS_PROB:  # 40% chance
    factor = random.uniform(0.8, 1.2)  # ±20% brightness
    image_np = np.clip(image_np * factor, 0, 1)
```
- Simulates different SAR acquisition conditions
- Makes model robust to intensity variations

**D. Random Noise**
```python
if random.random() < Config.NOISE_PROB:  # 30% chance
    noise = np.random.normal(0, 0.02, image_np.shape)
    image_np = np.clip(image_np + noise, 0, 1)
```
- Simulates SAR speckle noise
- Prevents overfitting to clean images

**Why Augmentation Matters**:
- Small datasets (< 100 images) → Risk of overfitting
- Augmentation creates "new" training samples
- Model learns invariances (rotation, brightness, etc.)

#### 5. `__getitem__()` - The Complete Pipeline

```python
def __getitem__(self, idx):
    # 1. Load raw data
    image = self._load_sar_image(self.image_paths[idx])
    mask = self._shapefile_to_mask(...)
    
    # 2. Convert to tensors
    image_tensor = torch.from_numpy(image).float()
    mask_tensor = torch.from_numpy(mask).float()
    
    # 3. Resize to 512×512
    image_tensor = self._resize_image(image_tensor)
    mask_tensor = self._resize_mask(mask_tensor)
    
    # 4. Apply augmentations (85% probability)
    if self.augment and random.random() < self.augment_prob:
        image_tensor, mask_tensor = self._apply_augmentations(...)
    
    # 5. Add channel dimension to mask
    mask_tensor = mask_tensor.unsqueeze(0)  # (H, W) → (1, H, W)
    
    return image_tensor, mask_tensor
```

**Output Shapes**:
- `image_tensor`: (2, 512, 512) - VV and VH bands
- `mask_tensor`: (1, 512, 512) - Binary water mask

---

## Loss Functions

### 1. Dice Loss (`DiceLoss`)

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
```

**Purpose**: Measures overlap between predicted and true masks.

**Formula**:
```
Dice Coefficient = 2 × |Prediction ∩ Target| / (|Prediction| + |Target|)
Dice Loss = 1 - Dice Coefficient
```

**Implementation**:
```python
# Apply sigmoid to convert logits to probabilities
predictions = torch.sigmoid(predictions)  # (B, 1, H, W) → [0, 1]

# Flatten spatial dimensions
predictions = predictions.view(-1)  # (B×H×W,)
targets = targets.view(-1)          # (B×H×W,)

# Calculate intersection and union
intersection = (predictions * targets).sum()
dice = (2 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

return 1 - dice
```

**Why Dice Loss?**
- Good for imbalanced datasets (water vs non-water)
- Focuses on overlap, not pixel-wise accuracy
- Smooth parameter prevents division by zero

**Visual Example**:
```
Prediction: ████░░░░
Target:     ██████░░

Intersection: ████ (4 pixels)
Prediction sum: 4 pixels
Target sum: 6 pixels

Dice = 2×4 / (4+6) = 8/10 = 0.8
Loss = 1 - 0.8 = 0.2
```

### 2. Combined Loss (`CombinedLoss`)

```python
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
```

**Purpose**: Combines Binary Cross-Entropy and Dice Loss for better training.

**Formula**:
```
Total Loss = 0.5 × BCE Loss + 0.5 × Dice Loss
```

**Why Combine?**
- **BCE**: Good for pixel-wise accuracy
- **Dice**: Good for overall overlap
- Together: Best of both worlds

**BCE Loss**:
```python
BCE = -[y×log(p) + (1-y)×log(1-p)]

Where:
y = target (0 or 1)
p = prediction probability (0 to 1)
```

**Example**:
```
True water pixel (y=1):
  Predicted 0.9: BCE = -log(0.9) = 0.105 (small loss ✓)
  Predicted 0.3: BCE = -log(0.3) = 1.204 (large loss ✗)

True land pixel (y=0):
  Predicted 0.1: BCE = -log(0.9) = 0.105 (small loss ✓)
  Predicted 0.7: BCE = -log(0.3) = 1.204 (large loss ✗)
```

---

## Trainer Class

### Initialization

```python
class Trainer:
    def __init__(self, model, train_loader, config):
```

**Purpose**: Manages the complete training process.

**Components Set Up**:

**1. Loss Function**:
```python
self.criterion = CombinedLoss()
```

**2. Optimizer** (AdamW):
```python
self.optimizer = optim.AdamW(
    self.model.parameters(),
    lr=3e-4,           # Learning rate
    weight_decay=0.01  # L2 regularization
)
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Better weight decay than Adam
- State-of-the-art for transformers

**3. Learning Rate Scheduler**:
```python
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer,
    mode='min',       # Reduce when loss stops decreasing
    factor=0.5,       # Multiply LR by 0.5
    patience=10,      # Wait 10 epochs before reducing
    min_lr=1e-6       # Don't go below 0.000001
)
```

**Scheduler Behavior**:
```
Epoch 1-10:  LR = 0.0003 (loss decreasing)
Epoch 11-20: LR = 0.0003 (loss still decreasing)
Epoch 21:    LR = 0.00015 (loss plateaued, reduce!)
Epoch 31:    LR = 0.000075 (reduce again if needed)
```

### Training Loop

```python
def train(self, num_epochs=100):
```

**Complete Training Process**:

```python
for epoch in range(num_epochs):
    # 1. Train one epoch
    epoch_loss = self._train_epoch(epoch, num_epochs)
    
    # 2. Adjust learning rate if loss plateaus
    self.scheduler.step(epoch_loss)
    
    # 3. Save history
    self.history['train_loss'].append(epoch_loss)
    self.history['learning_rate'].append(current_lr)
    
    # 4. Save best model if improved
    if epoch_loss < self.best_loss:
        self.best_loss = epoch_loss
        self._save_checkpoint('best')
```

### Single Epoch Training

```python
def _train_epoch(self, epoch, total_epochs):
```

**Batch-by-Batch Process**:

```python
for images, masks in train_loader:
    # 1. Move data to GPU
    images = images.to(device)  # (Batch, 2, 512, 512)
    masks = masks.to(device)    # (Batch, 1, 512, 512)
    
    # 2. Forward pass
    outputs = model(images)     # (Batch, 1, 512, 512)
    loss = criterion(outputs, masks)
    
    # 3. Backward pass
    optimizer.zero_grad()       # Clear old gradients
    loss.backward()             # Compute new gradients
    optimizer.step()            # Update weights
    
    # 4. Track progress
    total_loss += loss.item()
```

**What Happens During Backpropagation?**

1. **Forward Pass**: Input → Model → Output → Loss
2. **Backward Pass**: Loss → Gradients → Update weights

```
Example weight update:
weight_new = weight_old - learning_rate × gradient

If gradient is positive → decrease weight
If gradient is negative → increase weight
```

### Model Checkpointing

```python
def _save_checkpoint(self, checkpoint_type='best'):
```

**What Gets Saved**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),  # All weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'best_loss': self.best_loss,
    'history': self.history,  # Loss curves
    'config': {  # Model architecture params
        'image_size': 512,
        'patch_size': 16,
        'embed_dim': 768,
        ...
    }
}
```

**Why Save Optimizer State?**
- Can resume training from checkpoint
- Preserves momentum and learning rate adaptations

### Training History Visualization

```python
def plot_history(self):
```

**Creates Two Plots**:

**1. Loss Curve**:
```
Training Loss
   ^
   |     ╱╲
4.0|    ╱  ╲___
   |   ╱       ╲___
2.0|  ╱            ╲___  ← Best: 1.2
   | ╱                  ╲___
0.0|______________________|___>
   0   20   40   60   80  100  Epoch
```

**2. Learning Rate Schedule**:
```
Learning Rate (log scale)
   ^
1e-3|████████████╮
    |             ╰╮
1e-4|              ╰████╮
    |                   ╰╮
1e-5|                    ╰████
    |________________________>
    0   20   40   60   80  100
```

---

## Predictor Class

### Initialization

```python
class WaterSegmentationPredictor:
    def __init__(self, model, device='cpu', threshold=0.5):
```

**Purpose**: Handles inference on new SAR images.

### Prediction Pipeline

```python
def predict(self, image_path):
```

**Complete Process**:

```python
# 1. Load and normalize image
with rasterio.open(image_path) as src:
    image = src.read()  # (2, H, W)
    profile = src.profile  # Geospatial metadata

# 2. Normalize to [0, 1]
for i in range(2):
    band_min, band_max = image[i].min(), image[i].max()
    normalized[i] = (image[i] - band_min) / (band_max - band_min)

# 3. Resize to 512×512
image_tensor = F.interpolate(
    torch.from_numpy(normalized).unsqueeze(0),
    size=(512, 512),
    mode='bilinear'
)

# 4. Run model inference (no gradient computation)
with torch.no_grad():
    output = model(image_tensor)  # (1, 1, 512, 512) logits
    probability = torch.sigmoid(output)  # → [0, 1] probabilities

# 5. Resize back to original image size
probability = zoom(probability, (original_height/512, original_width/512))

# 6. Apply threshold
mask = (probability > 0.5).astype(np.uint8)  # Binary mask
```

**Output**:
- `mask`: Binary water/non-water (0 or 1)
- `probability`: Confidence map (0.0 to 1.0)
- `profile`: Geospatial metadata for saving results

### Visualization

```python
def visualize_prediction(self, image_path, save_path=None):
```

**Creates 4-Panel Visualization**:

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│             │             │             │             │
│  VV Band    │  VH Band    │ Probability │ Binary Mask │
│  (grayscale)│ (grayscale) │  (viridis)  │   (blue)    │
│             │             │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘
                                                ↑
                                         Shows 45.2% water
```

**Panel Details**:
1. **VV Band**: First SAR channel (vertical-vertical polarization)
2. **VH Band**: Second SAR channel (vertical-horizontal polarization)
3. **Probability Map**: Model confidence (0=certain non-water, 1=certain water)
4. **Binary Mask**: Final prediction after thresholding

---

## Data Utilities

### Finding Image-Shapefile Pairs

```python
@staticmethod
def find_image_shapefile_pairs(images_folder, labels_folder):
```

**Purpose**: Match SAR images with their corresponding shapefile labels.

**Matching Logic**:

```python
# Example filenames:
# Image: "scene_001.tif"
# Shapefile: "scene_001_water.shp"

for img_file in image_files:
    # Clean image name
    clean_name = img_file.replace(".tif", "")  # "scene_001"
    
    # Find matching shapefile
    for shp_name in shapefile_dict:
        if clean_name in shp_name:  # "scene_001" in "scene_001_water"
            matched_pairs.append({
                'image': '/path/to/scene_001.tif',
                'shapefile': '/path/to/scene_001_water.shp',
                'name': 'scene_001.tif'
            })
```

**Why This Matters**:
- Ensures each image has its corresponding label
- Prevents training on unpaired data
- Handles variations in naming conventions

### Model Checkpointing

**Saving**:
```python
@staticmethod
def save_model_checkpoint(model, path, metadata=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': 'ViTSegmentationModel',
        'config': {...}  # Architecture parameters
    }
    torch.save(checkpoint, path)
```

**Loading**:
```python
@staticmethod
def load_model_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    
    # Reconstruct model with saved config
    model = ViTSegmentationModel(
        image_size=checkpoint['config']['image_size'],
        patch_size=checkpoint['config']['patch_size'],
        ...
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model
```

**Why Save Config?**
- Can reconstruct exact model architecture
- No need to remember hyperparameters
- Ensures reproducibility

---

## Main Pipeline

### Complete Training Flow

```python
def main():
```

**Step-by-Step Process**:

**1. Setup**
```python
# Get file paths
paths = Config.get_paths()

# Find matching data pairs
matched_pairs = DataManager.find_image_shapefile_pairs(
    paths['images'],
    paths['labels']
)
```

**2. Create Dataset**
```python
dataset = SARSegmentationDataset(
    image_paths=[p['image'] for p in matched_pairs],
    shapefile_paths=[p['shapefile'] for p in matched_pairs],
    target_size=512,
    augment=True,      # Enable augmentation
    augment_prob=0.85  # Apply 85% of the time
)
```

**3. Create DataLoader**
```python
train_loader = DataLoader(
    dataset,
    batch_size=4,      # Process 4 images at a time
    shuffle=True,      # Randomize order each epoch
    num_workers=2,     # Parallel data loading
    pin_memory=True    # Faster GPU transfer
)
```

**DataLoader Benefits**:
- **Batching**: Process multiple images simultaneously
- **Shuffling**: Prevents learning order-dependent patterns
- **Parallel Loading**: CPU loads next batch while GPU processes current batch
- **Pin Memory**: Faster data transfer to GPU

**4. Initialize Model**
```python
model = ViTSegmentationModel(
    image_size=512,
    patch_size=16,
    in_channels=2,     # VV and VH bands
    num_classes=1,     # Binary segmentation
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_ratio=4,
    dropout=0.1
)
```

**Model Summary**:
```
Total Parameters: ~86 million
- Patch Embedding: ~2 million
- Transformer Encoder: ~80 million
- Segmentation Head: ~4 million

GPU Memory: ~4-6 GB (batch size 4)
```

**5. Train**
```python
trainer = Trainer(model, train_loader, Config)
history = trainer.train()
```

**Training Output**:
```
Epoch [1/100] - Loss: 0.8234 - LR: 3.00e-04
Epoch [2/100] - Loss: 0.7123 - LR: 3.00e-04
...
   ✅ New best model saved! (loss: 0.4521)
...
Epoch [100/100] - Loss: 0.1234 - LR: 1.50e-05
```

**6. Save and Evaluate**
```python
# Plot training curves
trainer.plot_history()

# Save final model
DataManager.save_model_checkpoint(
    model,
    paths['final_model'],
    metadata={'best_loss': trainer.best_loss, ...}
)

# Test on sample image
predictor = WaterSegmentationPredictor(model, ...)
predictor.visualize_prediction(matched_pairs[0]['image'])
```

---

## Usage Examples

### Example 1: Basic Training

```python
# Simply run the script
python sar_vit_water_segmentation.py
```

### Example 2: Custom Configuration

```python
# Modify hyperparameters
Config.BATCH_SIZE = 8  # Larger batch (needs more GPU memory)
Config.NUM_EPOCHS = 150  # Train longer
Config.LEARNING_RATE = 1e-4  # Lower learning rate

# Run training
main()
```

### Example 3: Inference on New Images

```python
from sar_vit_water_segmentation import (
    DataManager,
    WaterSegmentationPredictor,
    Config
)

# Load trained model
model = DataManager.load_model_checkpoint(
    '/path/to/vit_best_model.pth',
    device='cuda'
)

# Create predictor
predictor = WaterSegmentationPredictor(
    model,
    device='cuda',
    threshold=0.5
)

# Predict on new image
mask, prob, profile = predictor.predict('/path/to/new_image.tif')

# Visualize
predictor.visualize_prediction('/path/to/new_image.tif', 
                                save_path='output.png')
```

### Example 4: Batch Prediction

```python
import glob
import rasterio

# Get all test images
test_images = glob.glob('/path/to/test_images/*.tif')

# Process each image
for img_path in test_images:
    mask, prob, profile = predictor.predict(img_path)
    
    # Save mask as GeoTIFF
    output_path = img_path.replace('.tif', '_water_mask.tif')
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        transform=profile['transform'],
        crs=profile['crs']
    ) as dst:
        dst.write(mask, 1)
    
    print(f"Processed: {img_path}")
```

---

## Key Advantages of This Implementation

### 1. Vision Transformer Benefits
- **Global Context**: Self-attention captures long-range dependencies
- **No Convolution Bias**: Learns optimal feature extraction
- **Scalability**: Can handle larger images with more patches

### 2. Robust Training
- **Combined Loss**: BCE + Dice for balanced optimization
- **Learning Rate Scheduling**: Automatic adjustment when plateauing
- **Comprehensive Augmentation**: Increases effective dataset size 10-20×

### 3. Production Ready
- **Modular Design**: Easy to modify components
- **Checkpointing**: Can resume training or deploy models
- **Metadata Preservation**: Geospatial info maintained through pipeline
- **Visualization**: Built-in prediction visualization

### 4. Memory Efficient
- **Patch-Based Processing**: Handles large images efficiently
- **Gradient Checkpointing**: Can be added for even larger models
- **Mixed Precision**: Can use torch.cuda.amp for 2× speedup

---

## Common Issues and Solutions

### Issue 1: Out of Memory

**Solution**:
```python
Config.BATCH_SIZE = 2  # Reduce batch size
Config.IMAGE_SIZE = 256  # Use smaller images
Config.EMBED_DIM = 512  # Use smaller model
```

### Issue 2: Slow Training

**Solution**:
```python
Config.NUM_WORKERS = 4  # More parallel data loading
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

### Issue 3: Poor Convergence

**Solution**:
```python
Config.LEARNING_RATE = 1e-5  # Try lower LR
Config.AUGMENT_PROB = 0.5  # Less aggressive augmentation
# Add validation split to monitor overfitting
```

### Issue 4: Images Don't Match Shapefiles

**Solution**:
```python
# Check naming convention
matched_pairs = DataManager.find_image_shapefile_pairs(...)
print(f"Found {len(matched_pairs)} pairs")

# Manually inspect
for pair in matched_pairs:
    print(pair['name'], pair['shapefile'])
```

---

## Performance Metrics

### Expected Results (after 100 epochs):
- **Training Loss**: 0.1 - 0.3 (combined BCE + Dice)
- **Dice Coefficient**: 0.85 - 0.95
- **Inference Time**: ~0.5-1 second per 512×512 image (GPU)
- **Memory Usage**: 4-6 GB GPU, 8-12 GB RAM

### Compared to DeepLabV3:
- **Better**: Global context modeling, long-range dependencies
- **Slower**: ~2× slower training (more parameters)
- **More Data**: Benefits from larger datasets

---

## Summary

This implementation provides:
1. ✅ Complete Vision Transformer for SAR water segmentation
2. ✅ 512×512 input with 16×16 patches
3. ✅ Comprehensive data augmentation during training
4. ✅ Shuffled DataLoader for robust training
5. ✅ Combined BCE + Dice loss for balanced optimization
6. ✅ Learning rate scheduling for optimal convergence
7. ✅ Model checkpointing and inference utilities
8. ✅ Visualization and geospatial metadata preservation

The code is production-ready, well-documented, and easily modifiable for different datasets or requirements.
