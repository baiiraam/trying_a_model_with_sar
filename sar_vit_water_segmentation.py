"""
SAR Water Segmentation with Vision Transformer
===============================================
Complete pipeline for water body segmentation from 2-band SAR imagery using ViT.

This implementation uses a Vision Transformer encoder with a segmentation head
for pixel-level water classification.

Author: Claude
Date: 2026-04-18
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio import features
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom

# ============================================
# Configuration
# ============================================


class Config:
    """Central configuration for the training pipeline"""

    # Paths
    BASE_PATH = "/content/drive/MyDrive/SAR_water_images_and_labels"
    IMAGES_FOLDER = "images"
    LABELS_FOLDER = "labels"

    # Model Architecture
    NUM_CLASSES = 1  # Binary segmentation (water vs non-water)
    INPUT_BANDS = 2  # VV and VH bands
    IMAGE_SIZE = 512  # Input image size (512x512)
    PATCH_SIZE = 16  # Size of patches for ViT
    EMBED_DIM = 768  # Embedding dimension
    NUM_HEADS = 12  # Number of attention heads
    NUM_LAYERS = 12  # Number of transformer layers
    MLP_RATIO = 4  # MLP expansion ratio
    DROPOUT = 0.1  # Dropout rate

    # Training Hyperparameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    NUM_WORKERS = 2

    # Data Augmentation
    AUGMENT_PROB = 0.85  # Probability of applying augmentation
    FLIP_PROB = 0.5
    ROTATION_PROB = 0.7
    BRIGHTNESS_PROB = 0.4
    NOISE_PROB = 0.3
    ELASTIC_PROB = 0.2

    # Scheduler
    WARMUP_EPOCHS = 5
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Inference
    INFERENCE_THRESHOLD = 0.5

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_paths(cls):
        """Get all path configurations"""
        return {
            "base": cls.BASE_PATH,
            "images": os.path.join(cls.BASE_PATH, cls.IMAGES_FOLDER),
            "labels": os.path.join(cls.BASE_PATH, cls.LABELS_FOLDER),
            "best_model": os.path.join(cls.BASE_PATH, "vit_best_model.pth"),
            "final_model": os.path.join(cls.BASE_PATH, "vit_final_model.pth"),
        }


# ============================================
# Vision Transformer Components
# ============================================


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.

    Args:
        image_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 2,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding via convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        # x: (B, 2, 512, 512) -> (B, 768, 32, 32)
        x = self.projection(x)
        # (B, 768, 32, 32) -> (B, 768, 1024) -> (B, 1024, 768)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    Feed-forward network.

    Args:
        in_features: Input dimension
        hidden_features: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    Args:
        image_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 2,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class SegmentationHead(nn.Module):
    """
    Segmentation decoder head for ViT.

    Upsamples the patch embeddings back to image resolution.

    Args:
        embed_dim: Embedding dimension from encoder
        num_classes: Number of output classes
        image_size: Output image size
        patch_size: Patch size used in encoder
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1,
        image_size: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()

        self.num_patches_per_side = image_size // patch_size
        self.patch_size = patch_size

        # Decoder layers with progressive upsampling
        self.decoder = nn.Sequential(
            # (B, 768, 32, 32) -> (B, 512, 32, 32)
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # (B, 512, 32, 32) -> (B, 512, 64, 64)
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # (B, 512, 64, 64) -> (B, 256, 128, 128)
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # (B, 256, 128, 128) -> (B, 128, 256, 256)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (B, 128, 256, 256) -> (B, 64, 512, 512)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final classification layer
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            (B, num_classes, H, W)
        """
        B, N, C = x.shape
        H = W = self.num_patches_per_side

        # Reshape to spatial format
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decode
        x = self.decoder(x)

        return x


class ViTSegmentationModel(nn.Module):
    """
    Complete Vision Transformer model for semantic segmentation.

    Args:
        image_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 2,
        num_classes: int = 1,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.decoder = SegmentationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        # Encode
        features = self.encoder(x)

        # Decode
        output = self.decoder(features)

        return output

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# Dataset
# ============================================


class SARSegmentationDataset(Dataset):
    """
    Dataset for 2-band SAR images with shapefile labels.

    Args:
        image_paths: List of paths to SAR image files
        shapefile_paths: List of paths to corresponding shapefile labels
        target_size: Target size for resizing (height, width)
        augment: Whether to apply data augmentation
        augment_prob: Probability of applying augmentation
    """

    def __init__(
        self,
        image_paths: List[str],
        shapefile_paths: List[str],
        target_size: int = 512,
        augment: bool = False,
        augment_prob: float = 0.8,
    ):
        assert len(image_paths) == len(
            shapefile_paths
        ), "Number of images and shapefiles must match"

        self.image_paths = image_paths
        self.shapefile_paths = shapefile_paths
        self.target_size = target_size
        self.augment = augment
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            image: Tensor of shape (2, H, W)
            mask: Tensor of shape (1, H, W)
        """
        # Load image and mask
        image = self._load_sar_image(self.image_paths[idx])
        mask = self._shapefile_to_mask(self.shapefile_paths[idx], self.image_paths[idx])

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()

        # Resize to target size
        image_tensor = self._resize_image(image_tensor)
        mask_tensor = self._resize_mask(mask_tensor)

        # Apply augmentations
        if self.augment and random.random() < self.augment_prob:
            image_tensor, mask_tensor = self._apply_augmentations(
                image_tensor, mask_tensor
            )

        # Add channel dimension to mask
        mask_tensor = mask_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)

        return image_tensor, mask_tensor

    def _load_sar_image(self, image_path: str) -> np.ndarray:
        """
        Load and normalize SAR image to 2 bands.

        Returns:
            Array of shape (2, H, W) normalized to [0, 1]
        """
        with rasterio.open(image_path) as src:
            img = src.read()

        # Ensure exactly 2 bands
        if img.shape[0] > 2:
            img = img[:2]
        elif img.shape[0] == 1:
            img = np.repeat(img, 2, axis=0)

        # Normalize each band independently to [0, 1]
        normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(2):
            band_min, band_max = img[i].min(), img[i].max()
            if band_max > band_min:
                normalized[i] = (img[i] - band_min) / (band_max - band_min)
            else:
                normalized[i] = img[i]

        return normalized

    def _shapefile_to_mask(
        self, shapefile_path: str, reference_image_path: str
    ) -> np.ndarray:
        """
        Convert shapefile polygons to binary raster mask.

        Returns:
            Binary mask array of shape (H, W)
        """
        # Get spatial reference from image
        with rasterio.open(reference_image_path) as src:
            transform = src.transform
            shape = (src.height, src.width)

        # Read shapefile
        gdf = gpd.read_file(shapefile_path)

        # Create binary mask
        if len(gdf) > 0:
            geometries = [(geom, 1) for geom in gdf.geometry]
            mask = features.rasterize(
                geometries, out_shape=shape, transform=transform, fill=0, dtype=np.uint8
            )
        else:
            mask = np.zeros(shape, dtype=np.uint8)

        return mask

    def _resize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize image to target size using bilinear interpolation"""
        # Add batch dimension
        image = image.unsqueeze(0)
        # Resize
        image = F.interpolate(
            image,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        # Remove batch dimension
        return image.squeeze(0)

    def _resize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Resize mask to target size using nearest neighbor"""
        # Add batch and channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        # Resize
        mask = F.interpolate(
            mask, size=(self.target_size, self.target_size), mode="nearest"
        )
        # Remove batch and channel dimensions
        return mask.squeeze(0).squeeze(0)

    def _apply_augmentations(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to image and mask.

        Args:
            image: (2, H, W)
            mask: (H, W)
        Returns:
            Augmented image and mask
        """
        image_np = image.numpy()
        mask_np = mask.numpy()

        # Random horizontal flip
        if random.random() < Config.FLIP_PROB:
            image_np = np.flip(image_np, axis=2).copy()
            mask_np = np.flip(mask_np, axis=1).copy()

        # Random vertical flip
        if random.random() < Config.FLIP_PROB:
            image_np = np.flip(image_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=0).copy()

        # Random rotation (90, 180, 270 degrees)
        if random.random() < Config.ROTATION_PROB:
            k = random.randint(1, 3)
            image_np = np.rot90(image_np, k=k, axes=(1, 2)).copy()
            mask_np = np.rot90(mask_np, k=k, axes=(0, 1)).copy()

        # Random brightness adjustment
        if random.random() < Config.BRIGHTNESS_PROB:
            factor = random.uniform(0.8, 1.2)
            image_np = np.clip(image_np * factor, 0, 1)

        # Random noise
        if random.random() < Config.NOISE_PROB:
            noise = np.random.normal(0, 0.02, image_np.shape)
            image_np = np.clip(image_np + noise, 0, 1)

        return torch.from_numpy(image_np), torch.from_numpy(mask_np)


# ============================================
# Loss Functions
# ============================================


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1, H, W) logits
            targets: (B, 1, H, W) binary targets
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================
# Trainer
# ============================================


class Trainer:
    """
    Training manager for the segmentation model.

    Args:
        model: Model to train
        train_loader: Training data loader
        config: Configuration object
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, config: Config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.config = config
        self.device = config.DEVICE

        # Loss function
        self.criterion = CombinedLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=True,
        )

        # Training history
        self.history = {"train_loss": [], "learning_rate": []}

        self.best_loss = float("inf")

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.NUM_EPOCHS

        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            # Train one epoch
            epoch_loss = self._train_epoch(epoch, num_epochs)

            # Update learning rate
            self.scheduler.step(epoch_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Save history
            self.history["train_loss"].append(epoch_loss)
            self.history["learning_rate"].append(current_lr)

            # Save best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint("best")
                print(f"   ✅ New best model saved! (loss: {epoch_loss:.4f})")

            # Print progress
            print(
                f"   Epoch [{epoch+1}/{num_epochs}] - "
                f"Loss: {epoch_loss:.4f} - "
                f"LR: {current_lr:.2e}"
            )

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}\n")

        return self.history

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False
        )

        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def _save_checkpoint(self, checkpoint_type: str = "best"):
        """Save model checkpoint"""
        paths = self.config.get_paths()
        save_path = (
            paths["best_model"] if checkpoint_type == "best" else paths["final_model"]
        )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
            "config": {
                "image_size": self.config.IMAGE_SIZE,
                "patch_size": self.config.PATCH_SIZE,
                "embed_dim": self.config.EMBED_DIM,
                "num_layers": self.config.NUM_LAYERS,
                "num_heads": self.config.NUM_HEADS,
                "num_classes": self.config.NUM_CLASSES,
                "in_channels": self.config.INPUT_BANDS,
            },
        }

        torch.save(checkpoint, save_path)

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(self.history["train_loss"], "b-", linewidth=2, label="Train Loss")
        axes[0].axhline(
            y=self.best_loss,
            color="r",
            linestyle="--",
            label=f"Best: {self.best_loss:.4f}",
        )
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1].plot(self.history["learning_rate"], "g-", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Learning Rate", fontsize=12)
        axes[1].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Training history saved to {save_path}")

        plt.show()


# ============================================
# Predictor
# ============================================


class WaterSegmentationPredictor:
    """
    Inference class for water segmentation.

    Args:
        model: Trained model
        device: Device to run inference on
        threshold: Classification threshold
    """

    def __init__(self, model: nn.Module, device: str = "cpu", threshold: float = 0.5):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold

    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Predict water mask for a single image.

        Args:
            image_path: Path to input SAR image

        Returns:
            mask: Binary mask (H, W)
            probability: Probability map (H, W)
            profile: Raster metadata
        """
        # Load image
        with rasterio.open(image_path) as src:
            image = src.read()
            profile = src.profile
            original_shape = (src.height, src.width)

        # Ensure 2 bands
        if image.shape[0] > 2:
            image = image[:2]
        elif image.shape[0] == 1:
            image = np.repeat(image, 2, axis=0)

        # Normalize
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(2):
            band_min, band_max = image[i].min(), image[i].max()
            if band_max > band_min:
                normalized[i] = (image[i] - band_min) / (band_max - band_min)
            else:
                normalized[i] = image[i]

        # Convert to tensor and resize
        image_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        image_tensor = F.interpolate(
            image_tensor,
            size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).squeeze().cpu().numpy()

        # Resize back to original size
        probability = zoom(
            probability,
            (
                original_shape[0] / Config.IMAGE_SIZE,
                original_shape[1] / Config.IMAGE_SIZE,
            ),
            order=1,
        )

        # Threshold
        mask = (probability > self.threshold).astype(np.uint8)

        return mask, probability, profile

    def visualize_prediction(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize prediction on an image.

        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        # Load original image for visualization
        with rasterio.open(image_path) as src:
            image = src.read()

        # Get prediction
        mask, probability, _ = self.predict(image_path)

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # VV band
        if image.shape[0] >= 1:
            axes[0].imshow(image[0], cmap="gray")
            axes[0].set_title("VV Band", fontsize=14, fontweight="bold")
            axes[0].axis("off")

        # VH band
        if image.shape[0] >= 2:
            axes[1].imshow(image[1], cmap="gray")
            axes[1].set_title("VH Band", fontsize=14, fontweight="bold")
            axes[1].axis("off")

        # Probability map
        im = axes[2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Water Probability", fontsize=14, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # Binary mask
        water_coverage = mask.sum() / mask.size * 100
        axes[3].imshow(mask, cmap="Blues")
        axes[3].set_title(
            f"Water Mask\n({water_coverage:.1f}% water)", fontsize=14, fontweight="bold"
        )
        axes[3].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Visualization saved to {save_path}")

        plt.show()

        print(f"\n💧 Water coverage: {water_coverage:.2f}%")


# ============================================
# Data Utilities
# ============================================


class DataManager:
    """Utility class for managing SAR image and shapefile data"""

    @staticmethod
    def find_image_shapefile_pairs(
        images_folder: str, labels_folder: str
    ) -> List[Dict[str, str]]:
        """
        Find matching image-shapefile pairs.

        Args:
            images_folder: Path to images folder
            labels_folder: Path to labels folder

        Returns:
            List of dictionaries with 'image', 'shapefile', and 'name' keys
        """
        # Get all image files
        image_files = [
            f
            for f in os.listdir(images_folder)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        print(f"Found {len(image_files)} image files")

        # Find all shapefiles
        shapefile_dict = {}
        for file in os.listdir(labels_folder):
            if file.endswith(".shp"):
                base_name = file.replace(".shp", "")
                shapefile_dict[base_name] = os.path.join(labels_folder, file)

        print(f"Found {len(shapefile_dict)} shapefile labels")

        # Match images with shapefiles
        matched_pairs = []
        for img_file in image_files:
            # Clean image filename
            clean_name = (
                img_file.replace("Копия ", "").replace(".tif", "").replace(".tiff", "")
            )

            # Try to find matching shapefile
            for shp_name, shp_path in shapefile_dict.items():
                if clean_name in shp_name or shp_name in clean_name:
                    matched_pairs.append(
                        {
                            "image": os.path.join(images_folder, img_file),
                            "shapefile": shp_path,
                            "name": img_file,
                        }
                    )
                    break

        print(f"\n✅ Found {len(matched_pairs)} matching image-shapefile pairs")
        for pair in matched_pairs[:5]:  # Show first 5
            print(f"  - {pair['name']}")
        if len(matched_pairs) > 5:
            print(f"  ... and {len(matched_pairs) - 5} more")

        return matched_pairs

    @staticmethod
    def save_model_checkpoint(
        model: nn.Module, path: str, metadata: Optional[Dict] = None
    ):
        """
        Save model with metadata for later inference.

        Args:
            model: Model to save
            path: Save path
            metadata: Optional metadata dictionary
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_class": "ViTSegmentationModel",
            "config": {
                "image_size": Config.IMAGE_SIZE,
                "patch_size": Config.PATCH_SIZE,
                "in_channels": Config.INPUT_BANDS,
                "num_classes": Config.NUM_CLASSES,
                "embed_dim": Config.EMBED_DIM,
                "num_layers": Config.NUM_LAYERS,
                "num_heads": Config.NUM_HEADS,
                "mlp_ratio": Config.MLP_RATIO,
                "dropout": Config.DROPOUT,
            },
        }

        if metadata:
            checkpoint.update(metadata)

        torch.save(checkpoint, path)
        print(f"✅ Model checkpoint saved to {path}")

    @staticmethod
    def load_model_checkpoint(path: str, device: str = "cpu") -> nn.Module:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded model in eval mode
        """
        checkpoint = torch.load(path, map_location=device)

        config = checkpoint.get("config", {})
        model = ViTSegmentationModel(
            image_size=config.get("image_size", 512),
            patch_size=config.get("patch_size", 16),
            in_channels=config.get("in_channels", 2),
            num_classes=config.get("num_classes", 1),
            embed_dim=config.get("embed_dim", 768),
            num_layers=config.get("num_layers", 12),
            num_heads=config.get("num_heads", 12),
            mlp_ratio=config.get("mlp_ratio", 4),
            dropout=config.get("dropout", 0.1),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        print(f"✅ Model loaded from {path}")
        print(f"   Image size: {config.get('image_size', 'unknown')}")
        print(f"   Patch size: {config.get('patch_size', 'unknown')}")
        print(f"   Embed dim: {config.get('embed_dim', 'unknown')}")
        print(f"   Layers: {config.get('num_layers', 'unknown')}")

        return model


# ============================================
# Main Training Pipeline
# ============================================


def main():
    """Main training pipeline"""

    print("=" * 60)
    print("SAR Water Segmentation with Vision Transformer")
    print("=" * 60)

    # Get paths
    paths = Config.get_paths()

    # Find matching image-shapefile pairs
    matched_pairs = DataManager.find_image_shapefile_pairs(
        paths["images"], paths["labels"]
    )

    if len(matched_pairs) == 0:
        print("\n❌ No matching pairs found! Please check your folder structure.")
        return

    # Create dataset
    print(f"\n{'='*60}")
    print("Creating dataset with augmentation...")
    print(f"{'='*60}")
    dataset = SARSegmentationDataset(
        image_paths=[p["image"] for p in matched_pairs],
        shapefile_paths=[p["shapefile"] for p in matched_pairs],
        target_size=Config.IMAGE_SIZE,
        augment=True,
        augment_prob=Config.AUGMENT_PROB,
    )

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    print(f"   Training samples: {len(dataset)}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Batches per epoch: {len(train_loader)}")
    print(f"   Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")

    # Initialize model
    print(f"\n{'='*60}")
    print("Initializing Vision Transformer model...")
    print(f"{'='*60}")
    model = ViTSegmentationModel(
        image_size=Config.IMAGE_SIZE,
        patch_size=Config.PATCH_SIZE,
        in_channels=Config.INPUT_BANDS,
        num_classes=Config.NUM_CLASSES,
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        mlp_ratio=Config.MLP_RATIO,
        dropout=Config.DROPOUT,
    )

    print(f"   Model: Vision Transformer")
    print(
        f"   Input: {Config.INPUT_BANDS} bands ({Config.IMAGE_SIZE}x{Config.IMAGE_SIZE})"
    )
    print(f"   Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"   Embedding dim: {Config.EMBED_DIM}")
    print(f"   Layers: {Config.NUM_LAYERS}")
    print(f"   Attention heads: {Config.NUM_HEADS}")
    print(f"   Total parameters: {model.get_num_parameters():,}")

    # Initialize trainer
    trainer = Trainer(model, train_loader, Config)

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    print(f"{'='*60}")
    history = trainer.train()

    # Plot training history
    print("\nPlotting training history...")
    trainer.plot_history()

    # Save final model
    print(f"\nSaving final model...")
    DataManager.save_model_checkpoint(
        model,
        paths["final_model"],
        metadata={
            "best_loss": trainer.best_loss,
            "epochs_trained": Config.NUM_EPOCHS,
            "final_loss": history["train_loss"][-1],
        },
    )

    # Test prediction on first image
    print(f"\n{'='*60}")
    print("Testing prediction on sample image...")
    print(f"{'='*60}")
    predictor = WaterSegmentationPredictor(
        model, device=Config.DEVICE, threshold=Config.INFERENCE_THRESHOLD
    )
    predictor.visualize_prediction(matched_pairs[0]["image"])

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


# ============================================
# Inference Script
# ============================================


def run_inference(model_path: str, image_path: str, output_path: Optional[str] = None):
    """
    Run inference on a single image.

    Args:
        model_path: Path to saved model checkpoint
        image_path: Path to input SAR image
        output_path: Optional path to save visualization
    """
    # Load model
    model = DataManager.load_model_checkpoint(model_path, Config.DEVICE)

    # Create predictor
    predictor = WaterSegmentationPredictor(
        model, device=Config.DEVICE, threshold=Config.INFERENCE_THRESHOLD
    )

    # Predict and visualize
    predictor.visualize_prediction(image_path, output_path)


if __name__ == "__main__":
    main()
