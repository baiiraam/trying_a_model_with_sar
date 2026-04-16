"""
Water Segmentation Inference with Shapefile Output
"""

import os
import json
import torch
import numpy as np
import rasterio
from rasterio import features
from datetime import datetime
from pathlib import Path
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import pandas as pd


class WaterSegmentationInference:
    def __init__(self, inference_name=None):
        """Initialize inference run"""

        # Create timestamped inference folder
        if inference_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.inference_name = f"inference_{timestamp}"
        else:
            self.inference_name = inference_name

        # Setup paths
        self.base_dir = Path.cwd()
        self.inference_dir = self.base_dir / "inferences" / self.inference_name
        self.predictions_dir = self.inference_dir / "predictions"
        self.viz_dir = self.inference_dir / "visualizations"
        self.metrics_dir = self.inference_dir / "metrics"

        # Create directories
        for dir_path in [self.predictions_dir, self.viz_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load model
        self.load_model()

        # Save inference config
        self.save_config()

        print(f"\n📁 Inference directory: {self.inference_dir}")
        print(f"   Predictions will be saved to: {self.predictions_dir}")

    def load_model(self):
        """Load the segmentation model"""
        print("\n🤖 Loading model...")
        self.model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        self.model.eval()
        print(f"✅ Model loaded: {self.model_id}")

    def save_config(self):
        """Save inference configuration"""
        config = {
            "inference_name": self.inference_name,
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "water_class_index": 13,
            "device": "cpu",
        }

        with open(self.inference_dir / "inference_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load_image(self, image_path):
        """Load and preprocess .tif image"""
        print(f"\n📂 Loading image: {image_path}")

        with rasterio.open(image_path) as src:
            image = src.read()
            meta = src.meta

            # Convert to RGB
            if image.shape[0] >= 3:
                rgb = image[:3].transpose(1, 2, 0)
            else:
                rgb = np.stack([image[0], image[0], image[0]], axis=2)

            # Normalize
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = (rgb * 255).astype(np.uint8)

        print(f"   Image shape: {image.shape}")
        print(f"   RGB shape: {rgb.shape}")

        return image, rgb, meta

    def predict(self, rgb_image):
        """Run inference on RGB image"""
        print("\n🔮 Running inference...")

        inputs = self.processor(images=rgb_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Upsample to original size
            prediction = (
                torch.nn.functional.interpolate(
                    outputs.logits,
                    size=rgb_image.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )
                .argmax(dim=1)
                .squeeze()
                .cpu()
                .numpy()
            )

        print(f"✅ Prediction complete")
        print(f"   Unique classes: {np.unique(prediction)}")

        return prediction

    def mask_to_shapefile(self, mask, transform, crs, output_path, min_area=100):
        """Convert binary mask to shapefile (polygons)"""

        # Find polygons in the mask
        shapes = features.shapes(
            mask.astype(np.uint8), mask=mask.astype(np.uint8), transform=transform
        )

        # Collect polygons
        polygons = []
        values = []

        for geom, value in shapes:
            # Convert to shapely polygon
            polygon = Polygon(geom["coordinates"][0])

            # Filter by minimum area
            if polygon.area >= min_area:
                # Simplify polygon to reduce complexity
                simplified = polygon.simplify(1.0)
                if simplified.is_valid:
                    polygons.append(simplified)
                    values.append(value)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                "id": range(len(polygons)),
                "class": values,
                "area_m2": [
                    p.area * (abs(transform[0]) * abs(transform[4])) for p in polygons
                ],
                "geometry": polygons,
            },
            crs=crs,
        )

        # Save shapefile
        gdf.to_file(output_path)
        print(f"   Saved {len(polygons)} polygons to shapefile")

        return gdf

    def run_inference_on_image(self, image_path, ground_truth_path=None):
        """Complete inference pipeline for a single image"""

        # Get image name
        image_name = Path(image_path).stem

        # Load image
        image_array, rgb_image, meta = self.load_image(image_path)

        # Run prediction
        prediction = self.predict(rgb_image)

        # Extract water mask (class 13)
        water_mask = (prediction == 13).astype(np.uint8)
        water_percentage = (water_mask.sum() / water_mask.size) * 100

        print(f"\n💧 Water coverage: {water_percentage:.2f}%")

        # Save water mask as GeoTIFF
        mask_tif_path = self.predictions_dir / f"{image_name}_water_mask.tif"
        with rasterio.open(
            mask_tif_path,
            "w",
            driver="GTiff",
            height=water_mask.shape[0],
            width=water_mask.shape[1],
            count=1,
            dtype=water_mask.dtype,
            crs=meta["crs"],
            transform=meta["transform"],
        ) as dst:
            dst.write(water_mask, 1)
        print(f"💾 Saved mask GeoTIFF: {mask_tif_path}")

        # Convert to shapefile
        shapefile_path = self.predictions_dir / f"{image_name}_water_mask.shp"
        gdf = self.mask_to_shapefile(
            water_mask, meta["transform"], meta["crs"], shapefile_path
        )

        # Save PNG visualization
        png_path = self.predictions_dir / f"{image_name}_water_mask.png"
        self.save_visualization(rgb_image, water_mask, png_path)

        # Evaluate against ground truth if provided
        metrics = None
        if ground_truth_path and Path(ground_truth_path).exists():
            metrics = self.evaluate_against_ground_truth(
                water_mask, ground_truth_path, meta, image_name
            )

        return {
            "image_name": image_name,
            "water_mask_path": str(mask_tif_path),
            "shapefile_path": str(shapefile_path),
            "water_coverage": water_percentage,
            "metrics": metrics,
        }

    def save_visualization(self, rgb_image, water_mask, save_path):
        """Save water mask visualization"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Water mask
        axes[1].imshow(water_mask, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title(f"Water Mask ({water_mask.sum()/water_mask.size*100:.1f}%)")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(rgb_image)
        axes[2].imshow(water_mask, alpha=0.5, cmap="Blues")
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"💾 Saved visualization: {save_path}")

    def evaluate_against_ground_truth(
        self, predicted_mask, gt_shapefile_path, meta, image_name
    ):
        """Compare predicted mask with ground truth shapefile"""
        import geopandas as gpd

        # Load ground truth
        gt_gdf = gpd.read_file(gt_shapefile_path)

        # Rasterize ground truth to match prediction size
        from rasterio import features

        gt_mask = np.zeros_like(predicted_mask, dtype=np.uint8)

        shapes = ((geom, 1) for geom in gt_gdf.geometry)
        rasterized = features.rasterize(
            shapes=shapes,
            out_shape=predicted_mask.shape,
            transform=meta["transform"],
            fill=0,
            dtype=np.uint8,
        )

        # Calculate metrics
        intersection = np.logical_and(predicted_mask, rasterized).sum()
        union = np.logical_or(predicted_mask, rasterized).sum()

        iou = intersection / union if union > 0 else 0
        precision = (
            intersection / predicted_mask.sum() if predicted_mask.sum() > 0 else 0
        )
        recall = intersection / rasterized.sum() if rasterized.sum() > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics = {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "gt_water_pixels": int(rasterized.sum()),
            "pred_water_pixels": int(predicted_mask.sum()),
            "intersection_pixels": int(intersection),
        }

        # Save metrics
        metrics_path = self.metrics_dir / f"{image_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n📊 Metrics for {image_name}:")
        print(f"   IoU: {iou:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")

        return metrics


def main():
    """Run inference on all images"""

    # Initialize inference run
    inference = WaterSegmentationInference(inference_name="first_inference")

    # Define paths
    images_dir = Path("data/raw/images")
    labels_dir = Path("data/raw/labels")

    # Find all .tif images
    tif_files = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff"))

    if not tif_files:
        print(f"❌ No .tif files found in {images_dir}")
        return

    print(f"\n📸 Found {len(tif_files)} images to process")

    results = []

    # Process each image
    for tif_path in tif_files:
        print("\n" + "=" * 60)
        print(f"Processing: {tif_path.name}")
        print("=" * 60)

        # Find corresponding ground truth shapefile
        gt_path = labels_dir / f"{tif_path.stem}.shp"
        if not gt_path.exists():
            gt_path = None
            print("⚠️ No ground truth found for this image")

        # Run inference
        result = inference.run_inference_on_image(tif_path, gt_path)
        results.append(result)

    # Save summary
    summary_path = inference.inference_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"📁 Results saved to: {inference.inference_dir}")
    print(f"   - Predictions: {inference.predictions_dir}")
    print(f"   - Visualizations: {inference.viz_dir}")
    print(f"   - Metrics: {inference.metrics_dir}")
    print(f"\n📄 Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
