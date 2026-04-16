import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import rasterio
from rasterio import features
from pathlib import Path
from shapely.geometry import shape, Polygon
import geopandas as gpd
from skimage import measure
from PIL import Image

# ============================================
# Load Model
# ============================================
checkpoint = torch.load("rn50_ssl4eo-s12_sar_decur_ep100.pth", map_location="cpu")
print("Checkpoint loaded")

# Create a standard ResNet-50 model
model = models.resnet50(weights=None)

# Modify first conv layer for 2-band SAR input
model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Remove the final fc layer
model.fc = nn.Identity()

# Load the state dict
state_dict = checkpoint
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]

# Remove 'module.' prefix if present
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

# Load weights
model.load_state_dict(new_state_dict, strict=False)
print("Weights loaded successfully")


# Add segmentation head
class WaterSegmentationModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.seg_head(x)
        return torch.sigmoid(x)


# Create full model
full_model = WaterSegmentationModel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_model = full_model.to(device)
full_model.eval()
print(f"Model ready on {device}")

# ============================================
# Load and Process Image
# ============================================
image_path = (
    "images/S1GRD_part_19_5_4_20250111_20250120_f914ee191bc94c5ca54d645bbbc6f9d0.tif"
)

with rasterio.open(image_path) as src:
    image = src.read()
    meta = src.meta
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width

print(f"Image shape: {image.shape}")
print(f"Image min: {image.min():.2f}, max: {image.max():.2f}")

# Preprocess for model
image_tensor = torch.from_numpy(image).float().unsqueeze(0)

# Normalize SAR data (typical dB range)
image_tensor = torch.clamp(image_tensor, -30, 0)
image_tensor = (image_tensor + 30) / 30

# Resize if needed (ResNet expects divisible by 32)
h, w = image_tensor.shape[2], image_tensor.shape[3]
new_h = ((h + 31) // 32) * 32
new_w = ((w + 31) // 32) * 32
if h != new_h or w != new_w:
    image_tensor = torch.nn.functional.interpolate(
        image_tensor, size=(new_h, new_w), mode="bilinear"
    )
    print(f"Resized from {h}x{w} to {new_h}x{new_w}")

image_tensor = image_tensor.to(device)

# ============================================
# Run Inference
# ============================================
print("Running inference...")
with torch.no_grad():
    output = full_model(image_tensor)

    # Try different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_mask = None
    best_threshold = 0.5

    for thresh in thresholds:
        mask = (output > thresh).long().squeeze().cpu().numpy()
        mask = mask[:height, :width]
        pct = (mask.sum() / mask.size) * 100
        print(f"  Threshold {thresh}: {pct:.2f}% water coverage")
        if 0 < pct < 50:  # Reasonable water coverage
            best_mask = mask
            best_threshold = thresh
            break

    if best_mask is None:
        best_mask = (output > 0.5).long().squeeze().cpu().numpy()
        best_mask = best_mask[:height, :width]
        best_threshold = 0.5

water_mask = best_mask
water_pct = (water_mask.sum() / water_mask.size) * 100

print(f"\nUsing threshold {best_threshold}: {water_pct:.2f}% water coverage")

# ============================================
# Save Outputs
# ============================================
# Create output directory
Path("predictions").mkdir(exist_ok=True)

# Save as GeoTIFF
output_path = "predictions/water_mask.tif"
out_meta = meta.copy()
out_meta.update({"count": 1, "dtype": np.uint8, "compress": "lzw"})

with rasterio.open(output_path, "w", **out_meta) as dst:
    dst.write(water_mask.astype(np.uint8), 1)

print(f"\nSaved GeoTIFF to {output_path}")

# Convert mask to shapefile
print("\nConverting to shapefile...")


def mask_to_polygons(mask, transform, crs, min_area=100):
    """Convert binary mask to polygons"""
    contours = measure.find_contours(mask, 0.5)

    polygons = []
    areas = []

    for contour in contours:
        if len(contour) >= 3:
            coords = []
            for point in contour:
                col, row = point[1], point[0]
                x = transform[0] + col * transform[1] + row * transform[2]
                y = transform[3] + col * transform[4] + row * transform[5]
                coords.append((x, y))

            try:
                polygon = Polygon(coords)
                if polygon.is_valid and polygon.area >= min_area:
                    simplified = polygon.simplify(1.0, preserve_topology=True)
                    if simplified.is_valid and not simplified.is_empty:
                        polygons.append(simplified)
                        areas.append(simplified.area)
            except Exception:
                continue

    return polygons, areas


water_mask_uint8 = water_mask.astype(np.uint8)
polygons, areas = mask_to_polygons(water_mask_uint8, transform, crs)

if polygons:
    gdf = gpd.GeoDataFrame(
        {"id": range(1, len(polygons) + 1), "area_m2": areas, "geometry": polygons},
        crs=crs,
    )

    shp_path = "predictions/water_mask.shp"
    gdf.to_file(shp_path)
    print(f"Saved shapefile to {shp_path}")
    print(f"  Number of water polygons: {len(polygons)}")
    print(f"  Total water area: {sum(areas):.2f} square meters")

    geojson_path = "predictions/water_mask.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved GeoJSON to {geojson_path}")
else:
    print("No water polygons found (area < 100 sq meters)")

# Save PNG preview
png_path = "predictions/water_mask.png"
Image.fromarray((water_mask * 255).astype(np.uint8)).save(png_path)
print(f"Saved preview to {png_path}")

print("\nDone! Check the 'predictions' folder for outputs.")
