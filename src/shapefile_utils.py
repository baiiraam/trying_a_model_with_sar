"""
Utilities for viewing and comparing shapefiles
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


def view_shapefile(shapefile_path, title=None):
    """Display shapefile"""
    gdf = gpd.read_file(shapefile_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor="blue", facecolor="lightblue", alpha=0.5)
    ax.set_title(title or Path(shapefile_path).stem)
    ax.set_aspect("equal")
    plt.show()

    return gdf


def compare_predictions_vs_ground_truth(pred_shp_path, gt_shp_path, save_path=None):
    """Compare predicted and ground truth shapefiles"""

    pred = gpd.read_file(pred_shp_path)
    gt = gpd.read_file(gt_shp_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    gt.plot(ax=axes[0], edgecolor="green", facecolor="lightgreen", alpha=0.5)
    axes[0].set_title(f"Ground Truth ({len(gt)} polygons)")
    axes[0].set_aspect("equal")

    # Prediction
    pred.plot(ax=axes[1], edgecolor="blue", facecolor="lightblue", alpha=0.5)
    axes[1].set_title(f"Prediction ({len(pred)} polygons)")
    axes[1].set_aspect("equal")

    # Overlay
    gt.plot(
        ax=axes[2],
        edgecolor="green",
        facecolor="none",
        linewidth=2,
        label="Ground Truth",
    )
    pred.plot(
        ax=axes[2], edgecolor="blue", facecolor="none", linewidth=2, label="Prediction"
    )
    axes[2].set_title("Overlay (Green=GT, Blue=Pred)")
    axes[2].legend()
    axes[2].set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved comparison to: {save_path}")

    plt.show()


def shapefile_to_dataframe(shapefile_path):
    """Convert shapefile to pandas DataFrame"""
    gdf = gpd.read_file(shapefile_path)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["area"] = gdf.geometry.area
    return df


# Usage example
if __name__ == "__main__":
    # View prediction shapefile
    pred_path = "inferences/first_inference/predictions/scene_1_water_mask.shp"
    if Path(pred_path).exists():
        view_shapefile(pred_path, "Predicted Water Bodies")
