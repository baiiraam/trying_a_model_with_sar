# Training
# python sar_vit_water_segmentation.py

# Inference
from sar_vit_water_segmentation import run_inference

run_inference(
    model_path="vit_best_model.pth",
    image_path="new_image.tif",
    output_path="result.png",
)
