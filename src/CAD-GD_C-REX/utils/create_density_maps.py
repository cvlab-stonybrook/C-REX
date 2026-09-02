import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Paths
anno_path = "/home/panos/rec_obj/CAD-GD/datasets/rec-8k/annotations.json"
image_root = "/nfs/bigtensor/add_disk0/kostas/counting/rec-8k/"  # folder where the original images are stored
output_root = "/nfs/bigtensor/add_disk0/kostas/counting/rec-8k/density_maps/"

sigma = 15  # fixed Gaussian kernel size as stated
os.makedirs(output_root, exist_ok=True)

def generate_density_map(img_shape, points, sigma=15):
    """Generate density map using normalized Gaussian kernels."""
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < h and 0 <= ix < w:
            density[iy, ix] += 1.0
    # apply Gaussian blur (normalization via sum-preserving filter)
    density = gaussian_filter(density, sigma=sigma, mode='constant')
    # optional: normalize so integral ≈ number of points
    if density.sum() > 0:
        density *= len(points) / density.sum()
    return density

# Load annotation JSON
with open(anno_path, "r") as f:
    annotations = json.load(f)

for img_name, objects in tqdm(annotations.items()):
    img_path = os.path.join(image_root, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}, skipping.")
        continue

    # load image to get shape
    img = Image.open(img_path)
    w, h = img.size

    for ref_expr, obj in objects.items():
        points = obj.get("points", [])
        if not points:
            continue
        density = generate_density_map((h, w), points, sigma=sigma)

        # save density map
        out_dir = os.path.join(output_root, os.path.splitext(img_name)[0])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ref_expr.lower().strip()}.npy")
        np.save(out_path, density)

print("Density map generation complete.")
