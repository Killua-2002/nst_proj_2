"""
generator.py
Realistic simulation of 2 or 3 overlapping chromosomes.
Features: 
- 50% 2-overlap, 50% 3-overlap
- Speckle/Gaussian noise
- Blurred intersection regions
- Random rotations and translations (sometimes overlapping at tips)
"""
import argparse
import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "preprocessing" / "base_chromosomes"
OUT_ROOT = ROOT / "generated_data_v2.1"
OUT_IMAGE_DIR = OUT_ROOT / "images"

# Masks
OUT_MASK_DIRS = [OUT_ROOT / f"mask_{i}" for i in range(1, 4)]
OUT_OVERLAP_DIRS = {
    "12": OUT_ROOT / "overlap_12", "23": OUT_ROOT / "overlap_23",
    "13": OUT_ROOT / "overlap_13", "123": OUT_ROOT / "overlap_123"
}
OUT_PRIOR_DIRS = {
    "skeleton": OUT_ROOT / "priors_skeleton",
    "distance": OUT_ROOT / "priors_distance"
}
OUT_LABEL_CSV = OUT_ROOT / "order_labels.csv"
OUT_PREVIEW_DIR = OUT_ROOT / "previews"

CANVAS_SIZE = 512
MIN_OVERLAP_PIXELS = 80
NOISE_PROB = 0.5
NOISE_STD = 3.0

def reset_output():
    for d in [OUT_IMAGE_DIR, OUT_PREVIEW_DIR] + OUT_MASK_DIRS + list(OUT_OVERLAP_DIRS.values()) + list(OUT_PRIOR_DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)

def add_noise(rgba_img, rng, np_rng):
    """Add some speckle/gaussian noise to the RGB part"""
    if rng.random() > NOISE_PROB:
        return rgba_img
        
    arr = np.array(rgba_img)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    
    # Gaussian noise
    noise = np_rng.normal(0, NOISE_STD, rgb.shape)
    rgb = np.clip(rgb + noise, 0, 255)
    
    # Re-apply to arr
    arr[:, :, :3] = rgb.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")

def get_shape_priors(mask_binary):
    skel = skeletonize(mask_binary > 0)
    skel_img = (skel * 255).astype(np.uint8)
    
    dist = cv2.distanceTransform((mask_binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = (dist / dist.max() * 255).astype(np.uint8)
    else:
        dist = np.zeros_like(mask_binary, dtype=np.uint8)
    return skel_img, dist

def alpha_over_with_blur(base_rgb: np.ndarray, top_rgba: np.ndarray, intersection_mask: np.ndarray) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    top_rgb = top_rgba[:, :, :3].astype(np.float32)
    
    # Erode the alpha mask to hide the white halo (viền trắng) when overlapping
    alpha = top_rgba[:, :, 3]
    kernel = np.ones((5, 5), np.uint8)
    alpha = cv2.erode(alpha, kernel, iterations=1)
    
    alpha = alpha.astype(np.float32) / 255.0
    
    # Blur the top RGB slightly where they intersect to simulate depth/transparency blur
    if np.sum(intersection_mask) > 0:
        inter_soft = cv2.GaussianBlur(intersection_mask.astype(np.float32), (5, 5), 2.0)
        top_blur = cv2.GaussianBlur(top_rgb, (3, 3), 1.0)
        
        # Mix top_rgb with blurred top_rgb at intersection
        blend_factor = inter_soft[:, :, None]
        top_rgb = top_rgb * (1 - blend_factor) + top_blur * blend_factor
        
        # Also reduce alpha slightly at intersection to show bottom layer
        alpha = alpha * (1 - 0.3 * inter_soft)
        
    alpha = alpha[:, :, None]
    return np.clip(top_rgb * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)

def make_sample(items, num_chrs, rng, np_rng):
    layers = []
    masks = []
    
    # Place chromosomes
    for item in items[:num_chrs]:
        obj = item.copy()
        obj = add_noise(obj, rng, np_rng)
        
        target_size = rng.randint(220, 360)
        scale = target_size / max(obj.size)
        new_size = (max(1, int(obj.size[0] * scale)), max(1, int(obj.size[1] * scale)))
        
        obj = obj.resize(new_size, Image.BILINEAR)
        angle = rng.uniform(0, 360)
        obj = obj.rotate(angle, expand=True, resample=Image.BILINEAR, fillcolor=(255,255,255,0))
        
        # Tips overlap: Instead of center clustering, spread them out slightly
        # Normal cluster
        cx = rng.randint(CANVAS_SIZE//2 - 60, CANVAS_SIZE//2 + 60)
        cy = rng.randint(CANVAS_SIZE//2 - 60, CANVAS_SIZE//2 + 60)
        
        layer = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255,255,255,0))
        x = int(cx - obj.size[0]/2)
        y = int(cy - obj.size[1]/2)
        layer.alpha_composite(obj, dest=(x, y))
        
        arr = np.array(layer)
        layers.append(arr)
        masks.append((arr[:, :, 3] > 127).astype(np.uint8))
        
    # Pad to 3 layers for consistency in output (empty for the 3rd if num_chrs == 2)
    while len(layers) < 3:
        layers.append(np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8))
        masks.append(np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8))
        
    m1, m2, m3 = masks
    o12 = m1 & m2
    o23 = m2 & m3
    o13 = m1 & m3
    o123 = m1 & m2 & m3
    
    # Validate overlap
    if num_chrs == 2:
        if np.sum(o12) < MIN_OVERLAP_PIXELS: return None
    else:
        if np.sum(o12) < MIN_OVERLAP_PIXELS or np.sum(o23) < MIN_OVERLAP_PIXELS: return None
        
    # Order: 0=bottom, 1=middle, 2=top
    base = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    out = alpha_over_with_blur(base, layers[0], np.zeros_like(m1))
    out = alpha_over_with_blur(out, layers[1], o12)
    if num_chrs == 3:
        out = alpha_over_with_blur(out, layers[2], o13 | o23)
        
    # Pre-compute priors
    combined_skel = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    combined_dist = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for m in masks[:num_chrs]:
        skel, dist = get_shape_priors(m)
        combined_skel = np.maximum(combined_skel, skel)
        combined_dist = np.maximum(combined_dist, dist)
        
    # Make preview image (Side by side)
    overlay = out.copy()
    overlay[m1 > 0] = overlay[m1 > 0] * 0.8 + np.array([255, 0, 0]) * 0.2
    overlay[m2 > 0] = overlay[m2 > 0] * 0.8 + np.array([0, 255, 0]) * 0.2
    if num_chrs == 3:
        overlay[m3 > 0] = overlay[m3 > 0] * 0.8 + np.array([0, 0, 255]) * 0.2
        
    preview = np.concatenate((out, overlay), axis=1) # Left: Raw, Right: Overlay
        
    return {
        "image": out, "preview": preview,
        "m1": m1, "m2": m2, "m3": m3,
        "o12": o12, "o23": o23, "o13": o13, "o123": o123,
        "skel": combined_skel, "dist": combined_dist,
        "order": [0, 1, 2] if num_chrs == 3 else [0, 1, -1],
        "num_chrs": num_chrs
    }

def worker(idx, objects, rng, np_rng):
    num_chrs = 2 if rng.random() < 0.5 else 3
    for _ in range(20):
        items = random.sample(objects, num_chrs)
        res = make_sample(items, num_chrs, rng, np_rng)
        if res is not None:
            name = f"sample_{idx:05d}.png"
            Image.fromarray(res["image"]).save(OUT_IMAGE_DIR / name, compress_level=1)
            if idx < 20:  # save 20 previews
                Image.fromarray(res["preview"]).save(OUT_PREVIEW_DIR / name, compress_level=1)
                
            Image.fromarray(res["m1"]*255).save(OUT_MASK_DIRS[0] / name, compress_level=1)
            Image.fromarray(res["m2"]*255).save(OUT_MASK_DIRS[1] / name, compress_level=1)
            Image.fromarray(res["m3"]*255).save(OUT_MASK_DIRS[2] / name, compress_level=1)
            Image.fromarray(res["o12"]*255).save(OUT_OVERLAP_DIRS["12"] / name, compress_level=1)
            Image.fromarray(res["o23"]*255).save(OUT_OVERLAP_DIRS["23"] / name, compress_level=1)
            Image.fromarray(res["o13"]*255).save(OUT_OVERLAP_DIRS["13"] / name, compress_level=1)
            Image.fromarray(res["o123"]*255).save(OUT_OVERLAP_DIRS["123"] / name, compress_level=1)
            Image.fromarray(res["skel"]).save(OUT_PRIOR_DIRS["skeleton"] / name, compress_level=1)
            Image.fromarray(res["dist"]).save(OUT_PRIOR_DIRS["distance"] / name, compress_level=1)
            return name, res["order"], res["num_chrs"]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    
    reset_output()
    paths = list(BASE_DIR.glob("*.png"))
    if not paths:
        print("No base chromosomes found. Run extract_base.py first.")
        return
        
    print(f"Loading {len(paths)} base objects into RAM...")
    objects = [Image.open(p).convert("RGBA") for p in paths]
    
    labels = []
    print(f"Generating {args.count} samples...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = []
        for i in range(args.count):
            rng = random.Random(i)
            np_rng = np.random.default_rng(i)
            futures.append(pool.submit(worker, i, objects, rng, np_rng))
            
        for f in as_completed(futures):
            res = f.result()
            if res:
                labels.append(res)
                
    with open(OUT_LABEL_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "order", "num_chrs"])
        for name, order, num_chrs in labels:
            w.writerow([name, "_".join(map(str, order)), num_chrs])
            
    print("Done!")

if __name__ == "__main__":
    main()
