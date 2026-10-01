"""
extract_base.py
Extract chromosomes from source_data/2025/*.2.K.JPG.
Erodes the bounding mask to remove white halo.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [
    ROOT / "source_data" / "2023+2",
    ROOT / "source_data" / "2024",
    ROOT / "source_data" / "2025"
]
OUT_DIR = ROOT / "preprocessing" / "base_chromosomes"

def extract_and_save(img_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Background is usually white (255). We threshold to find the chromosome.
    # Chromosomes are darker.
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Find largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:
        return False
        
    # 3. Create mask for the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Do not erode deeply here, keep the original data
    # Just a tiny smoothing if needed, or none.
    eroded_mask = mask
    
    # 5. Crop the image to the bounding box of the eroded mask
    ys, xs = np.where(eroded_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return False
        
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    
    # Add a small padding
    pad = 2
    h, w = gray.shape
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)
    
    cropped_bgr = img[y1:y2, x1:x2]
    cropped_mask = eroded_mask[y1:y2, x1:x2]
    
    # 6. Convert to RGBA
    rgba = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask
    
    # Save as PNG
    cv2.imwrite(str(out_path), rgba)
    return True

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in SOURCE_DIRS:
        if d.exists():
            paths.extend(list(d.rglob("*.2.K.JPG")) + list(d.rglob("*.2.k.jpg")) + list(d.rglob("*.2.k.JPG")))
            
    print(f"Found {len(paths)} source images.")
    
    success = 0
    for p in tqdm(paths, desc="Extracting base chromosomes", mininterval=5.0, dynamic_ncols=True):
        out_name = p.stem + ".png"
        out_path = OUT_DIR / out_name
        if extract_and_save(p, out_path):
            success += 1
            
    print(f"Extracted {success}/{len(paths)} chromosomes successfully to {OUT_DIR}")

if __name__ == "__main__":
    main()
