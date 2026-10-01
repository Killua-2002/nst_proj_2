"""
evaluate.py
Evaluate the trained PyTorch Teacher-Student model on the held-out test set
(split='test', last 10% of the dataset = 700 samples) and save visualizations.

Usage:
    python testing/evaluate.py \
        --weights /path/to/best_model.pth \
        --out-dir  /path/to/predictions \
        [--dataset-dir data/processed] \
        [--model-type segformer] \
        [--batch-size 8] \
        [--split test]   # 'test' (default) | 'val'
"""
import os
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from data_pipeline.dataloader import make_ds
from models.architectures import build_model

def blend_mask(image_gray, mask, color, alpha=0.3):
    """Blend a binary mask onto a grayscale image with a specific color and alpha."""
    if len(image_gray.shape) == 2:
        img_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image_gray.copy()

    idx = (mask > 0.5)
    if not np.any(idx):
        return img_bgr

    color_mask = np.zeros_like(img_bgr)
    color_mask[idx] = color
    img_bgr[idx] = cv2.addWeighted(img_bgr[idx], 1 - alpha, color_mask[idx], alpha, 0)
    return img_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="data/processed")
    parser.add_argument("--weights",     type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--model-type",  type=str, default="segformer",
                        choices=["segformer", "swin"])
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--out-dir",     type=str, default="results/predictions")
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which data split to evaluate on (default: 'test').")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset split='{args.split}' ...")
    loader, n_samples = make_ds(
        Path(args.dataset_dir), args.batch_size, split=args.split
    )
    print(f"Evaluating on {n_samples} samples ({args.split} split).")

    print(f"Loading model ({args.model_type}) from {args.weights}")
    model = build_model(args.model_type).to(device)
    checkpoint = torch.load(args.weights, map_location=device)

    # Load Teacher weights by default for evaluation if available
    if 'teacher' in checkpoint:
        model.load_state_dict(checkpoint['teacher'])
        print("Loaded Teacher weights.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights (no teacher key found).")

    model.eval()

    tp, fp, fn = 0, 0, 0

    colors = [
        (255,   0,   0),   # Blue   (Mask 1)
        (  0, 255,   0),   # Green  (Mask 2)
        (  0,   0, 255),   # Red    (Mask 3)
        (  0, 255, 255),   # Yellow (Overlap 12)
        (255,   0, 255),   # Magenta(Overlap 23)
        (255, 255,   0),   # Cyan   (Overlap 13)
        (255, 255, 255),   # White  (Overlap 123)
    ]

    print("Evaluating and saving predictions...")
    first_20_visuals = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y   = x.to(device), y.to(device)
            preds   = model(x)
            pred_probs = torch.sigmoid(preds)
            pred_bin   = (pred_probs > 0.5).float()

            # Pixel-level F1 / Dice stats
            tp += torch.sum(pred_bin * y).item()
            fp += torch.sum(pred_bin * (1 - y)).item()
            fn += torch.sum((1 - pred_bin) * y).item()

            imgs_np   = x.cpu().numpy()
            pred_np   = pred_bin.cpu().numpy()
            target_np = y.cpu().numpy()

            for i in range(imgs_np.shape[0]):
                idx = batch_idx * args.batch_size + i

                img_gray = (imgs_np[i, 0, :, :] * 255).astype(np.uint8)
                img_bgr  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

                pred_overlay = img_gray.copy()
                gt_overlay   = img_gray.copy()

                for c in range(7):
                    pred_overlay = blend_mask(pred_overlay, pred_np[i, c, :, :],   colors[c], alpha=0.3)
                    gt_overlay   = blend_mask(gt_overlay,   target_np[i, c, :, :], colors[c], alpha=0.3)

                # Side-by-side: Original | Ground Truth | Prediction
                combined = np.hstack((img_bgr, gt_overlay, pred_overlay))

                if idx < 50:
                    cv2.imwrite(str(out_dir / f"test_result_{idx:04d}.png"), combined)

                if len(first_20_visuals) < 20:
                    small = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))
                    first_20_visuals.append(small)

            print(f"\rProcessed {(batch_idx + 1) * args.batch_size}/{n_samples} images", end="")

    print()
    if len(first_20_visuals) == 20:
        rows = [np.hstack(first_20_visuals[r * 4:(r + 1) * 4]) for r in range(5)]
        grid = np.vstack(rows)
        cv2.imwrite(str(out_dir / "summary_20_cases.jpg"), grid)

    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-6)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Split          : {args.split}")
    print(f"Samples        : {n_samples}")
    print(f"Overall Dice (F1): {f1_score:.4f}")
    print(f"Saved to       : {out_dir}/")


if __name__ == "__main__":
    main()
