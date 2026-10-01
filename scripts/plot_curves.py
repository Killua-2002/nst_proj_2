"""
plot_curves.py
Vẽ biểu đồ Learning Curve (Train/Val Loss và Dice Score) từ file log CSV.
Phục vụ trực tiếp cho báo cáo khoa học.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_curves(csv_path, output_path):
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    print(f"[plot_curves] Columns found: {cols}")

    epochs = df['epoch']

    # --- Auto-detect train loss column ---
    # Baseline: 'loss' | Fuzzy: 'fuzzy_loss'
    if 'loss' in cols:
        train_loss = df['loss']
        loss_label = 'Train Loss (Standard)'
    elif 'fuzzy_loss' in cols:
        train_loss = df['fuzzy_loss']
        loss_label = 'Train Loss (Fuzzy)'
    else:
        raise KeyError(f"Không tìm thấy cột loss. Các cột hiện có: {cols}")

    # --- Auto-detect val loss columns ---
    # Baseline: 'val_loss_student', 'val_loss_teacher'
    # Fuzzy:    'val_fuzzy_student', 'val_fuzzy_teacher'
    if 'val_loss_student' in cols:
        val_loss_s = df['val_loss_student']
        val_loss_t = df['val_loss_teacher']
    elif 'val_fuzzy_student' in cols:
        val_loss_s = df['val_fuzzy_student']
        val_loss_t = df['val_fuzzy_teacher']
    else:
        raise KeyError(f"Không tìm thấy cột val_loss. Các cột hiện có: {cols}")

    # --- Auto-detect Dice columns ---
    val_dice_s = df['val_dice_student']
    val_dice_t = df['val_dice_teacher']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Biểu đồ 1: Train Loss
    axes[0].plot(epochs, train_loss, label=loss_label, color='royalblue', linewidth=1.5)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)

    # Biểu đồ 2: Val Loss (Student vs Teacher)
    axes[1].plot(epochs, val_loss_s, label='Val Loss - Student', color='orange', linewidth=1.5)
    axes[1].plot(epochs, val_loss_t, label='Val Loss - Teacher (EMA)', color='red', linewidth=1.5, linestyle='--')
    axes[1].set_title("Validation Loss: Student vs Teacher")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)

    # Biểu đồ 3: Dice Score (Student vs Teacher)
    axes[2].plot(epochs, val_dice_s, label='Dice - Student', color='mediumseagreen', linewidth=1.5)
    axes[2].plot(epochs, val_dice_t, label='Dice - Teacher (EMA)', color='darkgreen', linewidth=1.5, linestyle='--')
    axes[2].set_title("Validation Dice: Student vs Teacher")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Dice Coefficient")
    axes[2].legend(); axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Lưu biểu đồ tại: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Đường dẫn tới file epoch_log.csv")
    ap.add_argument("--out", required=True, help="Đường dẫn lưu ảnh (vd: curve.png)")
    args = ap.parse_args()

    plot_training_curves(args.csv, args.out)
