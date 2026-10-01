"""
train_baseline.py
Vòng lặp huấn luyện PyTorch cho mô hình Teacher-Student (Baseline).
"""
import os
import argparse
import csv
import math
import time
from pathlib import Path
import sys

# Thêm đường dẫn gốc để import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from models.architectures import build_model
from models.losses import standard_loss
from data_pipeline.dataloader import make_ds

def update_ema(student, teacher, alpha=0.999):
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
        for s_buf, t_buf in zip(student.buffers(), teacher.buffers()):
            t_buf.data.copy_(s_buf.data)

def calculate_dice(y_true, y_pred_logits):
    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float()
    inter = torch.sum(y_true * y_pred, dim=[1,2,3])
    den = torch.sum(y_true + y_pred, dim=[1,2,3])
    dice = (2.0 * inter + 1e-6) / (den + 1e-6)
    return torch.mean(dice).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/processed")
    ap.add_argument("--results-dir", default="experiments")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    ap.add_argument("--patience", type=int, default=25, help="Số epoch đợi trước khi Early Stopping")
    ap.add_argument("--model-type", choices=["segformer", "swin"], default="segformer")
    ap.add_argument("--resume", type=str, default=None, help="Đường dẫn tới file checkpoint (.pth)")
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / args.dataset_dir
    out_dir = root_dir / args.results_dir / f"ts_standard_{args.model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_loader, nt = make_ds(dataset_dir, args.batch_size, split='train')
    val_loader,   nv = make_ds(dataset_dir, args.batch_size, split='val')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building architectures...")
    student = build_model(args.model_type).to(device)
    teacher = build_model(args.model_type).to(device)
    
    # Init teacher = student
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    epoch_log_file = out_dir / "epoch_log.csv"
    step_log_file = out_dir / "step_log.csv"
    
    import glob
    if not args.resume:
        # Auto-detect latest checkpoint in out_dir
        ckpts = glob.glob(str(out_dir / "epoch_*.pth"))
        if ckpts:
            ckpts.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
            args.resume = ckpts[-1]
            print(f"Auto-detected checkpoint: {args.resume}")

    best_val_dice = 0.0
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming weights from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        # Always sync the scheduler deterministically to avoid corrupted states 
        # from previous interrupted runs.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_epoch - 1):
                scheduler.step()
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
    else:
        start_epoch = 1
        with open(epoch_log_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "sup_loss", "cons_loss", "val_loss_student", "val_dice_student", "val_loss_teacher", "val_dice_teacher"])
        with open(step_log_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "loss", "sup_loss", "cons_loss"])

    global_step = 0

    print(f"Starting Training: {args.model_type.upper()} Teacher-Student (Standard Loss)")
    print(f"Train: {nt} imgs | Val: {nv} imgs | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("-" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        teacher.eval()
        
        cons_weight = math.exp(-5.0 * (1.0 - min(1.0, epoch / 20.0))**2)
        t0 = time.time()
        
        train_loss, train_sup, train_cons = 0.0, 0.0, 0.0
        steps = 0
        
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            global_step += 1
            
            # Noise for consistency loss
            noise = torch.randn_like(x) * 0.05
            x_student = torch.clamp(x + noise, 0.0, 1.0)
            
            optimizer.zero_grad()
            
            with autocast():
                y_pred_s = student(x_student)
                with torch.no_grad():
                    y_pred_t = teacher(x)
                    
                l_sup = standard_loss(y, y_pred_s)
                
                # Consistency loss (MSE on probabilities)
                prob_s = torch.sigmoid(y_pred_s)
                prob_t = torch.sigmoid(y_pred_t)
                l_cons = nn.MSELoss()(prob_s, prob_t)
                
                loss = l_sup + cons_weight * l_cons
                
            scaler.scale(loss).backward()
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            skip_ema = scale_before > scale_after
            if not skip_ema:
                update_ema(student, teacher)
            
            ls_val, lc_val, l_val = l_sup.item(), l_cons.item(), loss.item()
            train_loss += l_val
            train_sup += ls_val
            train_cons += lc_val
            steps += 1
            
            with open(step_log_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch, global_step, l_val, ls_val, lc_val])
                
            print(f"\r  Ep {epoch:03d} step {steps:4d}/{len(train_loader):4d} | loss: {l_val:.4f}", end="", flush=True)
                
        train_loss /= steps
        train_sup /= steps
        train_cons /= steps
        
        # Validation
        student.eval()
        val_loss_s, val_dice_s, val_loss_t, val_dice_t = 0.0, 0.0, 0.0, 0.0
        val_steps = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                with autocast():
                    p_s = student(x)
                    p_t = teacher(x)
                    
                    v_ls = standard_loss(y, p_s).item()
                    v_lt = standard_loss(y, p_t).item()
                    
                v_ds = calculate_dice(y, p_s)
                v_dt = calculate_dice(y, p_t)
                
                val_loss_s += v_ls; val_dice_s += v_ds
                val_loss_t += v_lt; val_dice_t += v_dt
                val_steps += 1
                
        val_loss_s /= val_steps; val_dice_s /= val_steps
        val_loss_t /= val_steps; val_dice_t /= val_steps
        
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        print(f"\rEp {epoch:03d}/{args.epochs} | {elapsed:.0f}s | LR: {current_lr:.6f} | loss {train_loss:.4f} | S_Dice {val_dice_s:.4f} | T_Dice {val_dice_t:.4f}")
        
        with open(epoch_log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_sup, train_cons, val_loss_s, val_dice_s, val_loss_t, val_dice_t])
            
        val_dice = val_dice_t
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
                'best_val_dice': best_val_dice
            }, str(out_dir / "best_model.pth"))
            print(f"  └─ ✅ New best model saved (Dice={val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  └─ ⚠️ No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs!")
                break
            
        epoch_path = out_dir / f"epoch_{epoch:03d}.pth"
        checkpoint = {
            'epoch': epoch,
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_dice': best_val_dice
        }
        torch.save(checkpoint, str(epoch_path))
        
        old_epoch = epoch - 5
        if old_epoch > 0:
            old_path = out_dir / f"epoch_{old_epoch:03d}.pth"
            if old_path.exists(): os.remove(str(old_path))

if __name__ == "__main__":
    main()
