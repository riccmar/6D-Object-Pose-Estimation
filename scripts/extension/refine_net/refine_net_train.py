import os
import sys
import time
import argparse
import gdown
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import RgbdFusionNetDataset
from models.models_extension import RGBD_Fusion_Net, PoseRefineNet
from models.losses_extension import PoseLoss, calc_add_distance

def refine_net_training(coarse_model_path, device='cuda', epochs=10, batch_size=32, lr=0.0001):
    
    # Enable TF32 for A100 if available
    torch.set_float32_matmul_precision('high') 
    torch.backends.cudnn.benchmark = True

    # Checkpoint Dirs
    CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints', 'refine_net')
    MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, 'best')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Handle Google Drive URL
    if coarse_model_path.startswith('http'):
        print(f"Model path detected as URL. Downloading from Google Drive...")

        download_path = os.path.join(CHECKPOINT_DIR, 'refine_net_downloaded.pth')
        gdown.download(coarse_model_path, download_path, quiet=False, fuzzy=True)
        coarse_model_path = download_path
        print(f"Model downloaded to: {coarse_model_path}")

    if not os.path.exists(coarse_model_path):
        print(f"Error: Model not found at {coarse_model_path}")
        sys.exit(1)

    # Dataset Root
    dataset_root = os.path.join(project_root, 'data/linemod')
    if not os.path.exists(dataset_root):
        print(f"Error: Linemod dataset not found at {dataset_root}.")
        sys.exit(1)

    # Datasets
    print("Initializing Datasets...")
    train_set = RgbdFusionNetDataset(dataset_root, split='train')
    val_set = RgbdFusionNetDataset(dataset_root, split='val')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Coarse Model
    print("Initializing Coarse Model...")
    coarse_model = RGBD_Fusion_Net().to(device)

    # Load weights    
    if os.path.exists(coarse_model_path):
        checkpoint = torch.load(coarse_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            coarse_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            coarse_model.load_state_dict(checkpoint)
        print(f"Coarse Model Loaded from {coarse_model_path}")
    else:
        print(f"Error: Coarse weights not found at {coarse_model_path}")
        sys.exit(1)
        
    # Freeze Coarse Model
    coarse_model.eval()
    for param in coarse_model.parameters():
        param.requires_grad = False

    # Initialize Refinement Model
    print("Initializing Refinement Model...")
    refine_model = PoseRefineNet().to(device)
    optimizer = optim.Adam(refine_model.parameters(), lr=lr)
    
    # Loss & Scaler
    criterion = PoseLoss(w_x=1.0, w_r=10.0).to(device)
    scaler = GradScaler('cuda')

    # History
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_add': [],
    }

    best_val_add = float('inf') # We want minimal distance error (cm)
    
    print(f"\nStarting Refinement Training Loop...")

    for epoch in range(epochs):
        # Training
        refine_model.train()
        train_loss_accum = 0.0
        
        loop = tqdm(train_loader, desc=f"Refine Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for batch in loop:
            rgb = batch['rgb'].to(device, non_blocking=True)
            points = batch['points'].to(device, non_blocking=True)
            gt_t = batch['gt_t'].to(device, non_blocking=True)
            gt_R = batch['gt_R'].to(device, non_blocking=True)
            centroid = batch['centroid'].to(device, non_blocking=True)
            obj_ids = batch['obj_id'].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with autocast('cuda'):
                # Coarse Pass (Frozen)
                with torch.no_grad():
                    pred_q_coarse, pred_t_res_coarse = coarse_model(rgb, points)
                    pred_t_coarse = centroid + pred_t_res_coarse
                    pred_q_coarse = F.normalize(pred_q_coarse, p=2, dim=1)
                
                # Transform to Local Frame
                points_T = points.transpose(1, 2) # (B, 3, N)
                b = pred_q_coarse.shape[0]
                x, y, z, w = pred_q_coarse[:, 0], pred_q_coarse[:, 1], pred_q_coarse[:, 2], pred_q_coarse[:, 3]
                pred_R_coarse = torch.stack([
                    1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
                    2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
                    2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
                ], dim=1).reshape(b, 3, 3) # (B, 3, 3)
                
                points_centered = points_T - pred_t_res_coarse.unsqueeze(2) 
                # Rotate
                points_local = torch.bmm(pred_R_coarse.transpose(1, 2), points_centered) # (B, 3, N)
                
                # Refine Forward
                delta_q, delta_t = refine_model(points_local)
                
                # Update Pose
                t_update = torch.bmm(pred_R_coarse, delta_t.unsqueeze(2)).squeeze(2)
                pred_t_final = pred_t_coarse + t_update
                pred_q_final = F.normalize(pred_q_coarse + delta_q, p=2, dim=1)
                
                # Loss
                loss = criterion(pred_t_final, pred_q_final, gt_t, gt_R, points)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # Validation
        refine_model.eval()
        val_loss_accum = 0.0
        val_add_total = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                points = batch['points'].to(device, non_blocking=True)
                gt_t = batch['gt_t'].to(device, non_blocking=True)
                gt_R = batch['gt_R'].to(device, non_blocking=True)
                centroid = batch['centroid'].to(device, non_blocking=True)

                with autocast('cuda'):
                    # Coarse
                    pred_q_c, pred_t_res_c = coarse_model(rgb, points)
                    pred_t_c = centroid + pred_t_res_c
                    pred_q_c = F.normalize(pred_q_c, p=2, dim=1)
                    
                    # Local Prep
                    points_T = points.transpose(1, 2)
                    b = pred_q_c.shape[0]
                    x, y, z, w = pred_q_c[:, 0], pred_q_c[:, 1], pred_q_c[:, 2], pred_q_c[:, 3]
                    pred_R_c = torch.stack([
                        1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
                        2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
                        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2
                    ], dim=1).reshape(b, 3, 3)
                    
                    pts_centered = points_T - pred_t_res_c.unsqueeze(2)
                    pts_local = torch.bmm(pred_R_c.transpose(1, 2), pts_centered)
                    
                    # Refine
                    delta_q, delta_t = refine_model(pts_local)
                    
                    t_up = torch.bmm(pred_R_c, delta_t.unsqueeze(2)).squeeze(2)
                    pred_t_f = pred_t_c + t_up
                    pred_q_f = F.normalize(pred_q_c + delta_q, p=2, dim=1)
                    
                    loss = criterion(pred_t_f, pred_q_f, gt_t, gt_R, points)
                    
                    # ADD Metric (cm)
                    add_err = calc_add_distance(pred_t_f.float(), pred_q_f.float(), gt_t, gt_R, points)
                    add_err_cm = add_err * 100
                
                val_loss_accum += loss.item()
                val_add_total += add_err_cm

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_add = val_add_total / len(val_loader)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_add'].append(avg_val_add)
        
        print(f"Epoch {epoch+1:02d} | "
              f"Loss: T={avg_train_loss:.4f} V={avg_val_loss:.4f} | "
              f"ADD: {avg_val_add:.2f}cm")
        
        # Save Best
        if avg_val_add < best_val_add:
            best_val_add = avg_val_add
            torch.save(refine_model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_refine_model.pth'))
            print(f"\t\tBest Refine Model Saved (ADD: {best_val_add:.2f}cm)")
        
        # Save Latest Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': refine_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_val_loss,
            'best_add': best_val_add,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'last_checkpoint.pth'))
    
    print(f"Training Complete. Best ADD: {best_val_add:.2f}cm")
    print(f"Results saved to: {CHECKPOINT_DIR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--coarse_model_path', type=str, required=True, help='Path to trained Coarse Model weights or Google Drive URL')
    
    args = parser.parse_args()
    
    refine_net_training(
        coarse_model_path=args.coarse_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
