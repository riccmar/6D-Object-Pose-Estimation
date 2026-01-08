import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_extension import RgbdFusionNetDataset
from models.models_extension import RGBD_Fusion_Net
from models.losses_extension import PoseLoss, calc_add_distance

def rgbd_fusion_net_training(device='cuda', epochs=50, batch_size=32, lr=0.0001):
    
    # Enable TF32 for A100/Ampere if available
    torch.set_float32_matmul_precision('high') 
    torch.backends.cudnn.benchmark = True

    # Dataset Root
    dataset_root = os.path.join(project_root, 'data/linemod')
    
    if not os.path.exists(dataset_root):
        print(f"Error: Linemod_preprocessed not found at {dataset_root}.")
        print("Please ensure the dataset is downloaded")
        sys.exit(1)

    # Datasets
    print("Initializing Datasets...")
    train_set = RgbdFusionNetDataset(dataset_root, split='train')
    val_set = RgbdFusionNetDataset(dataset_root, split='val')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print("Initializing Model...")
    model = RGBD_Fusion_Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    # Loss
    criterion = PoseLoss(w_x=1.0, w_r=10.0).to(device)

    scaler = GradScaler('cuda')

    recap = f"""\nReady to train with Augmentation on:
      - {len(train_set)} samples for training,
      - {len(val_set)} samples for validation,
      - {epochs} epochs,
      - {batch_size} batch size,
      - {lr} learning rate,
      - {device} device
    """
    print(recap)


    CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints', 'rgbd_fusion_net')
    MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, 'best')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) 
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_add': [],
        'lr': []
    }

    best_val_add = float('inf')
    start_epoch = 0

    print(f"\nStarting Training Loop...")

    for epoch in range(start_epoch, epochs):
        
        # Training
        model.train()
        train_loss = 0.0
        
        # Wrapped in tqdm for progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for batch in loop:
            # Move to device
            rgb = batch['rgb'].to(device, non_blocking=True)
            points = batch['points'].to(device, non_blocking=True)
            gt_t = batch['gt_t'].to(device, non_blocking=True)
            gt_R = batch['gt_R'].to(device, non_blocking=True)
            centroid = batch['centroid'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=(device=='cuda')):
                pred_q, pred_t_res = model(rgb, points)
                pred_t_abs = centroid + pred_t_res
                
                loss = criterion(pred_t_abs, pred_q, gt_t, gt_R, points)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_add_total = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                points = batch['points'].to(device, non_blocking=True)
                gt_t = batch['gt_t'].to(device, non_blocking=True)
                gt_R = batch['gt_R'].to(device, non_blocking=True)
                centroid = batch['centroid'].to(device, non_blocking=True)
                
                with autocast('cuda', enabled=(device=='cuda')):
                    pred_q, pred_t_res = model(rgb, points)
                    pred_t_abs = centroid + pred_t_res
                    
                    loss = criterion(pred_t_abs, pred_q, gt_t, gt_R, points)
                
                # ADD Metric
                add_err = calc_add_distance(pred_t_abs.float(), pred_q.float(), gt_t, gt_R, points)
                
                val_loss += loss.item()
                val_add_total += add_err
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_add_cm = (val_add_total / len(val_loader)) * 100 # Convert m to cm
        
        # Scheduler
        scheduler.step(avg_val_add_cm)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update History
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_add'].append(avg_val_add_cm)
        history['lr'].append(current_lr)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Loss: T={avg_train_loss:.4f} V={avg_val_loss:.4f} | "
              f"ADD (Val): {avg_val_add_cm:.2f}cm")
        
        # Save Best Model
        if avg_val_add_cm < best_val_add:
            best_val_add = avg_val_add_cm
            best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_rgbd_fusion_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"\t\tBest Model Saved! ADD: {avg_val_add_cm:.2f}cm")
            
        # Save Latest Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_val_loss,
            'best_add': best_val_add,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'last_checkpoint.pth'))
        
    print(f"Training Complete. Best ADD: {best_val_add:.2f}cm")
    print(f"Results saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    rgbd_fusion_net_training(
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
