import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_baseline import RotationResNetDataset
from models.models_baseline import RotationResNet
from models.losses_baseline import QuaternionLoss

def resnet_training(device='cpu', epochs=50, batch_size=64, lr=0.0001):
    """
    Trains a ResNet model for rotation estimation on the LineMod dataset.
    
    Args:
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
    
    Returns:
        str: Path to the directory where results are saved.
    """
    
    # Enable TF32 for A100/Ampere if available
    torch.set_float32_matmul_precision('high') 
    torch.backends.cudnn.benchmark = True

    # Dataset Root (Assumed to be at project root 'data')
    dataset_root = os.path.join(project_root, 'data')
    
    if not os.path.exists(os.path.join(dataset_root, 'linemod/Linemod_preprocessed')):
        print(f"Error: Linemod_preprocessed not found in {dataset_root}.")
        print("Please ensure the dataset is downloaded")
        sys.exit(1)

    # Define Augmentations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # Color Jitter: Randomly change brightness, contrast, saturation, and hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),

        # Gaussian Blur: Simulate out-of-focus camera or motion blur
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),

        # Grayscale: Force model to look at shape, not just color
        transforms.RandomGrayscale(p=0.1),

        # Standard stuff
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initializing Datasets...")
    train_set = RotationResNetDataset(dataset_root, split='train', transform=train_transform)
    val_set = RotationResNetDataset(dataset_root, split='val', transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model
    print("Initializing Model...")
    model = RotationResNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Loss
    USE_QUAT_LOSS = False  # Set False to use MSE
    if USE_QUAT_LOSS:
        loss_fn = QuaternionLoss()
        loss_name = "QUAT_LRSCHED"
    else:
        loss_fn = nn.MSELoss()
        loss_name = "MSE_LRSCHED"

    recap = f"""\nReady to train with Augmentation on:
      - {len(train_set)} samples for training,
      - {len(val_set)} samples for validation,
      - {epochs} epochs,
      - {batch_size} batch size,
      - {lr} learning rate,
      - {device} device
    """
    print(recap)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     
    CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints', 'resnet')
    MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, 'best')
    
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'latest_e{epochs}_b{batch_size}_lr{lr}_t{timestamp}_{loss_name}.pth')
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'best_e{epochs}_b{batch_size}_lr{lr}_t{timestamp}_{loss_name}.pth')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # History tracking
    history = {
        'train_loss': [], 'val_loss': []
    }

    best_val_loss = float('inf')
    start_epoch = 0

    # Initialize scaler for AMP
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print("\nStarting Training Loop...")

    for epoch in range(start_epoch, epochs):
        print('\nEpoch %d\n--------------------------' % (epoch+1))

        # Training
        model.train()
        train_loss_accum = 0.0
        size = len(train_loader.dataset)

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_rot = model(images)
                loss = loss_fn(pred_rot, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()

            if batch_idx % 100 == 0:
              current = batch_idx * len(images)
              print(f'Train loss: {loss.item():.7f} [{current}/{size}]')

        # Validation
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_rot = model(images)
                    loss = loss_fn(pred_rot, labels)

                val_loss_accum += loss.item()

        # Calculate Averages
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)

        scheduler.step(avg_val_loss)

        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"\nTrain Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history
        }, CHECKPOINT_PATH)

        # Save Best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"\t\tNew Best Model Saved! (Loss: {best_val_loss:.4f})")
    
    print("\nTraining Complete!")
    print(f"Results saved to: {CHECKPOINT_DIR}")
    return CHECKPOINT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    args = parser.parse_args()
    
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    epochs = args.epochs
    batch_size = args.batch_size

    # Run Training
    resnet_training(
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.lr
    )