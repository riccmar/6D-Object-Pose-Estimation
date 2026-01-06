import os
import sys
import torch
import numpy as np
import argparse
import gdown
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.dataset_baseline import RotationResNetDataset
from models.models_baseline import load_rotationresnet_model
from utils.evaluation_metrics import calculate_degree_error

def resnet_inference(model_path, device='cpu', sample_idx=None):
    # Setup
    dataset_root = os.path.join(project_root, 'data')

    # Handle Google Drive URL
    if model_path.startswith('http'):
        print(f"Model path detected as URL. Downloading from Google Drive...")

        checkpoints_dir = os.path.join(project_root, 'checkpoints', 'resnet')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        download_path = os.path.join(checkpoints_dir, 'resnet_downloaded.pth')
        gdown.download(model_path, download_path, quiet=False, fuzzy=True)
        model_path = download_path
        print(f"Model downloaded to: {model_path}")

    # Load Model
    model = load_rotationresnet_model(model_path, device)
    if model is None:
        sys.exit(1)

    # Prepare Dataset
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initializing Validation Dataset...")
    val_set = RotationResNetDataset(dataset_root, split='val', transform=val_transform)

    # Select Sample
    if sample_idx is None:
        idx = np.random.randint(0, len(val_set))
    else:
        idx = int(sample_idx)
        if idx >= len(val_set) or idx < 0:
            print(f"Error: Sample index {idx} out of range (0-{len(val_set)-1})")
            return

    print(f"Running inference on sample index: {idx}")
    sample_img, sample_label, obj_id = val_set[idx]

    # Prepare Input
    img_tensor = sample_img.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred_rot = model(img_tensor)

    # Convert to Numpy
    pred_q = pred_rot.cpu().numpy()[0]
    gt_q = sample_label.numpy()

    # Calculate Angular Error
    angle_error_deg = calculate_degree_error(pred_q, gt_q)

    # Normalize for display
    pred_q_norm = pred_q / np.linalg.norm(pred_q)
    gt_q_norm = gt_q / np.linalg.norm(gt_q)

    # Visualization / Printing
    print("-" * 60)
    print(f"INFERENCE RESULTS (Sample ID: {idx} | Object ID: {obj_id})")
    print("-" * 60)
    print(f"{'TYPE':<20} | {'QUATERNION [x, y, z, w]':<40}")
    print("-" * 60)
    print(f"{'PREDICTED':<20} | {np.array2string(pred_q_norm, precision=4)}")
    print(f"{'GROUND TRUTH':<20} | {np.array2string(gt_q_norm, precision=4)}")
    print("-" * 60)
    print(f"{'ANGULAR ERROR':<20} | {angle_error_deg:.4f}°")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights or Google Drive URL')
    parser.add_argument('--sample_index', type=int, help='Specific sample index to test (optional)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    resnet_inference(args.model_path, device, args.sample_index)
