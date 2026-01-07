import os
import sys
import torch
import argparse
import gdown
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.process_dataset import process_linemod_for_yolo_seg

def yolo_segmentation_evaluation(model_path, device='cpu', batch_size=16, conf=0.001):
    """
    Evaluates a YOLO Segmentation model on the LineMod dataset.
    
    Args:
        model_path (str): Path to the trained model weights (.pt file) or Google Drive URL.
        device (str): Device to use for evaluation ('cpu', 'cuda', 'mps').
        batch_size (int): Batch size for evaluation.
        conf (float): Confidence threshold.
    """
    
    # Handle Google Drive URL
    if model_path.startswith('http'):
        print(f"Model path detected as URL. Downloading from Google Drive...")
        
        checkpoints_dir = os.path.join(project_root, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        download_path = os.path.join(checkpoints_dir, 'yolo_extension_seg.pt')
        gdown.download(model_path, download_path, quiet=False, fuzzy=True)
        model_path = download_path
        print(f"Model downloaded to: {model_path}")

    # Prepare Dataset (ensure data.yaml exists)
    dataset_root = 'data/linemod'
    yolo_dataset_root = 'data/yolo_dataset_seg'
    data_yaml_path = os.path.join(yolo_dataset_root, 'data.yaml')
    
    linemod_prepocessed_dataset_root = os.path.join(dataset_root, 'Linemod_preprocessed')
    
    if not os.path.exists(linemod_prepocessed_dataset_root):
        print(f"Linemod preprocessed dataset not found at {linemod_prepocessed_dataset_root}.")
        print("Please download/preprocess it first.")
        exit(1)

    # Check/Generate YOLO Seg Dataset
    if not os.path.exists(yolo_dataset_root) or not os.path.exists(data_yaml_path):
        print(f"Yolo Segmentation dataset not found/complete at {yolo_dataset_root}. Preparing dataset...")
        data_yaml_path = process_linemod_for_yolo_seg(dataset_root, yolo_dataset_root)

    if not os.path.exists(model_path):
        print(f"Error: Could not find weights at {model_path}.")
        return

    print(f"Loading segmentation model from {model_path}...")
    model = YOLO(model_path)

    # Run validation
    print(f"Starting evaluation on device: {device}")
    metrics = model.val(
        data=data_yaml_path,
        split='val',     # Use the validation split
        imgsz=640,
        batch=batch_size,
        conf=conf,       # Use argument
        iou=0.6,         # IoU threshold for NMS
        plots=True,      # Save confusion matrices, PR curves, etc.
        device=device
    )

    # Print key metrics - Box
    print(f"\n--- BOX Metrics ---")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    
    # Print key metrics - Mask
    print(f"\n--- MASK Metrics ---")
    print(f"mAP@50:    {metrics.seg.map50:.4f}")
    print(f"mAP@50-95: {metrics.seg.map:.4f}")
    print(f"Precision: {metrics.seg.mp:.4f}")
    print(f"Recall:    {metrics.seg.mr:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    
    args = parser.parse_args()

    # Setup Device
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device = [i for i in range(gpu_count)]
            print(f"Using multiple GPUs: {device}")
        else:
            device = 0
            print(f"Using single GPU: {device}")
    else:
        device = 'cpu'
        print(f"Using device: {device}")
    
    yolo_segmentation_evaluation(
        model_path=args.model_path,
        device=device,
        batch_size=args.batch_size,
        conf=args.conf
    )
