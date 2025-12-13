import os
import torch
import argparse
from ultralytics import YOLO

from utils.prepare_dataset import process_linemod_to_yolo_fast

def yolo_evaluation(model_path, device='cpu', batch_size=16):
    """
    Evaluates a YOLO model on the LineMod dataset.
    
    Args:
        model_path (str): Path to the trained model weights (.pt file).
        device (str): Device to use for evaluation ('cpu', 'cuda', 'mps').
        batch_size (int): Batch size for evaluation.
    """
    
    # Prepare Dataset (ensure data.yaml exists)
    linemod_prepocessed_dataset_root = 'data/linemod/Linemod_preprocessed'
    
    if not os.path.exists(linemod_prepocessed_dataset_root):
        print(f"Linemod dataset not found at {linemod_prepocessed_dataset_root}. Please download it first.")
        exit(1)
    
    dataset_root = 'data/linemod'
    yolo_dataset_root = 'data/yolo_dataset'
    data_yaml_path = 'data/yolo_dataset/data.yaml'

    if not os.path.exists(yolo_dataset_root):
        print(f"Yolo dataset not found at {yolo_dataset_root}. Preparing dataset...")
        data_yaml_path = process_linemod_to_yolo_fast(dataset_root, yolo_dataset_root)

    if not os.path.exists(data_yaml_path):
        print(f"Error: data yaml not found at {data_yaml_path}.")
        exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Could not find weights at {model_path}.")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Run validation
    print(f"Starting evaluation on device: {device}")
    metrics = model.val(
        data=data_yaml_path,
        split='val',     # Use the validation split (which contains your test data)
        imgsz=640,
        batch=batch_size,
        conf=0.001,      # Confidence threshold
        iou=0.6,         # IoU threshold for NMS
        plots=True,      # Save plots like confusion matrices
        device=device
    )

    # Print key metrics
    print(f"\n--- Evaluation Results ---")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights (.pt file)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    
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
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using MPS device: {device}")
    else:
        device = 'cpu'
        print(f"Using device: {device}")

    yolo_evaluation(
        model_path=args.model_path,
        device=device,
        batch_size=args.batch_size
    )
