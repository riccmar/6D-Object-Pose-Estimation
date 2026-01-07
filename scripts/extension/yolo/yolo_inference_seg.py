import os
import sys
import random
import math
import argparse
import cv2
import torch
import gdown
import matplotlib.pyplot as plt
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.process_dataset import process_linemod_for_yolo_seg

def yolo_segmentation_inference(model_path, device='cpu', conf=0.5, num_samples=3):
    """
    Runs inference (segmentation + bbox) on random validation images.
    
    Args:
        model_path (str): Path to the trained model weights (.pt file).
        device (str): Device to use for inference ('cpu', 'cuda', 'mps').
        conf (float): Confidence threshold for predictions.
        num_samples (int): Number of random samples to test.
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
        
    
    # Define dataset root
    dataset_root = 'data/linemod'
    yolo_dataset_root = 'data/yolo_dataset_seg'
    val_images_dir = os.path.join(yolo_dataset_root, 'images', 'val')
    
    # Verify or Create Dataset
    if not os.path.exists(val_images_dir):
        print(f"Validation images not found at {val_images_dir}")
        print("Dataset not ready. Generating YOLO Segmentation dataset now...")
        
        linemod_prepocessed_path = os.path.join(dataset_root, 'Linemod_preprocessed')
    
        if os.path.exists(linemod_prepocessed_path):
            process_linemod_for_yolo_seg(dataset_root, yolo_dataset_root)
        else:
            print("Error: Linemod dataset not found. Please run 'download_dataset.py' first.")
            exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Could not find weights at {model_path}.")
        return

    print(f"Loading segmentation model from {model_path}...")
    model = YOLO(model_path)

    # Get list of validation images
    val_images = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    if not val_images:
        print(f"No images found in {val_images_dir}")
        return

    # Pick random images
    actual_samples = min(num_samples, len(val_images))
    random_images = random.sample(val_images, actual_samples)
    print(f"Running inference on {actual_samples} random images on device {device}...")

    # Run inference
    results = model.predict(random_images, conf=conf, device=device, retina_masks=True)

    # Prepare plot
    cols = 3
    rows = math.ceil(actual_samples / cols)
    
    plt.figure(figsize=(15, 6 * rows))
    
    for i, result in enumerate(results):
        # Plot the image with bounding boxes and masks
        res_plotted = result.plot(prob=True, boxes=True, masks=True)

        # Convert BGR (OpenCV) to RGB (Matplotlib)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(res_rgb)
        plt.axis('off')
        plt.title(f"Prediction Sample {i+1}")

    plt.tight_layout()
    plt.show()

    output_filename = 'inference_results.png'
    plt.savefig(output_filename)
    print(f"Inference results saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--samples', type=int, default=3, help='Number of random samples to test')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
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

    yolo_segmentation_inference(
        model_path=args.model_path,
        device=device,
        conf=args.conf,
        num_samples=args.samples
    )
