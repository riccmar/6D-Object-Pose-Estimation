import os
import sys
import torch
import argparse
from datetime import datetime
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.prepare_dataset import process_linemod_for_yolo

def yolo_finetuning(device='cpu', epochs=10, batch_size=16):
    """
    Fine-tunes a YOLO model on the LineMod dataset.
    
    Args:
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        str: Path to the directory where results are saved.
    """
    # Enable TF32 for A100
    torch.set_float32_matmul_precision('high') 
    torch.backends.cudnn.benchmark = True

    # Prepare Dataset
    dataset_root = 'data/linemod'
    yolo_dataset_root = 'data/yolo_dataset'

    data_yaml_path = process_linemod_for_yolo(dataset_root, yolo_dataset_root)
    
    # Check if dataset config exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found.")
        print("Please run 'prepare_dataset' first.")
        exit(1)

    # Load Model
    model_name = 'yolo11n.pt'
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Run Training
    epochs = epochs
    batch_size = batch_size
    imgsz = 640
    project_name = 'YOLO-LineMod-Finetuning'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"yolo_linemod_e{epochs}_b{batch_size}_t{timestamp}"

    print(f"Starting YOLO finetuning.")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name=run_name,
        save=True,           # Save checkpoints
        device=device,
        plots=True,          # Save plots
        workers=8,
        amp=True
    )

    save_dir = str(results.save_dir)
    print(f"Training completed. Results saved to: {save_dir}")

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Setup Device
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        if gpu_count > 1:
            device = [i for i in range(gpu_count)] # Restituisce [0, 1] se hai 2 GPU
            print(f"Using multiple GPUs: {device}")
        else:
            device = 0
            print(f"Using single GPU: {device}")
    else:
        device = 'cpu'
        print(f"Using device: {device}")

    epochs = args.epochs
    batch_size = args.batch_size

    # Run Training
    yolo_finetuning(
        device=device,
        epochs=epochs,
        batch_size=batch_size
    )