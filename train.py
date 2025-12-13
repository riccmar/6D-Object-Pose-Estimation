import os
import torch
import wandb
import argparse
import shutil
from datetime import datetime
from ultralytics import YOLO

from utils.prepare_dataset import prepare_linemod_dataset

def yolo_finetuning(use_wandb=False, device='cpu', epochs=10, batch_size=16, export_path=None):
    """
    Fine-tunes a YOLO model on the LineMod dataset.
    
    Args:
        use_wandb (bool): Whether to enable WandB logging.
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        export_path (str): Optional path to export the best model (e.g., shared drive).
    
    Returns:
        str: Path to the directory where results are saved.
    """

    # Prepare Dataset
    dataset_root = 'data/linemod'
    yolo_dataset_root = 'data/yolo_dataset'

    data_yaml_path = prepare_linemod_dataset(dataset_root, yolo_dataset_root)
    
    # Check if dataset config exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found.")
        print("Please run 'prepare_dataset' first.")
        exit(1)

    # Configure WandB
    if use_wandb:
        # Enable WandB in Ultralytics settings
        from ultralytics.utils import SETTINGS
        SETTINGS.update({'wandb': True})
        
        # Login if key is provided
        wandb_key = os.getenv('WANDB_API_KEY', None)
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            print("WandB API key not found in environment variables. Proceeding without login.")
    else:
        # Disable WandB explicitly to avoid prompts
        os.environ['WANDB_MODE'] = 'disabled'
        
        from ultralytics.utils import SETTINGS
        SETTINGS.update({'wandb': False})

    # Load Model
    model_name = 'yolo11n.pt'
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Run Training
    epochs = epochs
    batch_size = batch_size
    imgsz = 640
    project_name = 'YOLO_LineMod_Finetune'
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
    )

    save_dir = str(results.save_dir)
    print(f"Training completed. Results saved to: {save_dir}")
    
    # Export Model to Shared Drive
    if export_path:
        if os.path.exists(export_path):
            best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
            
            if os.path.exists(best_model_path):
                new_filename = f"{run_name}.pt"
                dest_path = os.path.join(export_path, new_filename)
                
                try:
                    shutil.copy(best_model_path, dest_path)
                    print(f"Model successfully exported to: {dest_path}")
                except Exception as e:
                    print(f"Error exporting model: {e}")
            else:
                print(f"Warning: Could not find best model at {best_model_path}")
        else:
            print(f"Warning: Export path {export_path} does not exist.")

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', default='false', help='Enable WandB logging')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--export_path', type=str, default=None, help='Path to export the trained model (e.g. shared drive)')
    
    args = parser.parse_args()

    # WandB Configuration
    WANDB_FLAG = args.wandb.lower() in {"1", "true", "yes"}  # Set to True to enable WandB logging
    
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    epochs = args.epochs
    batch_size = args.batch_size
    export_path = args.export_path

    # Run Training
    yolo_finetuning(
        use_wandb=WANDB_FLAG,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        export_path=export_path
    )