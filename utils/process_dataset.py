import os
import shutil
import yaml
import trimesh
import numpy as np
import sys

from dataset.dataset_baseline import YoloDataset, LINEMOD_ID_MAP

def process_linemod_for_yolo(dataset_root, yolo_dataset_root, max_samples_per_split=None):
    """
    Converts the LineMOD preprocessed dataset to YOLO format quickly by directly copying files.

    Args:
        dataset_root (str): Path to the root of the LineMOD preprocessed dataset.
        yolo_dataset_root (str): Path where the YOLO formatted dataset will be saved.
        max_samples_per_split (int, optional): Maximum number of samples to process per split (train/test).
                                               If None, processes all samples.
    
    Returns:
        str: Path to the generated YOLO dataset YAML configuration file.
    """

    # Cleanup and create YOLO dataset directory
    if os.path.exists(yolo_dataset_root):
        print(f"Removing existing YOLO dataset directory: {yolo_dataset_root}")
        shutil.rmtree(yolo_dataset_root)
    os.makedirs(yolo_dataset_root, exist_ok=True)

    # Mapping from object IDs to class IDs
    data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
    existing_obj_ids = sorted([int(x) for x in os.listdir(data_dir) if x.isdigit()])
    obj_id_to_class_id = {oid: i for i, oid in enumerate(existing_obj_ids)}
    print(f"Class Mapping found: {obj_id_to_class_id}")

    models_info_path = os.path.join(dataset_root, 'Linemod_preprocessed', 'models', 'models_info.yml')
    with open(models_info_path, 'r') as f:
        models_info = yaml.safe_load(f)

    class_names = []
    for oid in existing_obj_ids:
        # Get the raw name (e.g., "ape")
        raw_name = models_info.get(oid, {}).get('name')
        if not raw_name:
            raw_name = LINEMOD_ID_MAP.get(oid, f"Object {oid}")
        
        # FORCE THE FORMAT "01 - ape"
        formatted_name = f"{oid:02d} - {raw_name}"
        
        class_names.append(formatted_name)

    # Processing each split
    for split_type in ['train', 'test']:
        yolo_split = 'train' if split_type == 'train' else 'val'

        images_dir = os.path.join(yolo_dataset_root, 'images', yolo_split)
        labels_dir = os.path.join(yolo_dataset_root, 'labels', yolo_split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        if not os.path.exists(dataset_root):
            print(f"Error: Dataset root path does not exist: {dataset_root}.")
            return
    
        # Use the Lightweight dataset for faster processing
        dataset = YoloDataset(dataset_root, split=yolo_split)

        num_samples = len(dataset)
        if max_samples_per_split:
            num_samples = min(len(dataset), max_samples_per_split)

        print(f"Processing {num_samples} samples for {yolo_split}...")

        for i in range(num_samples):
            sample = dataset[i]

            # Unpack simple data
            src_path = sample['rgb_path']
            obj_id = sample['obj_id']
            frame_idx = sample['frame_idx']
            obj_bb = sample['obj_bb'].numpy() # [x_min, y_min, width, height]

            # Use a unique, common base name for both image and label
            base_filename = f"obj_{obj_id:02d}_{frame_idx:04d}"

            yolo_class_id = obj_id_to_class_id[obj_id]

            # YOLO Format Calculations
            img_width = 640
            img_height = 480
            x_min, y_min, bb_width, bb_height = obj_bb

            x_center = (x_min + bb_width / 2) / img_width
            y_center = (y_min + bb_height / 2) / img_height
            norm_width = bb_width / img_width
            norm_height = bb_height / img_height

            # Clamp values to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_width = max(0.0, min(1.0, norm_width))
            norm_height = max(0.0, min(1.0, norm_height))

            # Write Label File
            label_path = os.path.join(labels_dir, f"{base_filename}.txt")
            with open(label_path, 'a') as f:
                f.write(f"{yolo_class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

            # Copy Image
            shutil.copy(src_path, os.path.join(images_dir, f"{base_filename}.png"))

            if (i+1) % 1000 == 0:
                print(f"  Converted {i+1} files...")

    return create_yaml(yolo_dataset_root, class_names)

def create_yaml(root, class_names):
    """
    Creates a YOLO dataset YAML configuration file.

    Args:
        root (str): Root directory of the YOLO dataset.
        class_names (list): List of class names.
    Returns:
        str: Path to the generated YAML file.
    """

    yaml_path = os.path.join(root, 'data.yaml')

    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(root)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"Dataset generation complete. Config saved to {yaml_path}.")
    return yaml_path

def load_meshes(dataset_root):
    """
    Loads 3D meshes for all objects in the dataset.
    Returns a dict: {obj_id: {'vertices': np.array, 'diameter': float}}
    """
    models_dir = os.path.join(dataset_root, 'linemod', 'Linemod_preprocessed', 'models')
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        sys.exit(1)
        
    meshes = {}
    print("Pre-loading 3D models for ADD metric...")

    # Iterate over model files (obj_01.ply, obj_02.ply, etc.)
    for filename in sorted(os.listdir(models_dir)):
        if filename.endswith(".ply") and filename.startswith("obj_"):
            try:
                obj_id = int(filename.split('_')[1].split('.')[0])
                ply_path = os.path.join(models_dir, filename)

                # Load Mesh
                mesh = trimesh.load(ply_path)
                vertices = np.array(mesh.vertices)

                # Calculate Diameter (approximate via bbox diagonal)
                extents = vertices.max(axis=0) - vertices.min(axis=0)
                diameter = np.linalg.norm(extents)

                meshes[obj_id] = {'vertices': vertices, 'diameter': diameter}
            except Exception as e:
                print(f"Warning: Failed to load mesh {filename}: {e}")

    return meshes