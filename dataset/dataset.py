import os
import yaml
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class LightweightDataset(Dataset):
    """
    Lightweight version of Custom Dataset that only returns RGB image paths and object bounding boxes.
    Used in YOLO finetuning to speed up data loading.
    """

    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        self.dataset_root = dataset_root
        self.split = split

        self.train_ratio = train_ratio
        self.seed = seed

        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
        self.model_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'models')

        # Load models_info.yml
        models_info_path = os.path.join(self.model_dir, 'models_info.yml')
        with open(models_info_path, 'r') as f:
            self.models_info = yaml.safe_load(f)

        # Collect samples similar to before
        all_samples = []
        self.all_gt_data = {}

        for obj_id_str in sorted(os.listdir(self.data_dir)):
            if not obj_id_str.isdigit():
              continue  # Skip non-numeric folders

            obj_id = int(obj_id_str)
            gt_path = os.path.join(self.data_dir, obj_id_str, 'gt.yml')

            if not os.path.exists(gt_path):
              print(f"Warning: gt.yml or info.yml not found for object {obj_id_str}. Skipping.")
              continue

            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f)
            self.all_gt_data[obj_id] = gt_data

            # Add all valid frames to a list
            for frame_idx in gt_data.keys():
                # We skip the file existence check here for speed
                all_samples.append((obj_id, frame_idx))

        # Split logic
        np.random.seed(self.seed)
        np.random.shuffle(all_samples)
        train_size = int(len(all_samples) * train_ratio)

        if self.split == 'train':
            self.samples = all_samples[:train_size]
        else:
            self.samples = all_samples[train_size:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, frame_idx = self.samples[idx]
        obj_id_str = f"{obj_id:02d}"
        frame_idx_str = f"{frame_idx:04d}"

        # Just return paths, don't load images (faster)
        rgb_path = os.path.join(self.data_dir, obj_id_str, 'rgb', f"{frame_idx_str}.png")

        # Get ground truth pose and bounding box for the specific object in this frame
        gt_frame_data = self.all_gt_data[obj_id][frame_idx]
        obj_gt = gt_frame_data[0]
        obj_bb = torch.tensor(obj_gt['obj_bb'], dtype=torch.int32)

        # Object metadata (name, diameter, etc.) from models_info.yml
        obj_metadata = self.models_info.get(obj_id, {})
        obj_name = obj_metadata.get('name', f"Object {obj_id}")

        return {
            "rgb_path": rgb_path,      # Return image path, not image tensor
            "obj_id": obj_id,          # Object id (e.g., 1 for Ape)
            "frame_idx": frame_idx,    # Frame id for the class
            "obj_bb": obj_bb,          # Bounding box [x_min, y_min, width, height]
            "obj_name": obj_name,      # Object name (e.g., 'Ape')
        }


class CustomDataset(Dataset):
    """ 
    Custom Dataset for data from Linemod preprocessed dataset.
    """

    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        self.dataset_root = dataset_root # e.g., /content/datasets/linemod
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
        self.model_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'models')

        # Load models_info.yml for object metadata (names, diameters, etc.)
        models_info_path = os.path.join(self.model_dir, 'models_info.yml')
        with open(models_info_path, 'r') as f:
            self.models_info = yaml.safe_load(f)

        # Collect all samples (obj_id, frame_idx)
        all_samples = []
        self.all_gt_data = {}
        self.all_info_data = {}

        # Iterate through object folders (e.g., '01', '02', ...) in the 'data' directory
        for obj_id_str in sorted(os.listdir(self.data_dir)):
            if not obj_id_str.isdigit():
                continue # Skip non-numeric folders

            obj_id = int(obj_id_str)
            gt_path = os.path.join(self.data_dir, obj_id_str, 'gt.yml')
            info_path = os.path.join(self.data_dir, obj_id_str, 'info.yml')

            if not os.path.exists(gt_path) or not os.path.exists(info_path):
                print(f"Warning: gt.yml or info.yml not found for object {obj_id_str}. Skipping.")
                continue

            with open(gt_path, 'r') as f:
                gt_data_for_obj = yaml.safe_load(f)
            with open(info_path, 'r') as f:
                info_data_for_obj = yaml.safe_load(f)

            self.all_gt_data[obj_id] = gt_data_for_obj
            self.all_info_data[obj_id] = info_data_for_obj

            # Iterate through frames for which ground truth data is available
            # and check if all necessary files exist
            for frame_idx in gt_data_for_obj.keys():
                obj_id_padded_str = f"{obj_id:02d}"
                frame_idx_padded_str = f"{frame_idx:04d}"

                rgb_file_path = os.path.join(self.data_dir, obj_id_padded_str, 'rgb', f"{frame_idx_padded_str}.png")
                depth_file_path = os.path.join(self.data_dir, obj_id_padded_str, 'depth', f"{frame_idx_padded_str}.png")
                mask_file_path = os.path.join(self.data_dir, obj_id_padded_str, 'mask', f"{frame_idx_padded_str}.png")

                # Only add sample if all relevant files exist
                if os.path.exists(rgb_file_path) and \
                   os.path.exists(depth_file_path) and \
                   os.path.exists(mask_file_path):
                    all_samples.append((obj_id, frame_idx))
                else:
                    print(f"Warning: Skipping sample (obj_id={obj_id}, frame_idx={frame_idx}) due to missing file(s).")
                    if not os.path.exists(rgb_file_path): print(f"  Missing RGB: {rgb_file_path}")
                    if not os.path.exists(depth_file_path): print(f"  Missing Depth: {depth_file_path}")
                    if not os.path.exists(mask_file_path): print(f"  Missing Mask: {mask_file_path}")
                    #pass

        # Split data into train and test
        np.random.seed(self.seed)
        np.random.shuffle(all_samples)
        train_size = int(len(all_samples) * self.train_ratio)

        if self.split == 'train':
            self.samples = all_samples[:train_size]
        elif self.split == 'test':
            self.samples = all_samples[train_size:]
        else:
            raise ValueError("Split must be 'train' or 'test'")

        # Define image transformations
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

    def __len__(self):
        # Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        # Load a dataset sample.
        obj_id, frame_idx = self.samples[idx]
        obj_id_str = f"{obj_id:02d}"
        frame_idx_str = f"{frame_idx:04d}"

        # Construct paths for RGB, depth, and mask images
        rgb_path = os.path.join(self.data_dir, obj_id_str, 'rgb', f"{frame_idx_str}.png")
        depth_path = os.path.join(self.data_dir, obj_id_str, 'depth', f"{frame_idx_str}.png")
        mask_path = os.path.join(self.data_dir, obj_id_str, 'mask', f"{frame_idx_str}.png")

        # Load and transform RGB image
        rgb_img = self.transform_rgb(Image.open(rgb_path).convert("RGB"))

        # Load depth image (uint16) and convert to float32 tensor
        depth_img_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # Ensure depth is treated as a single-channel float tensor (e.g., in mm)
        depth_img = torch.from_numpy(depth_img_raw.astype(np.float32)).unsqueeze(0) # Add channel dimension

        # Load mask image (uint8) and convert to single-channel binary float32 tensor (0 or 1)
        mask_img_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # Convert to binary mask (object pixels = 1.0, background = 0.0) and add channel dimension
        mask_img = torch.from_numpy((mask_img_raw[:,:,0] > 0).astype(np.float32)).unsqueeze(0)

        # Get ground truth pose and bounding box for the specific object in this frame
        gt_frame_data = self.all_gt_data[obj_id][frame_idx]
        # Find the entry corresponding to the current obj_id
        obj_gt = gt_frame_data[0]

        cam_R_m2c = torch.tensor(obj_gt['cam_R_m2c'], dtype=torch.float32).reshape(3, 3)
        cam_t_m2c = torch.tensor(obj_gt['cam_t_m2c'], dtype=torch.float32) # Translation vector
        obj_bb = torch.tensor(obj_gt['obj_bb'], dtype=torch.int32) # [x_min, y_min, width, height]

        # Get camera intrinsics and depth scale from info.yml
        info_frame_data = self.all_info_data[obj_id][frame_idx]
        cam_K = torch.tensor(info_frame_data['cam_K'], dtype=torch.float32).reshape(3, 3)
        depth_scale = info_frame_data.get('depth_scale', 1.0) # Default to 1.0 if not specified

        # Object metadata (name, diameter, etc.) from models_info.yml
        obj_metadata = self.models_info.get(obj_id, {})
        obj_name = obj_metadata.get('name', f"Object {obj_id}")
        obj_diameter = obj_metadata.get('diameter', 0.0)

        # Path to the 3D model for this object
        obj_model_path = os.path.join(self.model_dir, f'obj_{obj_id:02d}.ply')

        return {
            "rgb": rgb_img,
            "depth": depth_img,
            "mask": mask_img,
            "obj_id": obj_id,
            "frame_idx": frame_idx,
            "cam_K": cam_K,
            "cam_R_m2c": cam_R_m2c,
            "cam_t_m2c": cam_t_m2c,
            "obj_bb": obj_bb,
            "depth_scale": depth_scale,
            "obj_name": obj_name,
            "obj_diameter": obj_diameter,
            "obj_model_path": obj_model_path,
        }


class CustomDataset2(Dataset):
    """
    Alternative Custom Dataset for data from Linemod preprocessed dataset.
    It takes as training samples those specified in 'train.txt' files and as test samples those in 'test.txt' files.
    """

    def __init__(self, dataset_root, split='train'):
        self.dataset_root = dataset_root  # e.g.: /content/datasets/linemod
        self.split = split  # 'train' or 'test'

        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
        self.model_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'models')

        # Load models_info.yml for object metadata (names, diameters, etc.)
        models_info_path = os.path.join(self.model_dir, 'models_info.yml')
        with open(models_info_path, 'r') as f:
            self.models_info = yaml.safe_load(f)

        self.all_gt_data = {}
        self.all_info_data = {}
        self.samples = [] # List to store (obj_id, frame_idx) pairs for the current split

        # Iterate through object folders (e.g., '01', '02', ...) in the 'data' directory
        for obj_id_str in sorted(os.listdir(self.data_dir)):
            if not obj_id_str.isdigit(): # Skip non-numeric folders
                continue

            obj_id = int(obj_id_str)
            obj_data_path = os.path.join(self.data_dir, obj_id_str)

            gt_path = os.path.join(obj_data_path, 'gt.yml')
            info_path = os.path.join(obj_data_path, 'info.yml')

            if not os.path.exists(gt_path) or not os.path.exists(info_path):
                print(f"Warning: gt.yml or info.yml not found for object {obj_id_str}. Skipping.")
                continue

            with open(gt_path, 'r') as f:
                self.all_gt_data[obj_id] = yaml.safe_load(f)
            with open(info_path, 'r') as f:
                self.all_info_data[obj_id] = yaml.safe_load(f)

            # Determine the split file path for this specific object
            split_file_name = f"{self.split}.txt"
            current_obj_split_file_path = os.path.join(obj_data_path, split_file_name)

            if not os.path.exists(current_obj_split_file_path):
                print(f"Warning: {split_file_name} not found for object {obj_id_str}. Skipping this object for current split.")
                continue

            # Load frame_idx from the split file for this object
            with open(current_obj_split_file_path, 'r') as f_split:
                for line in f_split:
                    frame_idx_str = line.strip()
                    if frame_idx_str:
                        try:
                            frame_idx = int(frame_idx_str)
                            obj_id_padded_str = f"{obj_id:02d}"
                            frame_idx_padded_str = f"{frame_idx:04d}"
                            rgb_file_path = os.path.join(self.data_dir, obj_id_padded_str, 'rgb', f"{frame_idx_padded_str}.png")
                            depth_file_path = os.path.join(self.data_dir, obj_id_padded_str, 'depth', f"{frame_idx_padded_str}.png")
                            mask_file_path = os.path.join(self.data_dir, obj_id_padded_str, 'mask', f"{frame_idx_padded_str}.png")

                            if os.path.exists(rgb_file_path) and \
                               os.path.exists(depth_file_path) and \
                               os.path.exists(mask_file_path):
                                self.samples.append((obj_id, frame_idx))
                            else:
                                print(f"Warning: Skipping sample (obj_id={obj_id}, frame_idx={frame_idx}) due to missing file(s).")
                                if not os.path.exists(rgb_file_path): print(f"  Missing RGB: {rgb_file_path}")
                                if not os.path.exists(depth_file_path): print(f"  Missing Depth: {depth_file_path}")
                                if not os.path.exists(mask_file_path): print(f"  Missing Mask: {mask_file_path}")
                                #pass

                        except ValueError:
                            print(f"Warning: Could not parse frame_idx '{frame_idx_str}' from {current_obj_split_file_path}")

        # Define image transformations
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])

    def __len__(self):
        # Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        # Load a dataset sample.
        obj_id, frame_idx = self.samples[idx]
        obj_id_str = f"{obj_id:02d}"
        frame_idx_str = f"{frame_idx:04d}"

        # Construct paths for RGB, depth, and mask images
        rgb_path = os.path.join(self.data_dir, obj_id_str, 'rgb', f"{frame_idx_str}.png")
        depth_path = os.path.join(self.data_dir, obj_id_str, 'depth', f"{frame_idx_str}.png")
        mask_path = os.path.join(self.data_dir, obj_id_str, 'mask', f"{frame_idx_str}.png")

        # Load and transform RGB image
        rgb_img = self.transform_rgb(Image.open(rgb_path).convert("RGB"))

        # Load depth image (uint16) and convert to float32 tensor
        depth_img_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # Ensure depth is treated as a single-channel float tensor (e.g., in mm)
        depth_img = torch.from_numpy(depth_img_raw.astype(np.float32)).unsqueeze(0) # Add channel dimension

        # Load mask image (uint8) and convert to single-channel binary float32 tensor (0 or 1)
        mask_img_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # Convert to binary mask (object pixels = 1.0, background = 0.0) and add channel dimension
        mask_img = torch.from_numpy((mask_img_raw[:,:,0] > 0).astype(np.float32)).unsqueeze(0)

        # Get ground truth pose and bounding box for the specific object in this frame
        gt_frame_data = self.all_gt_data[obj_id][frame_idx]
        # Find the entry corresponding to the current obj_id
        obj_gt = gt_frame_data[0]

        cam_R_m2c = torch.tensor(obj_gt['cam_R_m2c'], dtype=torch.float32).reshape(3, 3)
        cam_t_m2c = torch.tensor(obj_gt['cam_t_m2c'], dtype=torch.float32) # Translation vector
        obj_bb = torch.tensor(obj_gt['obj_bb'], dtype=torch.int32) # [x_min, y_min, width, height]

        # Get camera intrinsics and depth scale from info.yml
        info_frame_data = self.all_info_data[obj_id][frame_idx]
        cam_K = torch.tensor(info_frame_data['cam_K'], dtype=torch.float32).reshape(3, 3)
        depth_scale = info_frame_data.get('depth_scale', 1.0) # Default to 1.0 if not specified

        # Object metadata (name, diameter, etc.) from models_info.yml
        obj_metadata = self.models_info.get(obj_id, {})
        obj_name = obj_metadata.get('name', f"Object {obj_id}")
        obj_diameter = obj_metadata.get('diameter', 0.0)

        # Path to the 3D model for this object
        obj_model_path = os.path.join(self.model_dir, f'obj_{obj_id:02d}.ply')

        return {
            "rgb": rgb_img,
            "depth": depth_img,
            "mask": mask_img,
            "obj_id": obj_id,
            "frame_idx": frame_idx,
            "cam_K": cam_K,
            "cam_R_m2c": cam_R_m2c,
            "cam_t_m2c": cam_t_m2c,
            "obj_bb": obj_bb,
            "depth_scale": depth_scale,
            "obj_name": obj_name,
            "obj_diameter": obj_diameter,
            "obj_model_path": obj_model_path,
        }