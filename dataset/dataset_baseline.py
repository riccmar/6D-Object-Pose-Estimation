import os
import yaml
import torch
import hashlib
import numpy as np
import cv2
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

LINEMOD_ID_MAP = {
    1: 'ape', 2: 'benchvise', 3: 'bowl', 4: 'camera', 5: 'can',
    6: 'cat', 7: 'cup', 8: 'driller', 9: 'duck', 10: 'eggbox', 
    11: 'glue', 12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
}

class YoloDataset(Dataset):
    """
    Lightweight version of Custom Dataset that only returns RGB image paths and object bounding boxes.
    Used in YOLO finetuning to speed up data loading.
    """

    def __init__(self, dataset_root, split='train'):
        self.dataset_root = dataset_root
        self.split = split

        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')
        self.model_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'models')

        # Load models_info.yml
        models_info_path = os.path.join(self.model_dir, 'models_info.yml')
        with open(models_info_path, 'r') as f:
            self.models_info = yaml.safe_load(f)

        # Collect samples similar to before
        self.samples = []
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
            for frame_idx_str in gt_data.keys():
                frame_idx = int(frame_idx_str)
                
                unique_id = f"{obj_id}_{frame_idx}"
                
                # Deterministic Hash
                hash_val = int(hashlib.md5(unique_id.encode()).hexdigest(), 16) % 100
                is_train = hash_val < 80 # 80% Train, 20% Val

                # Filter based on split
                if (split == 'train' and is_train) or (split in ['val', 'test'] and not is_train):
                    self.samples.append((obj_id, frame_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, frame_idx = self.samples[idx]
        obj_id_str = f"{obj_id:02d}"
        frame_idx_str = f"{frame_idx:04d}"

        # Just return paths, don't load images (faster)
        rgb_path = os.path.join(self.data_dir, obj_id_str, 'rgb', f"{frame_idx_str}.png")

        # Get the list of all objects in this frame
        gt_list = self.all_gt_data[obj_id][frame_idx]

        # Find the specific entry for the current obj_id
        gt_frame_data = None
        for item in gt_list:
            if item['obj_id'] == obj_id:
                gt_frame_data = item
                break

        # A check in case the object isn't found
        if gt_frame_data is None:
            # Handle error or skip
            return self.__getitem__((idx + 1) % len(self))

        obj_bb = torch.tensor(gt_frame_data['obj_bb'], dtype=torch.int32)

        raw_name = self.models_info.get(obj_id, {}).get('name')
        if not raw_name:
            raw_name = LINEMOD_ID_MAP.get(obj_id, f"Object {obj_id}")
        # Format it
        obj_name = f"{obj_id:02d} - {raw_name}"

        return {
            "rgb_path": rgb_path,      # Return image path, not image tensor
            "obj_id": obj_id,          # Object id (e.g., 1 for Ape)
            "frame_idx": frame_idx,    # Frame id for the class
            "obj_bb": obj_bb,          # Bounding box [x_min, y_min, width, height]
            "obj_name": obj_name,      # Object name (e.g., 'Ape')
        }

class RotationResNetDataset(Dataset):
    def __init__(self, dataset_root, split='train', transform=None):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform

        # Paths
        self.data_dir = os.path.join(dataset_root, 'Linemod_preprocessed', 'data')

        self.samples = []
        self.all_gt_data = {}

        print(f"Initializing {split} dataset using Hash Split...")

        # Iterate over object folders
        for obj_id_str in sorted(os.listdir(self.data_dir)):
            if not obj_id_str.isdigit():
                continue

            obj_id = int(obj_id_str)
            gt_path = os.path.join(self.data_dir, obj_id_str, 'gt.yml')

            if not os.path.exists(gt_path):
                continue

            # Load Ground Truth for this object
            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f)
            self.all_gt_data[obj_id] = gt_data

            # Iterate over frames and apply HASH SPLIT
            for frame_idx_str in gt_data.keys():
                frame_idx = int(frame_idx_str)

                # We create a unique string ID for this specific image
                unique_id = f"{obj_id}_{frame_idx}"

                # We use MD5 to ensure it returns the SAME number every single time you run the code
                hash_object = hashlib.md5(unique_id.encode())
                hex_dig = hash_object.hexdigest()
                hash_int = int(hex_dig, 16) # Convert hex to integer

                # Modulo 100 to get a number between 0 and 99
                random_val = hash_int % 100

                # Define Train/Val threshold (e.g., 80% train)
                is_train = random_val < 80

                # Filter based on the requested split
                if split == 'train' and is_train:
                    self.samples.append((obj_id, frame_idx))
                elif split == 'val' and not is_train:
                    self.samples.append((obj_id, frame_idx))

    def __len__(self):
        return len(self.samples)

    def jitter_bbox(self, bbox, img_w, img_h):
        x, y, w, h = bbox

        x_shift = int(np.random.uniform(-0.1, 0.1) * w) # less aggressive: int(np.random.uniform(-0.05, 0.05) * w) # +/- 5%
        y_shift = int(np.random.uniform(-0.1, 0.1) * h) # less aggressive: int(np.random.uniform(-0.05, 0.05) * w) # +/- 5%
        scale_factor = np.random.uniform(0.95, 1.05) # less aggressive: np.random.uniform(0.98, 1.02)

        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        new_x = x + x_shift - (new_w - w) // 2
        new_y = y + y_shift - (new_h - h) // 2
        new_x = max(0, min(new_x, img_w - 1))
        new_y = max(0, min(new_y, img_h - 1))
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)

        return [new_x, new_y, new_w, new_h]

    def __getitem__(self, idx):
        obj_id, frame_idx = self.samples[idx]
        obj_id_str = f"{obj_id:02d}"
        frame_idx_str = f"{frame_idx:04d}"

        rgb_path = os.path.join(self.data_dir, obj_id_str, 'rgb', f"{frame_idx_str}.png")
        image = cv2.imread(rgb_path)
        if image is None:
          return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image.shape

        # Get the list of all objects in this frame
        gt_list = self.all_gt_data[obj_id][frame_idx]

        # Find the specific entry for the current obj_id
        gt_frame_data = None
        for item in gt_list:
            if item['obj_id'] == obj_id:
                gt_frame_data = item
                break

        # A check in case the object isn't found
        if gt_frame_data is None:
            # Handle error or skip
            return self.__getitem__((idx + 1) % len(self))

        bbox = gt_frame_data['obj_bb']  # [x, y, w, h]

        if self.split == 'train':
            bbox = self.jitter_bbox(bbox, w_img, h_img)

        x, y, w, h = bbox
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)

        if w <= 0 or h <= 0:
            crop = cv2.resize(image, (224, 224))
        else:
            crop = image[y:y+h, x:x+w]
            crop = cv2.resize(crop, (224, 224))

        if self.transform:
            crop = self.transform(crop)

        rot_matrix = np.array(gt_frame_data['cam_R_m2c']).reshape(3, 3)
        rot_quat = R.from_matrix(rot_matrix).as_quat()
        label_quat = torch.tensor(rot_quat, dtype=torch.float32)

        return crop, label_quat, obj_id