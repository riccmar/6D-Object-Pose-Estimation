import os
import yaml
import torch
import hashlib
from torch.utils.data import Dataset

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