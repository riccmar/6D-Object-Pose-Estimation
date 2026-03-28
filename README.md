# 6D Object Pose Estimation

## Advanced Machine Learning Project

## Members
- Francesco Palmisani
- Giosue Pinto
- Riccardo Marconi
- Samuele Reale

## Enhancing 6D Object Pose Estimation
This project aims to explore and implement advanced techniques for 6D pose estimation using RGB-D images. The goal is to build an end-to-end pipeline for estimating the 6D pose of objects by initially replicating a model that uses just RGB images. The pipeline will then be enhanced by incorporating depth information to improve accuracy in the pose predictions.
You will adapt and implement the methodology, starting from pose prediction and then extend the model with your own innovative improvements.

## Usage

**Note:** All scripts and commands must be run from the root directory of the repository `aml_project/`.

### 1. Setup
Install dependencies and download the dataset (data will be saved in `data/linemod`).
```bash
pip install -r requirements.txt
python utils/download_dataset.py
```

### 2. Running the scripts (Example using YOLO for segmentation)
Note: The general workflow applies to other models located in `scripts/*` (e.g. resnet, rgbd_fusion_net, refine_net), but specific arguments such as model paths or thresholds may vary. Use the `--help` flag with any script to see its required parameters.

**Training**
```bash
python scripts/extension/yolo/yolo_train_seg.py --epochs 100 --batch_size 64
```

**Evaluation**
```bash
python scripts/extension/yolo/yolo_eval_seg.py --model_path <path_to_model> --batch_size 64 --conf 0.25
```

**Inference**
```bash
python scripts/extension/yolo/yolo_inference_seg.py --model_path <path_to_model> --samples 3 --conf 0.25
```
