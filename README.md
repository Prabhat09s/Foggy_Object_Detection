# MSFFA-YOLO: Multiclass Object Detection in Foggy Weather

This project implements the MSFFA-YOLO architecture for object detection in foggy traffic scenes, combining a Multi-Scale Feature Fusion Attention (MSFFA) restoration subnet with a YOLOv7 detection subnet.

## Project Structure

- `models/`: Contains the network definitions.
  - `msffa/`: The restoration subnet (Encoder, Decoder, Attention).
  - `yolo/`: The detection subnet (YOLOv7 blocks).
  - `msffa_yolo.py`: The combined model.
- `data/`: Dataset directory.
- `utils/`: Helper scripts (Dataloader, Soft-NMS).
- `loss/`: Loss functions.
- `train.py`: Main training script.
- `create_dummy_dataset.py`: Script to generate synthetic data for testing.

## Prerequisites

Install the required packages:

```bash
pip install ultralytics opencv-python numpy torch torchvision matplotlib Pillow
```

## Setup & Running

1.  **Generate Dummy Data** (If you don't have the real dataset yet):
    ```bash
    python create_dummy_dataset.py
    ```
    This will create synthetic foggy images in `data/foggy_cityscapes/FC01`.

2.  **Train the Model**:
    ```bash
    python train.py
    ```
    This will start the training loop using the dummy data (or real data if placed correctly).

## Dataset Preparation (Real Data)

To use real data (Foggy Cityscapes):

1.  Download **Cityscapes** and **Foggy Cityscapes** from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/).
2.  Extract them into `data/cityscapes` and `data/foggy_cityscapes`.
3.  Ensure the directory structure matches the one created by `setup_dirs.py`.
4.  Convert annotations to YOLO format (class x_center y_center width height).

## Features Implemented

- [x] Directory Structure
- [x] MSFFA Restoration Subnet (Encoder-Decoder + Attention)
- [x] YOLOv7 Detection Subnet
- [x] Custom Dataloader
- [x] Combined Loss Function
- [x] Soft-NMS (Utility)
- [x] Training Loop
