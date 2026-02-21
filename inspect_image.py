
import cv2
import os

img_path = 'data/cityscapes/images/train/train/img/train1.png'
if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
else:
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
    else:
        print(f"Image Shape: {img.shape}")
