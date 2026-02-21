
import cv2
import numpy as np
import os

label_path = 'data/cityscapes/images/train/train/label/train1.png'
if not os.path.exists(label_path):
    print(f"File not found: {label_path}")
else:
    img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load image")
    else:
        print(f"Shape: {img.shape}")
        print(f"Dtype: {img.dtype}")
        unique_values = np.unique(img)
        print(f"Unique values: {unique_values}")
        if len(unique_values) > 50:
             print("Too many unique values, showing first 50:", unique_values[:50])
