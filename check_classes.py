
import cv2
import numpy as np
import os

label_dir = 'data/cityscapes/labels_source/train'
classes_found = set()

if os.path.exists(label_dir):
    for fname in os.listdir(label_dir)[:20]: # Check first 20
        path = os.path.join(label_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        uniques = np.unique(img)
        classes_found.update(uniques)

print("Classes found:", sorted(list(classes_found)))
