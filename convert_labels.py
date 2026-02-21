
import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
base_path = 'data/cityscapes'
dirs = ['train', 'val']

# Mapping: labelId -> class_id
id_map = {
    24: 0, # person
    25: 1, # rider
    26: 2, # car
    27: 3, # truck
    28: 4, # bus
    31: 5, # train
    32: 6, # motorcycle
    33: 7  # bicycle
}

def convert_dir(split):
    src_dir = os.path.join(base_path, 'labels_source', split)
    dst_dir = os.path.join(base_path, 'labels', split)
    
    if not os.path.exists(src_dir):
        print(f"{src_dir} does not exist. Skipping.")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    
    print(f"Converting {len(files)} labels in {split}...")
    
    count_saved = 0
    for fname in tqdm(files):
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            continue
            
        if len(img.shape) == 3:
            img = img[:, :, 0] # Assume grayscale/index in first channel
            
        H, W = img.shape
        
        labels = []
        
        for label_id, class_id in id_map.items():
            mask = (img == label_id).astype(np.uint8)
            
            if np.sum(mask) < 5: # Skip tiny noise
                continue
                
            # Connect components
            num_labels, labels_im = cv2.connectedComponents(mask)
            
            for i in range(1, num_labels): # 0 is background
                component_mask = (labels_im == i).astype(np.uint8)
                
                # Get bounding box
                y_indices, x_indices = np.where(component_mask)
                if len(y_indices) == 0: continue
                
                x_min = np.min(x_indices)
                x_max = np.max(x_indices)
                y_min = np.min(y_indices)
                y_max = np.max(y_indices)
                
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                
                # Filter small boxes
                if w < 3 or h < 3:
                    continue
                
                # Normalize
                x_center = (x_min + w / 2) / W
                y_center = (y_min + h / 2) / H
                w_norm = w / W
                h_norm = h / H
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Save txt
        txt_name = fname.replace('.png', '.txt')
        with open(os.path.join(dst_dir, txt_name), 'w') as f:
            f.write('\n'.join(labels))
        count_saved += 1
            
    print(f"Converted {count_saved} files for {split}.")

if __name__ == "__main__":
    for d in dirs:
        convert_dir(d)
