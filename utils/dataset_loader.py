import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FoggyDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load Label
        # Filename assumption: image.png -> image.txt
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, basename + ".txt")
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # class x_c y_c w h
                    data = line.strip().split()
                    if len(data) == 5:
                        cls = int(data[0])
                        xc, yc, w, h = map(float, data[1:])
                        boxes.append([0, cls, xc, yc, w, h]) # 0 is placeholder for batch index

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Target shape: [num_boxes, 6] (batch_idx, class, x, y, w, h)
        if len(boxes) > 0:
            targets = torch.tensor(boxes)
        else:
            targets = torch.zeros((0, 6))

        return img, targets

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    
    # Add batch index to targets
    new_targets = []
    for i, t in enumerate(targets):
        if t.numel() > 0:
            t[:, 0] = i # Set batch index
            new_targets.append(t)
    
    if len(new_targets) > 0:
        targets = torch.cat(new_targets, 0)
    else:
        targets = torch.zeros((0, 6))
        
    return imgs, targets

def create_dataloader(img_path, label_path, batch_size=8, shuffle=True):
    dataset = FoggyDataset(img_path, label_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
