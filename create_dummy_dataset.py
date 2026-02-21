import cv2
import numpy as np
import os
import random

def create_dummy_foggy_data(base_path, num_images=10):
    """
    Creates dummy foggy images and YOLO labels for testing the pipeline.
    """
    # Categories: person, rider, car, truck, bus, motorcycle, bicycle, caravan
    # IDs: 0..7
    
    sets = ['train', 'val']
    
    # 1. Create Foggy Cityscapes Dummy
    for s in sets:
        img_dir = os.path.join(base_path, 'foggy_cityscapes', 'FC01', 'images', s)
        lbl_dir = os.path.join(base_path, 'foggy_cityscapes', 'FC01', 'labels', s)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for i in range(num_images):
            # Create a dummy image (grayish/foggy background)
            img = np.ones((640, 640, 3), dtype=np.uint8) * 200 # Foggy white/gray
            
            # Draw some random rectangles (objects)
            label_str = ""
            for _ in range(random.randint(1, 5)):
                cls_id = random.randint(0, 7)
                x1, y1 = random.randint(50, 500), random.randint(50, 500)
                w, h = random.randint(50, 100), random.randint(50, 100)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, -1)
                
                # YOLO Format: class x_center y_center width height (normalized)
                xc = (x1 + w/2) / 640
                yc = (y1 + h/2) / 640
                wn = w / 640
                hn = h / 640
                label_str += f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"

            # Add artificial fog (simple blend)
            fog = np.ones_like(img) * 255
            alpha = 0.5
            img = cv2.addWeighted(img, 1-alpha, fog, alpha, 0)

            # Save
            filename = f"dummy_{i:04d}"
            cv2.imwrite(os.path.join(img_dir, filename + ".png"), img)
            with open(os.path.join(lbl_dir, filename + ".txt"), 'w') as f:
                f.write(label_str)
                
    print(f"Created {num_images} dummy images and labels in {base_path}/foggy_cityscapes/FC01")

if __name__ == "__main__":
    create_dummy_foggy_data("data")
