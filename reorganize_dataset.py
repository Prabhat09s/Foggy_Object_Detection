
import os
import shutil

base_path = os.path.abspath('data/cityscapes')

# Define moves (source -> destination)
# We want data/cityscapes/images/train/file.png
moves = [
    ('images/train/train/img', 'images/train'),
    ('images/train/train/label', 'labels_source/train'), 
    ('images/val/val/img', 'images/val'),
    ('images/val/val/label', 'labels_source/val'),
]

for src_rel, dst_rel in moves:
    src = os.path.join(base_path, src_rel)
    dst = os.path.join(base_path, dst_rel)
    
    if os.path.exists(src):
        print(f"Processing {src}...")
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.exists(d):
                # print(f"Warning: {d} already exists. Skipping.")
                continue
            shutil.move(s, d)
        print(f"Moved files to {dst}")
            
        # Cleanup
        try:
            os.rmdir(src)
            print(f"Removed {src}")
            # Try to remove parent 'train/train' or 'val/val'
            parent = os.path.dirname(src)
            # Only remove if empty
            if not os.listdir(parent):
                os.rmdir(parent)
                print(f"Removed {parent}")
        except OSError as e:
            print(f"Cleanup info: {e}")
    else:
        print(f"Source {src} does not exist.")

print("Reorganization complete.")
