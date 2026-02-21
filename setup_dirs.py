import os

dirs = [
    "data/cityscapes/images/train", "data/cityscapes/images/val", "data/cityscapes/images/test",
    "data/cityscapes/labels/train", "data/cityscapes/labels/val", "data/cityscapes/labels/test",
    "data/foggy_cityscapes/FC005/images", "data/foggy_cityscapes/FC005/labels",
    "data/foggy_cityscapes/FC01/images", "data/foggy_cityscapes/FC01/labels",
    "data/foggy_cityscapes/FC02/images", "data/foggy_cityscapes/FC02/labels",
    "data/rtts/images", "data/rtts/labels",
    "models/yolo", "models/msffa",
    "loss", "utils", "configs", "weights",
    "results/logs", "results/plots", "results/predictions",
    "report"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    # Create __init__.py in python module folders
    if "models" in d or "loss" in d or "utils" in d:
        init_path = os.path.join(d, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                pass

print("Directory structure created.")
