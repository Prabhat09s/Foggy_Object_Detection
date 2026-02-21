import torch
import torch.nn as nn
from .msffa.msffa import MSFFANet
from .yolo.yolov7 import YOLOv7

class MSFFA_YOLO(nn.Module):
    """
    Combined MSFFA and YOLOv7 model.
    The MSFFA module acts as a front-end restoration subnet, cleaning the image before passing it to YOLO.
    """
    def __init__(self, num_classes=8):
        super(MSFFA_YOLO, self).__init__()
        # Restoration Subnet
        self.msffa = MSFFANet()
        
        # Detection Subnet
        self.yolo = YOLOv7(num_classes=num_classes)

    def forward(self, x):
        # 1. Restoration
        # msffa returns (features, restored_image)
        # We need to decide what to pass to YOLO.
        # Ideally, we pass the 'restored_image' if it reconstructs the RGB,
        # OR we pass the 'enhanced_features' if they are compatible.
        # Given the Decoder outputs an image-like tensor (tanh, 3 channels), 
        # we can feed that directly into YOLO.
        
        enhanced_features, restored_image = self.msffa(x)
        
        # 2. Detection (on the restored image)
        # We might need to un-normalize or scale if MSFFA output is Tanh (-1, 1) and YOLO expects (0, 1)
        # Assuming YOLO expects standard normalized input.
        
        # If training, we might want to return both for separate losses
        detections = self.yolo(restored_image)
        
        if self.training:
            # detections is a list of features
            return detections, restored_image
        else:
            # detections is (predictions, features)
            return detections, restored_image
