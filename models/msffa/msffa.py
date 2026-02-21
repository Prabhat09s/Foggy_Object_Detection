import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .feature_conversion import FeatureConversation

class MSFFANet(nn.Module):
    def __init__(self):
        super(MSFFANet, self).__init__()
        self.encoder = Encoder()
        self.feature_conv = FeatureConversation(256) # Encoder output is 256 channels
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        # MSFFA Step: Enhance features
        enhanced_features = self.feature_conv(features)
        # Restore image (optional for visual debugging, but used for loss)
        restored_image = self.decoder(enhanced_features)
        
        # We can return both features (for YOLO) and restored image (for loss)
        return enhanced_features, restored_image
