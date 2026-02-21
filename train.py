import torch
import torch.optim as optim
from models.msffa_yolo import MSFFA_YOLO
from utils.dataset_loader import create_dataloader
from loss.combined_loss import TotalLoss
import os

def train():
    # settings
    epochs = 5 # Small for demo
    batch_size = 4
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    # Training usually requires Foggy Image -> Clear Image (for restoration) AND Labels (for detection)
    # Our dataloader currently yields (foggy_img, labels).
    # To properly train restoration, we need the clear counterparts.
    # For now, we will self-supervise or assume clear=foggy for the dummy run, 
    # OR we ignore restoration loss if clear images aren't paired in our simple loader.
    train_loader = create_dataloader("data/cityscapes/images/train", "data/cityscapes/labels/train", batch_size=batch_size)
    
    # Model
    model = MSFFA_YOLO(num_classes=8).to(device)
    
    # Loss & Optimizer
    criterion = TotalLoss() # Note: logic inside needs real clear images to be meaningful
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward
            # restored_img is what MSFFA produced.
            yolo_preds, restored_img = model(imgs)
            
            # Loss Calculation
            # In a real scenario, 'imgs' are foggy, and we need 'clear_imgs' for loss.
            # Using 'imgs' as target for restoration is just an autoencoder (denoising absent).
            # We use imgs itself just to make the code runnable without error.
            loss = criterion(restored_img, imgs, yolo_preds, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{epochs}] complete. Average Loss: {total_loss/len(train_loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"weights/msffa_yolo_epoch_{epoch+1}.pt")
    
    print("Training complete. Model saved.")

import traceback

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Training failed: {e}")
        print("Did you run 'python create_dummy_dataset.py' to generate data?")
