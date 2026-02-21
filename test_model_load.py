
import torch
from models.msffa_yolo import MSFFA_YOLO
import traceback

print("=" * 60)
print("MSFFA-YOLO MODEL VALIDATION TEST")
print("=" * 60)

try:
    # Test model instantiation
    print("\n1. Testing model instantiation...")
    model = MSFFA_YOLO(num_classes=8)
    print("   ✅ Model created successfully")
    
    # Test forward pass with dummy input
    print("\n2. Testing forward pass with dummy input...")
    dummy_input = torch.randn(2, 3, 256, 256)  # batch_size=2, channels=3, H=256, W=256
    print(f"   Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        yolo_preds, restored_img = model(dummy_input)
    
    print(f"   ✅ Forward pass successful!")
    print(f"   Restored image shape: {restored_img.shape}")
    print(f"   Number of YOLO prediction scales: {len(yolo_preds)}")
    for i, pred in enumerate(yolo_preds):
        print(f"   YOLO pred[{i}] shape: {pred.shape}")
    
    # Test model parameters
    print("\n3. Analyzing model parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test with training mode
    print("\n4. Testing training mode...")
    model.train()
    output = model(dummy_input)
    print(f"   ✅ Training mode works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ ERROR DETECTED: {str(e)}")
    print("=" * 60)
    print("\nFull traceback:")
    traceback.print_exc()
    print("\n" + "=" * 60)
