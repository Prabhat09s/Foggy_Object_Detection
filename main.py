from ultralytics import YOLO
import cv2
import os

def run_detection():
    # Load the standard YOLOv8 model (using the nano version for speed)
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # This will download the weights if not present

    # Define source image (using the built-in sample from ultralytics if possible, or a placeholder)
    # Ultralytics can handle URLs directly.
    source = 'https://ultralytics.com/images/bus.jpg'
    
    print(f"Running detection on {source}...")
    
    # Run inference
    results = model(source)

    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        
        # Save results to disk
        result.save(filename='result.jpg')
        print("Detection complete. Result saved to 'result.jpg'.")

if __name__ == "__main__":
    run_detection()
