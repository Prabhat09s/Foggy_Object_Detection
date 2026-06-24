import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use pretrained or your custom trained model
    model = YOLO("yolov8n.pt")  
    # OR use your trained model:
    # model = YOLO("runs/detect/train/weights/best.pt")

    return model, device


model, device = load_model()

st.title("Foggy Object Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload Foggy Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Run Detection"):

        # Run YOLOv8 detection
        results = model(image_np, device=device)

        # Get annotated image
        annotated_image = results[0].plot()

        # Convert BGR → RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.image(annotated_image, caption="Detection Result", use_column_width=True)

        # Show detected objects
        boxes = results[0].boxes

        if boxes is not None:

            st.write("Detected Objects:")

            for box in boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = model.names[cls_id]

                st.write(f"{class_name} : {conf:.2f}")

        st.success("Detection completed!")