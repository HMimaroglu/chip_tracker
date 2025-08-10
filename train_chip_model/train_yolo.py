import os
import torch
from ultralytics import YOLO

# 1. Define the path to the dataset's data.yaml file
data_yaml_path = os.path.expanduser("~/chip_photos/Poker Chip Color.v1i.yolov11/data.yaml")

# 2. Load a pre-trained YOLOv11 model
# Using a small, fast model for demonstration.
# You can choose other models like yolov11s, yolov11m, etc.
model = YOLO("yolov11n.pt")

# 3. Train the model
results = model.train(
    data=data_yaml_path,
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    batch=16,  # Batch size
    name="yolov11_chip_model",  # Name for the trained model
)

# 4. Save the trained model
# The model is automatically saved in the 'runs/train/yolov11_chip_model' directory
print(f"Trained model saved in: {results.save_dir}")

# To run this script:
# 1. Make sure you have the ultralytics package installed: pip install ultralytics
# 2. Make sure you have the 'yolov11n.pt' file in the same directory as this script.
# 3. Run the script: python train_yolo.py
