"""Example Train a YOLO model on our dataset.
Author: Przemek Sekula
Created on: October 2024
"""
from ultralytics import YOLO

# Load a pre-trained YOLO model (e.g., YOLOv8)
# yolov8n is a small model. You can use yolov8m, yolov8l, etc.
model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    model.train(data='data_config.yaml', epochs=10, imgsz=640)

    model.val()
    model.save('yolo11_finetuned.pt')
