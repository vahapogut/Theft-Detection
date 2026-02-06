from ultralytics import YOLO
try:
    print("Loading model...")
    model = YOLO('yolov8n-pose.pt')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
