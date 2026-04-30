# local_inference.py

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration (UPDATE THESE PATHS) ---
model_path = r'C:\blade_defect_project\best.pt' # e.g., 'C:/Users/ADITHYA SHAJEE/blade_defect_project/models/yolov8s_blade_detect/weights/best.pt'
images_dir = r'C:\blade_defect_project\Testing_Image' # Path to a folder of images

# --- Load the YOLOv8 model ---
model = YOLO(model_path)

# --- Perform inference ---
# Passing a directory to predict() will automatically process all images inside it
results = model.predict(images_dir, conf=0.5, iou=0.5) # conf and iou thresholds can be adjusted

# --- Process and display results ---
for r in results:
    im_array = r.plot()  # plot outputs (bboxes, labels, scores)
    im = Image.fromarray(im_array[..., ::-1])  # RGB BGR conversion for matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(im)
    plt.title(f"Detected defects in {r.path}")
    plt.axis('off')
    plt.show()

    print(f"\nDetection Results for {r.path}:")
    print(r.tojson(indent=2)) # Print results in JSON format