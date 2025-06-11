from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8
frame_dir = "data/frames"
crop_dir = "data/crops"
os.makedirs(crop_dir, exist_ok=True)

def detect_and_crop(label):
    label_path = os.path.join(frame_dir, label)
    crop_label_dir = os.path.join(crop_dir, label)
    os.makedirs(crop_label_dir, exist_ok=True)

    for file in os.listdir(label_path):
        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path)
        results = model(img)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = img[y1:y2, x1:x2]
                save_path = os.path.join(crop_label_dir, file)
                cv2.imwrite(save_path, cropped)

for label in os.listdir(frame_dir):
    detect_and_crop(label)
