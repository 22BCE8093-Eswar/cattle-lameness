import os
import torch
import numpy as np
import cv2
from torchvision import models, transforms
from torch import nn
from lstm_model import LSTMModel
import pandas as pd
import json
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- Paths and Device ---
frames_dir = 'data/temp/frames'
crops_dir = 'data/temp/crops'
dataset_dir = 'data/permanent_dataset'
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(crops_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Models ---
yolo_model = YOLO('yolov8n.pt')
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

lstm_model = LSTMModel(input_size=512, hidden_size=128, num_classes=4)
lstm_model.load_state_dict(torch.load('models/lstm_classifier/lstm.pth', map_location=device))
lstm_model = lstm_model.to(device).eval()

classes = ['healthy', 'low_lame', 'medium_lame', 'very_lame']
prediction_map = {'healthy': 0, 'low_lame': 1, 'medium_lame': 2, 'very_lame': 3}
reverse_map = ['Healthy', 'Low Lame', 'Medium Lame', 'Very Lame']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- GUI App ---
class LamenessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cattle Lameness Classifier")

        self.upload_btn = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack()

        self.canvas = tk.Label(self.root)
        self.canvas.pack()

    def upload_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        messagebox.showinfo("Processing", "Processing video, please wait...")
        prediction_data, final_category = self.process_video(video_path)
        self.result_label.config(text=f"Final Prediction: {final_category}")

        self.plot_predictions(prediction_data)
        self.display_frame_with_bbox(video_path)

    def process_video(self, video_path):
        prediction_data = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f'{count}.jpg')
            cv2.imwrite(frame_path, frame)
            count += 1
        cap.release()

        features = []
        valid_frames = []
        for frame_name in sorted(os.listdir(frames_dir), key=lambda x: int(x.split('.')[0])):
            frame_path = os.path.join(frames_dir, frame_name)
            results = yolo_model(frame_path)
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        label_id = int(box.cls[0].item())
                        label_name = yolo_model.names[label_id]
                        if label_name.lower() != 'cow':
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        img = cv2.imread(frame_path)
                        crop = img[y1:y2, x1:x2]
                        crop_path = os.path.join(crops_dir, frame_name)
                        cv2.imwrite(crop_path, crop)

                        img_tensor = transform(crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feature = resnet(img_tensor).cpu().numpy().squeeze()
                        features.append(feature)
                        valid_frames.append(frame_name)

        if not features:
            messagebox.showerror("Error", "No cow detected in the video.")
            return [], "No Prediction"

        features_tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = lstm_model(features_tensor)
            predicted = torch.argmax(output, dim=1).item()

        final_category = classes[predicted].upper()

        for i, fname in enumerate(valid_frames):
            label = final_category.lower()
            prediction_data.append({'frame': fname, 'prediction': final_category})

            label_dir = os.path.join(dataset_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            src_crop_path = os.path.join(crops_dir, fname)
            dst_crop_path = os.path.join(label_dir, fname)
            if os.path.exists(src_crop_path):
                cv2.imwrite(dst_crop_path, cv2.imread(src_crop_path))

        df = pd.DataFrame([{'image_path': os.path.join(label, fname), 'label': label} for fname in valid_frames])
        mapping_csv_path = os.path.join(dataset_dir, 'metadata.csv')
        if os.path.exists(mapping_csv_path):
            df.to_csv(mapping_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(mapping_csv_path, index=False)

        pd.DataFrame(prediction_data).to_csv('predictions.csv', index=False)
        with open('predictions.json', 'w') as f:
            json.dump(prediction_data, f)

        return prediction_data, final_category

    def plot_predictions(self, prediction_data):
        time_steps = list(range(len(prediction_data)))
        predictions = [data['prediction'] for data in prediction_data]
        numerical = [prediction_map[p.lower()] for p in predictions]

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, numerical, marker='o', linestyle='-')
        plt.title("Prediction Trend Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Lameness Level")
        plt.yticks([0, 1, 2, 3], reverse_map)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def display_frame_with_bbox(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error reading video")
            return

        frame_resized = cv2.resize(frame, (640, 360))
        results = yolo_model(frame_resized)

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = yolo_model.names[cls_id]
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label.lower() == 'cow':
                        color = (238, 130, 238)  # Violet for cow
                    else:
                        color = (0, 255, 0)       # Green for others

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

# --- Run the GUI App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LamenessApp(root)
    root.mainloop()
