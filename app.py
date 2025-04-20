from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO
import os
import sys
from werkzeug.utils import secure_filename

# Paths
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from lstm_model import LSTMModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
yolo_model = YOLO('yolov8n.pt')
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = nn.Identity()
resnet.eval()

lstm_model = LSTMModel(input_size=512, hidden_size=128, num_classes=4)
lstm_model.load_state_dict(torch.load('models/lstm_classifier/lstm.pth', map_location='cpu'))
lstm_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
lstm_model = lstm_model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_labels = ['Healthy', 'Low Lame', 'Medium Lame', 'Very Lame']

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    sequence = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_model.names[cls]

            if name.lower() == "cow":
                crop = frame[y1:y2, x1:x2]
                try:
                    crop_tensor = transform(crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feature = resnet(crop_tensor).cpu().numpy().squeeze()
                    sequence.append(feature)
                except:
                    continue

                if len(sequence) >= 4:
                    input_seq = torch.tensor([sequence[-4:]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        output = lstm_model(input_seq)
                        pred = torch.argmax(output, dim=1).item()
                        lameness_label = class_labels[pred]
                else:
                    lameness_label = "Analyzing..."

                label = f"Cow - {lameness_label} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                label = f"{name} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_live():
    return process_video(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_live(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return redirect(url_for('uploaded_video', filename=filename))

@app.route('/uploaded/<filename>')
def uploaded_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
