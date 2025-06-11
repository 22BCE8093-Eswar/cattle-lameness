from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from flask_cors import CORS
CORS(app)
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt

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
sequence = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/plot')
def plot_page():
    return render_template('plot.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global sequence
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # Decode base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = yolo_model(frame)[0]
    lameness_label = "Analyzing..."

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

            label = f"Cow - {lameness_label} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            label = f"{name} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'frame': 'data:image/jpeg;base64,' + encoded_frame})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect('/upload')
    file = request.files['file']
    if file.filename == '':
        return redirect('/upload')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return redirect(url_for('uploaded_video', filename=filename))

@app.route('/uploaded/<filename>')
def uploaded_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    def generate():
        cap = cv2.VideoCapture(video_path)
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
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_plot', methods=['POST'])
def analyze_plot():
    if 'file' not in request.files:
        return redirect('/plot')

    file = request.files['file']
    if file.filename == '':
        return redirect('/plot')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    sequence = []
    predictions = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
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
                        predictions.append(pred)
    cap.release()

    if not predictions:
        return render_template('plot.html', plot_url=None, final_label="No predictions")

    plt.figure(figsize=(10, 4))
    plt.plot(predictions, marker='o', linestyle='-', color='blue')
    plt.title("Lameness Prediction Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Lameness Class")
    plt.yticks(ticks=range(4), labels=class_labels)
    plt.grid(True)
    plot_filename = 'static/uploads/plot.png'
    plt.savefig(plot_filename)
    plt.close()

    most_common_class = Counter(predictions).most_common(1)[0][0]
    final_label = class_labels[most_common_class]

    return render_template('plot.html', plot_url=plot_filename, final_label=final_label)

if __name__ == '__main__':
    app.run(debug=True)
