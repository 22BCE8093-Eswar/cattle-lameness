# ANUGO-AI: Cattle Health Monitoring System 🐄🤖

**ANUGO-AI** is a multi-modal cattle monitoring platform designed to detect lameness and other health issues in livestock using computer vision and deep learning techniques. Built with scalability and affordability in mind, the system is intended for deployment in rural and resource-constrained environments. The system is non-invasive, providing farmers with early alerts to reduce animal suffering and economic loss.

---

## 🚀 Project Overview

ANUGO-AI leverages a hybrid deep learning architecture combining **YOLOv8**, **ResNet-50**, and **LSTM** to analyze cattle gait and posture from video footage, enabling accurate classification of lameness.

- 📹 **Video Input:** Monitors cattle using HD webcams or IP cameras.
- 🧠 **Deep Learning:** Utilizes state-of-the-art models trained on annotated cattle movement data.
- 🌱 **Edge-Ready:** Optimized for low-resource deployments using CPU and Jetson devices.
- 📊 **Accuracy:** Overall detection accuracy: **95%**, with YOLOv8 achieving **98.98%**.

---

## 🧰 System Design & Architecture

### Pipeline:

1. Capture cattle movement via camera.
2. Detect cattle using fine-tuned YOLOv8.
3. Extract spatial features using ResNet-50.
4. Analyze gait sequences with LSTM.
5. Output lameness classification.

> Developed and tested on **Google Colab**, using **Google Drive** for data storage.

---

## 🖥️ Hardware Requirements

| Component      | Specification                              |
|----------------|---------------------------------------------|
| **Camera**     | HD Webcam/IP Camera (720p minimum, 30fps)   |
| **CPU**        | Intel i7 (8th Gen+) / AMD Ryzen 7           |
| **RAM**        | 16 GB minimum (32 GB recommended)           |
| **GPU**        | NVIDIA RTX 3060 or higher *(Optional)*      |
| **Storage**    | 1 TB SSD for video & model storage          |

---

## 💻 Software Stack

- **Operating System:** Ubuntu 20.04 LTS / Windows 10 (64-bit)
- **Language:** Python 3.8+
- **IDE:** Jupyter Notebook, VS Code, PyCharm

### Core Libraries

| Category            | Libraries/Frameworks           |
|---------------------|--------------------------------|
| Deep Learning       | `PyTorch`, `Torchvision`       |
| Object Detection    | `YOLOv8` (Ultralytics)         |
| Computer Vision     | `OpenCV`, `imageio`            |
| Annotation Tools    | `CVAT`, `Roboflow`             |

---

## 🧪 Data Processing Workflow

- 🎞️ **Video to Frames:** Each video yields ~13 frames per second.
- 📐 **Resizing:**
  - YOLOv8: 640x640 px  
  - Neural Network Input: 180x180 px
- 🌀 **Augmentation:** Random rotations (±10°), horizontal flips.
- 🧪 **Normalization:** Pixel values scaled to [-1, 1].

---

## 🧠 Model Architecture

### Hybrid Architecture:

- **YOLOv8:** Pre-trained on COCO, fine-tuned to detect and track cattle.
- **ResNet-50:** Extracts high-dimensional spatial features.
- **LSTM:** Processes feature sequences (20–30 frames) to analyze gait and classify as *Normal* or *Lame*.

### Training Setup:

- **Epochs:** 50  
- **Batch Size:** 16  
- **Learning Rate:** 0.0001  
- **Optimizer:** Adam

---

## 📈 Performance & Optimization

| Component     | Accuracy/Status                              |
|---------------|-----------------------------------------------|
| **YOLOv8**     | 98.98% detection accuracy                     |
| **Full System**| 95% lameness classification accuracy          |
| **CPU Support**| Optimized for PyTorch CPU backends            |
| **Edge Devices**| Supports Jetson Nano / Raspberry Pi (via ONNX)|

### Optimization Techniques:

- Model quantization and pruning  
- ONNX export for lightweight inference  
- Multi-threaded execution for real-time performance on CPUs  

---

## ⚙️ Deployment

- ✅ **Google Colab + Drive** for model training
- 🔁 **ONNX Conversion** for edge/IoT deployment
- 🧩 **Future Scope:** REST API integration via FastAPI, Dockerized microservices

---

## 📸 Sample Workflow (Mermaid Diagram)

```mermaid
graph LR
A[Camera Feed] --> B[YOLOv8 Detection]
B --> C[ResNet-50 Feature Extraction]
C --> D[LSTM Temporal Analysis]
D --> E[Lameness Prediction]

