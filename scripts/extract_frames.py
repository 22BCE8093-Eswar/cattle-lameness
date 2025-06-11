import cv2
import os

video_dir = "data/videos"
frame_dir = "data/frames"
os.makedirs(frame_dir, exist_ok=True)

def extract_frames(video_path, label):
    cap = cv2.VideoCapture(video_path)
    count = 0
    label_dir = os.path.join(frame_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{label}_{count:04d}.jpg"
        cv2.imwrite(os.path.join(label_dir, filename), frame)
        count += 1

for video_file in os.listdir(video_dir):
    label = video_file.split(".")[0]
    extract_frames(os.path.join(video_dir, video_file), label)
