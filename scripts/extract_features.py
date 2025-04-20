import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

crop_dir = "data/crops"
feature_dir = "data/features"
os.makedirs(feature_dir, exist_ok=True)

for label in os.listdir(crop_dir):
    label_path = os.path.join(crop_dir, label)
    feat_label_dir = os.path.join(feature_dir, label)
    os.makedirs(feat_label_dir, exist_ok=True)

    for file in os.listdir(label_path):
        img = Image.open(os.path.join(label_path, file)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = resnet(tensor).squeeze().numpy()
        np.save(os.path.join(feat_label_dir, file.replace('.jpg', '.npy')), feature)
