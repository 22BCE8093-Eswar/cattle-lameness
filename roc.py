import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

labels_map = {'healthy': 0, 'low_lame': 1, 'medium_lame': 2, 'very_lame': 3}
num_classes = len(labels_map)

class FeatureDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        base_path = "data/features"
        for label in os.listdir(base_path):
            path = os.path.join(base_path, label)
            features = []
            for file in sorted(os.listdir(path)):
                features.append(np.load(os.path.join(path, file)))
            self.data.append(torch.tensor(np.array(features), dtype=torch.float32))
            self.labels.append(labels_map[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

dataset = FeatureDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = LSTMModel()
model.load_state_dict(torch.load("models/lstm_classifier/lstm.pth"))
model.eval()

y_true = []
y_scores = []

with torch.no_grad():
    for x, y in loader:
        output = model(x)
        y_scores.append(output.numpy()[0])
        y_true.append(y.item())

y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
y_scores = np.array(y_scores)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'purple']
class_names = list(labels_map.keys())

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
