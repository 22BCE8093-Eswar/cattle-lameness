import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

labels_map = {'healthy': 0, 'low_lame': 1, 'medium_lame': 2, 'very_lame': 3}

class FeatureDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        for label in os.listdir("data/features"):
            path = os.path.join("data/features", label)
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
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

dataset = FeatureDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = LSTMModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

torch.save(model.state_dict(), "models/lstm_classifier/lstm.pth")
