import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=1, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)  # Get the LSTM output
        print(f"Output shape from LSTM: {out.shape}")  # Debugging the shape of the output
        if len(out.shape) == 3:
            out = out[:, -1, :]  # Take output of last time step for sequence
        else:
            out = out  # Only hidden state is returned, no sequence dimension
        out = self.fc(out)  # Final output layer
        return out
