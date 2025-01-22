import torch
import torch.nn as nn


class LSTM_SOHPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length):
        super(LSTM_SOHPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size // 4, pred_length)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        h0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_size // 2).to(x.device)
        c0_2 = torch.zeros(self.num_layers, batch_size, self.hidden_size // 2).to(x.device)
        
        lstm1_out, _ = self.lstm1(x, (h0_1, c0_1))
        
        lstm2_out, _ = self.lstm2(lstm1_out, (h0_2, c0_2))
        
        out = self.fc1(lstm2_out[:, -1])
        out = self.relu(out)
        out = self.dropout(out)
        predictions = self.fc2(out)
        
        return predictions
