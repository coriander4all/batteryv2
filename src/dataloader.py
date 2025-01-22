import torch
import torch.nn as nn


class LSTM_SOHPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length):
        super(LSTM_SOHPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, pred_length)
    
    def forward(self, x):
        #initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        predictions = self.fc(lstm_out[:, -1])
        return predictions
