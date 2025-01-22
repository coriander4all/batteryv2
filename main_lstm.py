import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from src.dataset import SOHDataset
from src.dataloader import LSTM_SOHPredictor
from src.trainer import train, eval

from hyperparams import *

learning_rate = 0.001
num_epochs = 30

#paths
train_csv = "data/processed2/train.csv"
test_csv = "data/processed2/test.csv"

scaler = MinMaxScaler()
train_dataset = SOHDataset(train_csv, seq_length, pred_length, scaler, "train")
val_dataset = train_dataset.get_validation_data()
test_dataset = SOHDataset(test_csv, seq_length, pred_length, scaler, "test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LSTM_SOHPredictor(input_size, hidden_size, num_layers, pred_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(
    num_epochs, model, train_loader, val_loader, criterion, optimizer, "experiments/1"
)
eval(model, test_loader, criterion)
