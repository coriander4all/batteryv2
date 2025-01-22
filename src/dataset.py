import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SOHDataset(Dataset):
    def __init__(
        self,
        csv_path,
        seq_length,
        pred_length,
        scaler=None,
        mode="train",
        val_split=0.2,
        seed=42,
    ):
        self.data = pd.read_csv(csv_path)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.scaler = scaler if scaler else MinMaxScaler()
        self.mode = mode
        self.val_split = val_split
        self.seed = seed

        self.preprocessed_data = self.preprocess()

        if self.mode == "train":
            #découpage en ensembles train/val
            train_size = int((1 - self.val_split) * len(self.preprocessed_data))
            val_size = len(self.preprocessed_data) - train_size
            self.train_data, self.val_data = random_split(
                self.preprocessed_data,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        elif self.mode == "test":
            self.test_data = self.preprocessed_data

    def preprocess(self):
        features = self.data.iloc[:, 2:]
        features = features.apply(pd.to_numeric, errors="coerce")

        features_scaled = self.scaler.fit_transform(features.fillna(0))

        self.data.loc[:, features.columns] = features_scaled

        grouped = self.data.groupby("seq_id")
        sequences = []
        for _, group in grouped:
            group = group.sort_values(by="cycle_index")
            values = group.values[:, 2:]
            for i in range(len(values) - self.seq_length):
                seq_input = values[i : i + self.seq_length]
                available_future = min(
                    len(values) - (i + self.seq_length),
                    self.pred_length
                )
                if available_future > 0:
                    seq_target = values[
                        i + self.seq_length : i + self.seq_length + available_future, -1
                    ]
                    if available_future < self.pred_length:
                        padding = np.full(self.pred_length - available_future, np.nan)
                        seq_target = np.concatenate([seq_target, padding])
                    sequences.append((seq_input, seq_target))
        return sequences

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "test":
            return len(self.test_data)

    def __getitem__(self, index):
        if self.mode == "train":
            seq_input, seq_target = self.train_data[index]
        elif self.mode == "test":
            seq_input, seq_target = self.test_data[index]
        seq_input = np.array(seq_input, dtype=np.float32)
        seq_target = np.array(seq_target, dtype=np.float32)

        target_mask = ~np.isnan(seq_target)
        seq_target = np.nan_to_num(seq_target, 0.0)

        return (
            torch.tensor(seq_input, dtype=torch.float32),
            torch.tensor(seq_target, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.bool),
        )

    def get_validation_data(self):
        if self.mode != "train":
            raise ValueError("Validation data is only available in 'train' mode.")
        val_data = []

        for seq_input, seq_target in self.val_data:
            seq_input = np.asarray(seq_input, dtype=np.float32)
            seq_target = np.asarray(seq_target, dtype=np.float32)
            
            target_mask = ~np.isnan(seq_target)
            seq_target = np.nan_to_num(seq_target, 0.0)
            
            val_data.append(
                (
                    torch.tensor(seq_input, dtype=torch.float32),
                    torch.tensor(seq_target, dtype=torch.float32),
                    torch.tensor(target_mask, dtype=torch.bool),
                )
            )
        return val_data
