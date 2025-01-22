import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class DataCleaner:
    def __init__(self, dir_path, output_path, test_rate=0.15):
        self.dir_path = dir_path
        self.output_path = output_path
        self.test_rate = test_rate
        self.columns_to_keep = [
            "cycle_index",
            "discharge_capacity",
            "discharge_energy",
            "charge_energy",
            "dc_internal_resistance",
            "temperature_maximum",
            "temperature_average",
            "temperature_minimum",
            # "date_time_iso",
            "energy_efficiency",
            "charge_throughput",
            "energy_throughput",
            "charge_duration",
            "time_temperature_integrated",
            # "paused",
        ]
        self.df = pd.DataFrame(columns=["seq_id"] + self.columns_to_keep)
        self.files = os.listdir(dir_path)

    def load_data(self):
        for idx, file in enumerate(self.files):
            print(f"file {idx + 1} / {len(self.files)}")
            filename = file.split("_")
            seq_id = filename[1] + "_" + filename[2]
            with open(self.dir_path + file, "r") as file:
                data = json.load(file)

            summaries_df = pd.DataFrame(data["summary"])
            summaries_df = summaries_df[self.columns_to_keep]
            summaries_df["seq_id"] = seq_id

            self.df = pd.concat([self.df, summaries_df], ignore_index=True)
            # if idx == 3:
            #     break
        print(
            f"Data loaded, {self.df.shape[0]} rows, {self.df['seq_id'].nunique()} sequences"
        )

    def clean_data(self, tolerance=0.2):
        self.df["diff"] = self.df.groupby("seq_id")["discharge_capacity"].diff()
        self.df = self.df[self.df["diff"].abs() < tolerance]
        self.df = self.df.drop(columns=["diff"])

    def prepare_data(self):
        # soh calculation
        self.df["soh"] = self.df.groupby("seq_id")["discharge_capacity"].transform(
            lambda x: x / x.max()
        )

        # split data in train and test
        unique_seq_ids = self.df["seq_id"].unique()
        train_seq_ids, test_seq_ids = train_test_split(
            unique_seq_ids, test_size=self.test_rate, random_state=42
        )
        self.train_df = self.df[self.df["seq_id"].isin(train_seq_ids)]
        self.test_df = self.df[self.df["seq_id"].isin(test_seq_ids)]

    def save_data(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.train_df.to_csv(self.output_path + "train.csv", index=False)
        self.test_df.to_csv(self.output_path + "test.csv", index=False)
