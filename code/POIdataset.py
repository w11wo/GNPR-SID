from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os

current_dir = os.getcwd()

# Pid,Uid,Catname,Region,Time,neighbors,forward_neighbors


class EmbDataset(Dataset):
    def __init__(self, datapath):
        poi_info_path = Path(current_dir) / datapath
        data_dir = poi_info_path.parent

        catname_mapping = pd.read_csv(data_dir / "catname_mapping.csv")
        region_mapping = pd.read_csv(data_dir / "region_mapping.csv")
        pid_mapping = pd.read_csv(data_dir / "pid_mapping.csv")

        data = pd.read_csv(poi_info_path)
        self.ids = data["Pid"].tolist()
        self.catname_raw = data["Catname"].tolist()
        self.region_raw = data["Region"].tolist()
        self.time_raw = data["Time"].apply(eval).tolist()
        self.uid_raw = data["Uid"].apply(eval).tolist()

        self.time_num = 24
        self.cat_num = len(catname_mapping) + 1
        self.region_num = len(region_mapping) + 1
        self.neighbor_num = len(pid_mapping) + 1

    @staticmethod
    def _to_one_hot(indices, num_classes):
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[indices] = 1
        return one_hot

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cat = self._to_one_hot(self.catname_raw[idx], self.cat_num)
        region = self._to_one_hot(self.region_raw[idx], self.region_num)
        time = self._to_one_hot(self.time_raw[idx], self.time_num)
        neighbor = self._to_one_hot(self.uid_raw[idx], self.neighbor_num)
        return self.ids[idx], torch.cat([cat, region, time, neighbor])
