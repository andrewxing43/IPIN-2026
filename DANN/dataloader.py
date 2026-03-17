# dataloader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import rssi_to_fingerprint_image

class LabeledRSSIDataset(Dataset):
    """
    用于源域（train.csv）：有 (x,y) 标签；域标签=0
    CSV 列: 0..174 -> RSSI(175), 175->x, 176->y, 177->floor(忽略)
    """
    def __init__(self, csv_path, rssi_cols=slice(0,175), x_col=175, y_col=176,
                 R_ref=-30.0, eta=2.0, missing_value=-200.0):
        df = pd.read_csv(csv_path)
        self.rssi = df.iloc[:, rssi_cols].to_numpy(np.float32)     # (N,175)
        self.xy   = df.iloc[:, [x_col, y_col]].to_numpy(np.float32) # (N,2)
        self.R_ref, self.eta, self.missing_value = R_ref, eta, missing_value

    def __len__(self): return self.rssi.shape[0]

    def __getitem__(self, idx):
        rssi = self.rssi[idx]  # (175,)
        img  = rssi_to_fingerprint_image(
            rssi, R_ref=self.R_ref, eta=self.eta, missing_value=self.missing_value
        )  # (175,175)
        img  = torch.from_numpy(img).unsqueeze(0)  # (1,175,175)
        xy   = torch.from_numpy(self.xy[idx])      # (2,)
        domain = torch.tensor(0, dtype=torch.long) # 源域=0
        return img, xy, domain

class UnlabeledRSSIDataset(Dataset):
    """
    用于目标域（test.csv）：训练时只参与域判别；域标签=1
    评测时另有专门的 EvaluationDataset 读取 (x,y)。
    """
    def __init__(self, csv_path, rssi_cols=slice(0,175),
                 R_ref=-30.0, eta=2.0, missing_value=-200.0):
        df = pd.read_csv(csv_path)
        self.rssi = df.iloc[:, rssi_cols].to_numpy(np.float32)
        self.R_ref, self.eta, self.missing_value = R_ref, eta, missing_value

    def __len__(self): return self.rssi.shape[0]

    def __getitem__(self, idx):
        rssi = self.rssi[idx]
        img  = rssi_to_fingerprint_image(
            rssi, R_ref=self.R_ref, eta=self.eta, missing_value=self.missing_value
        )
        img  = torch.from_numpy(img).unsqueeze(0)  # (1,175,175)
        domain = torch.tensor(1, dtype=torch.long) # 目标域=1
        return img, domain

class EvaluationDataset(Dataset):
    """ 测试评估用：从 test.csv 读取 (x,y) 以计算 ALE """
    def __init__(self, csv_path, rssi_cols=slice(0,175), x_col=175, y_col=176,
                 R_ref=-30.0, eta=2.0, missing_value=-200.0):
        df = pd.read_csv(csv_path)
        self.rssi = df.iloc[:, rssi_cols].to_numpy(np.float32)
        self.xy   = df.iloc[:, [x_col, y_col]].to_numpy(np.float32)
        self.R_ref, self.eta, self.missing_value = R_ref, eta, missing_value

    def __len__(self): return self.rssi.shape[0]

    def __getitem__(self, idx):
        rssi = self.rssi[idx]
        img  = rssi_to_fingerprint_image(
            rssi, R_ref=self.R_ref, eta=self.eta, missing_value=self.missing_value
        )
        img  = torch.from_numpy(img).unsqueeze(0)
        xy   = torch.from_numpy(self.xy[idx])
        return img, xy

def make_loaders(train_csv, test_csv, batch_size=64, num_workers=2,
                 R_ref=-30.0, eta=2.0, missing_value=-200.0):
    src_ds = LabeledRSSIDataset(train_csv, R_ref=R_ref, eta=eta, missing_value=missing_value)
    tgt_ds = UnlabeledRSSIDataset(test_csv, R_ref=R_ref, eta=eta, missing_value=missing_value)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return src_loader, tgt_loader

def make_eval_loader(test_csv, batch_size=128, num_workers=2,
                     R_ref=-30.0, eta=2.0, missing_value=-200.0):
    ds = EvaluationDataset(test_csv, R_ref=R_ref, eta=eta, missing_value=missing_value)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)