# dataloader.py
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils import rssi_to_fingerprint_image

class LabeledRSSIDataset(Dataset):
    # CSV: 0..99 -> RSSI(100), 100->x, 101->y, 102->floor(忽略)
    def __init__(self, csv_path, rssi_cols=slice(0,175), x_col=175, y_col=176,
                 R_ref=-30.0, eta=2.0, missing_value=-200.0):
        df = pd.read_csv(csv_path)
        self.rssi = df.iloc[:, rssi_cols].to_numpy(np.float32)
        self.xy   = df.iloc[:, [x_col, y_col]].to_numpy(np.float32)
        self.R_ref, self.eta, self.missing_value = R_ref, eta, missing_value

    def __len__(self): return self.rssi.shape[0]
    def __getitem__(self, idx):
        img = rssi_to_fingerprint_image(self.rssi[idx],
                                        R_ref=self.R_ref, eta=self.eta, missing_value=self.missing_value)
        img = torch.from_numpy(img).unsqueeze(0)  # (1,100,100)
        xy  = torch.from_numpy(self.xy[idx])      # (2,)
        dom = torch.tensor(0, dtype=torch.long)
        return img, xy, dom

class UnlabeledRSSIDataset(Dataset):
    # 目标域无标签（iToLoc 阶段用于伪标签池）
    def __init__(self, csv_path, rssi_cols=slice(0,175),
                 R_ref=-30.0, eta=2.0, missing_value=-200.0):
        df = pd.read_csv(csv_path)
        self.rssi = df.iloc[:, rssi_cols].to_numpy(np.float32)
        self.R_ref, self.eta, self.missing_value = R_ref, eta, missing_value

    def __len__(self): return self.rssi.shape[0]
    def __getitem__(self, idx):
        img = rssi_to_fingerprint_image(self.rssi[idx],
                                        R_ref=self.R_ref, eta=self.eta, missing_value=self.missing_value)
        img = torch.from_numpy(img).unsqueeze(0)  # (1,100,100)
        dom = torch.tensor(1, dtype=torch.long)
        return img, dom

def make_eval_loader(test_csv, batch_size=32, num_workers=2,
                     R_ref=-30.0, eta=2.0, missing_value=-200.0):
    class EvaluationDataset(LabeledRSSIDataset):
        pass
    ds = EvaluationDataset(test_csv, R_ref=R_ref, eta=eta, missing_value=missing_value)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def make_anchor_sampler(train_csv, batch_size=32, num_workers=2,
                        R_ref=-30.0, eta=2.0, missing_value=-200.0):
    ds = LabeledRSSIDataset(train_csv, R_ref=R_ref, eta=eta, missing_value=missing_value)
    loader = DataLoader(ds, batch_size=batch_size,
                        sampler=RandomSampler(ds, replacement=True, num_samples=batch_size*100),
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader

def make_target_loader(test_csv, batch_size=32, num_workers=2,
                       R_ref=-30.0, eta=2.0, missing_value=-200.0):
    ds = UnlabeledRSSIDataset(test_csv, R_ref=R_ref, eta=eta, missing_value=missing_value)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
