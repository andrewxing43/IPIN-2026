import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class RSSIFingerprintDataset(Dataset):
    def __init__(self, csv_path, selected_ap_index_path=None):
        # 读取完整 CSV，无表头
        all_data = pd.read_csv(csv_path, header=None).values  

        # 【核心修改】：利用负索引，动态剥离特征和标签
        # :-2 表示取从第 0 列到倒数第 3 列的所有数据作为 RSSI 特征
        full_rssi = all_data[:, :-2].astype(np.float32)   
        # -2: 表示取最后两列作为 X, Y 坐标
        coords = all_data[:, -2:].astype(np.float32)      

        # ===== AP 筛选（如果提供了 AP 索引）=====
        if selected_ap_index_path is not None:
            selected_idx = np.load(selected_ap_index_path)  
            full_rssi = full_rssi[:, selected_idx]          
        # ========================================

        # 归一化 (保留你原来经过验证的合理缩放比例)
        self.X = (full_rssi + 100.0) / 69.0    
        self.coord = coords / 30.0             

        # 动态获取当前 AP 特征的实际维度
        self.input_dim = self.X.shape[1]       

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),     # shape: (input_dim,)
            torch.tensor(self.coord[idx], dtype=torch.float32)  # shape: (2,)
        )

def load_dataset(csv_path, selected_ap_index_path=None, val_split=0.2, batch_size=64, seed=42):
    dataset = RSSIFingerprintDataset(csv_path, selected_ap_index_path)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    def _worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )

    # 额外返回 dataset.input_dim，方便在主函数里动态初始化模型
    return train_loader, val_loader, dataset.input_dim


# ✅ 测试主函数
if __name__ == "__main__":
    csv_file = "train.csv"
    ap_index_path = None  # 测试原始文件

    train_loader, val_loader, feature_dim = load_dataset(csv_file, ap_index_path)

    print(f"✅ 成功加载数据集！检测到输入的 RSSI 特征维度为: {feature_dim}")
    
    for rssi, coord in train_loader:
        print(f"RSSI Batch Shape: {rssi.shape}")   # 应该是 (B, feature_dim)
        print(f"Coord Batch Shape: {coord.shape}") # 应该是 (B, 2)
        break