import os
import torch
import pandas as pd
import numpy as np
from CNN import RSSICNN
from brute_dataloader import RSSIFingerprintDataset
from utils import compute_rmse, compute_ale

METHOD_NAME = "CNNLoc"  # 改名即可复用这套保存逻辑

def test(
    csv_path,
    model_path,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"[Device] Using {device}")

    # 加载数据集
    dataset = RSSIFingerprintDataset(csv_path, selected_ap_index_path=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    # 加载模型
    model = RSSICNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 推理
    preds, targets = [], []
    with torch.no_grad():
        for rssi, coord in loader:
            rssi = rssi.to(device)
            pred = model(rssi).cpu()
            preds.append(pred)
            targets.append(coord)

    preds = torch.cat(preds, dim=0)      # (N,2) 归一化坐标
    targets = torch.cat(targets, dim=0)  # (N,2) 归一化坐标

    # 反归一化到米（按你现有实现）
    preds_real = preds * 30
    targets_real = targets * 30

    # 指标：RMSE / ALE
    rmse = compute_rmse(preds_real, targets_real)
    ale  = compute_ale(preds_real, targets_real)

    # 逐样本误差（米）：用于 P50/P90 和 CDF
    errors = torch.linalg.norm(preds_real - targets_real, dim=1).cpu().numpy().astype(np.float32)
    p50 = float(np.percentile(errors, 50))
    p90 = float(np.percentile(errors, 90))

    print("Test Results:")
    print(f"RMSE = {rmse:.4f} m")
    print(f"ALE  = {ale:.4f} m")
    print(f"P50  = {p50:.4f} m")
    print(f"P90  = {p90:.4f} m")

    # 保存：逐样本误差 + 汇总行
    os.makedirs("results", exist_ok=True)
    np.save(f"results/errors_{METHOD_NAME}.npy", errors)
    print(f"[Saved] results/errors_{METHOD_NAME}.npy")

    summary_path = "results/summary.csv"
    header = "method,ALE_m,P50_m,P90_m\n"
    row = f"{METHOD_NAME},{float(ale):.6f},{p50:.6f},{p90:.6f}\n"
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header)
    with open(summary_path, "a") as f:
        f.write(row)
    print(f"[Append] {summary_path} ({METHOD_NAME}: ALE={ale:.3f}, P50={p50:.3f}, P90={p90:.3f})")

    return preds_real.numpy(), targets_real.numpy(), errors

if __name__ == "__main__":
    test_csv = "test.csv"   # 替换为你的测试集路径
    model_file = "./checkpoints/CNN/best_model.pth"  # 替换为模型路径
    pred, target, errors = test(test_csv, model_file)
