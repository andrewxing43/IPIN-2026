import os
import numpy as np
import torch
from model import RSSITransformerModel
from dataloader1 import load_data_and_build_dataloaders
from utils import ale, load_checkpoint, set_seed

# --- 仿照 DANN 代码设置常量 ---
METHOD_NAME = "HGTLoc"
RESULTS_DIR = "results"

def evaluate_test(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu())
            all_targets.append(data['label'].cpu())
            
    # 反归一化，恢复到实际的物理米数
    pred = torch.cat(all_preds, dim=0) * 30
    target = torch.cat(all_targets, dim=0) * 30
    
    test_ale = ale(pred, target)
    return test_ale, pred, target

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading test data...")
    # 注意：确保此处加载的是你想要评估的数据集（Dataset 1 或 2）
    _, _, test_loader = load_data_and_build_dataloaders('train.csv', 'test.csv', batch_size=32)

    model = RSSITransformerModel().to(device)

    model_path = './checkpoints/new/best_model.pth'
    print(f"Loading weights from {model_path} ...")
    
    try:
        model = load_checkpoint(model, model_path, device=device)
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Cannot find {model_path}. Please make sure the model is trained and saved.")
        return

    print("Running inference on test set...")
    test_ale, preds, targets = evaluate_test(model, test_loader, device)
    
    # ---------------- 计算指标 ----------------
    # 计算每个测试样本的欧式距离误差 (meters)
    errors_tensor = torch.norm(preds.float() - targets.float(), dim=1)
    
    # 转换为 numpy 格式以便保存和计算
    errors_np = errors_tensor.numpy().astype(np.float32)
    
    ale_val = float(errors_np.mean())
    p50 = float(np.percentile(errors_np, 50))
    p75 = float(np.percentile(errors_np, 75))
    p90 = float(np.percentile(errors_np, 90))

    print("-" * 40)
    print(f"[{METHOD_NAME} Test Results]")
    print(f"ALE : {ale_val:.4f} m")
    print(f"P50 : {p50:.4f} m")
    print(f"P75 : {p75:.4f} m")
    print(f"P90 : {p90:.4f} m")
    print("-" * 40)

    # ---------------- 仿照 DANN 的保存方式 ----------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 保存逐样本误差 npy 文件 (用于画 CDF 图)
    npy_path = os.path.join(RESULTS_DIR, f"errors_{METHOD_NAME}.npy")
    np.save(npy_path, errors_np)
    print(f"[Saved] {npy_path}")

    # 2. 追加到 summary.csv 汇总表
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    header = "method,ALE_m,P50_m,P75_m,P90_m\n"
    row = f"{METHOD_NAME},{ale_val:.6f},{p50:.6f},{p75:.6f},{p90:.6f}\n"
    
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header)
            
    with open(summary_path, "a") as f:
        f.write(row)
    print(f"[Append] {summary_path} Done.")

if __name__ == '__main__':
    main()