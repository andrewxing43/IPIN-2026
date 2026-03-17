# test.py
import os
import numpy as np
import torch
from dataloader import make_eval_loader
from model import DANNModel
from utils import load_checkpoint

METHOD_NAME = "DANN"
CKPT_PATH   = "checkpoints/DANN/best_model.pth"  # 如需评估其它权重，改这里

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = "test.csv"

    # 由于在 dataloader.py 已经把默认维度改成了 175，这里直接调用即可自动适配
    loader = make_eval_loader(
        test_csv, batch_size=32, num_workers=2,
        R_ref=-30.0, eta=2.0, missing_value=-200.0
    )

    model = DANNModel(in_ch=1, feat_dim=128, grl_lambda=0.0).to(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()

    # 收集逐样本误差（米）用于 ALE/P50/P90 和 CDF
    all_errors = []
    n = 0
    for img, xy in loader:  # 你的 eval loader 返回的是 (img, xy)
        img, xy = img.to(device), xy.to(device)
        pred = model.predict_xy(img)
        err = (pred - xy).norm(dim=1).detach().cpu().numpy()  # (B,)
        all_errors.append(err)
        n += xy.size(0)

    errors = np.concatenate(all_errors).astype(np.float32)  # shape (N,)
    ale = float(errors.mean())
    p50 = float(np.percentile(errors, 50))
    # 【关键修改】：修正 90% 分位数的计算逻辑
    p90 = float(np.percentile(errors, 90))

    print(f"[Test:{METHOD_NAME}] N={n}")
    print(f"ALE = {ale:.3f} m")
    print(f"P50 = {p50:.3f} m")
    print(f"P90 = {p90:.3f} m")

    # 保存逐样本误差与汇总 CSV
    os.makedirs("results", exist_ok=True)
    np.save(f"results/errors_{METHOD_NAME}.npy", errors)
    print(f"[Saved] results/errors_{METHOD_NAME}.npy")

    summary_path = "results/summary.csv"
    header = "method,ALE_m,P50_m,P90_m\n"
    row = f"{METHOD_NAME},{ale:.6f},{p50:.6f},{p90:.6f}\n"
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header)
    with open(summary_path, "a") as f:
        f.write(row)
    print(f"[Append] {summary_path} ({METHOD_NAME}: ALE={ale:.3f}, P50={p50:.3f}, P90={p90:.3f})")

if __name__ == "__main__":
    main()