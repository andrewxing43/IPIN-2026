# test.py
import os, numpy as np, torch
from dataloader import make_eval_loader
from model import DANNModel
from utils import load_checkpoint

METHOD_NAME = "iToLoc"  # 改成 "DANN" 可评估 DANN
CKPT_PATH   = "checkpoints/itoloc/best_model.pth"  # 评估 DANN 时改为 checkpoints/dann_regressor.pth

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_eval_loader("test.csv", batch_size=128, num_workers=2,
                              R_ref=-30.0, eta=2.0, missing_value=-200.0)

    model = DANNModel(in_ch=1, feat_dim=128, grl_lambda=0.0).to(device)
    load_checkpoint(model, CKPT_PATH, device); model.eval()

    all_errors = []
    for img, xy, _ in loader:
        img, xy = img.to(device), xy.to(device)
        pred = model.predict_xy(img)
        all_errors.append((pred - xy).norm(dim=1).cpu().numpy())
    all_errors = np.concatenate(all_errors)

    os.makedirs("results", exist_ok=True)
    np.save(f"results/errors_{METHOD_NAME}.npy", all_errors)

    ale = float(all_errors.mean())
    p50 = float(np.percentile(all_errors, 50))
    p90 = float(np.percentile(all_errors, 90))
    print(f"[Test:{METHOD_NAME}] ALE={ale:.3f} m, P50={p50:.3f} m, P90={p90:.3f} m")

    row = f"{METHOD_NAME},{ale:.6f},{p50:.6f},{p90:.6f}\n"
    csv_path = "results/summary.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f: f.write("method,ALE_m,P50_m,P90_m\n")
    with open(csv_path, "a") as f: f.write(row)
    print(f"[Saved] results/errors_{METHOD_NAME}.npy")
    print(f"[Append] {csv_path}")

if __name__ == "__main__":
    main()
