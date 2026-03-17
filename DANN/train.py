# train.py
import math  # 【新增】：用于计算动态 lambda
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR

from dataloader import make_loaders
from model import DANNModel
from utils import set_seed, save_checkpoint

def main():
    set_seed(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 固定文件名
    train_csv = "train.csv"
    test_csv = "test.csv"

    # Data
    src_loader, tgt_loader = make_loaders(
        train_csv, test_csv, batch_size=64, num_workers=2,
        R_ref=-30.0, eta=2.0, missing_value=-200.0
    )

    # Model (初始 grl_lambda 置为0即可，因为 forward 时会动态覆盖)
    model = DANNModel(in_ch=1, feat_dim=128, grl_lambda=0.0).to(device)
    reg_loss = nn.MSELoss()
    dom_loss = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optim, step_size=20, gamma=0.1)

    best_src_reg = float("inf")
    total_epochs = 100

    for epoch in range(1, total_epochs + 1):
        model.train()
        
        # 【关键修改】：动态计算 DANN 的 lambda
        p = (epoch - 1) / total_epochs 
        grl_lambda = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        iters = min(len(src_loader), len(tgt_loader))
        running_reg, running_dom = 0.0, 0.0

        for _ in range(iters):
            try:
                src_img, src_xy, src_dom = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_img, src_xy, src_dom = next(src_iter)

            try:
                tgt_img, tgt_dom = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_img, tgt_dom = next(tgt_iter)

            src_img, src_xy, src_dom = src_img.to(device), src_xy.to(device), src_dom.to(device)
            tgt_img, tgt_dom = tgt_img.to(device), tgt_dom.to(device)

            optim.zero_grad()

            # 源域监督回归：传入动态计算的 grl_lambda
            y_pred, dom_logits_src = model(src_img, grl_lambda=grl_lambda)
            loss_reg = reg_loss(y_pred, src_xy)

            # 目标域域分类：传入动态计算的 grl_lambda
            _, dom_logits_tgt = model(tgt_img, grl_lambda=grl_lambda)
            dom_logits = torch.cat([dom_logits_src, dom_logits_tgt], dim=0)
            dom_labels = torch.cat([src_dom, tgt_dom], dim=0).to(device)
            loss_dom = dom_loss(dom_logits, dom_labels)

            loss = loss_reg + loss_dom
            loss.backward()
            optim.step()

            running_reg += loss_reg.item()
            running_dom += loss_dom.item()

        scheduler.step()
        avg_reg = running_reg / iters
        avg_dom = running_dom / iters

        # 打印时加上 lambda 方便监控
        print(f"[Epoch {epoch:03d}] reg={avg_reg:.4f}  dom={avg_dom:.4f}  lambda={grl_lambda:.4f}")

        if avg_reg < best_src_reg:
            best_src_reg = avg_reg
            save_checkpoint(model, "checkpoints/DANN/best_model.pth")
            print(f"  -> Saved best model (reg={best_src_reg:.4f})")

if __name__ == "__main__":
    main()