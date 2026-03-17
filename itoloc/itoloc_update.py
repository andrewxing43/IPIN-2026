# itoloc.py
import torch
import numpy as np
import os
import sys
from torch import nn
from torch.optim import AdamW
from itertools import cycle
from torch.utils.data import random_split, DataLoader, RandomSampler

# 自动处理路径，确保能导入平级目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 修改点：引入了 LabeledRSSIDataset 以便手动进行划分
from dataloader import LabeledRSSIDataset, make_target_loader
from model import DANNModel
from utils import set_seed, save_checkpoint

def pairwise_max_dist(y1, y2, y3):
    d12 = (y1 - y2).norm(dim=1)
    d23 = (y2 - y3).norm(dim=1)
    d13 = (y1 - y3).norm(dim=1)
    return torch.stack([d12, d23, d13], dim=1).max(dim=1).values  # (B,)

@torch.no_grad()
def evaluate(model, loader, device):
    """评估函数：计算平均定位误差（米）"""
    model.eval()
    errors = []
    for img, target, _ in loader:
        img, target = img.to(device), target.to(device)
        # 使用三个头的平均预测值作为最终结果
        pred, _ = model.regressor(model.extractor(img))
        dist = torch.norm(pred - target, dim=1)
        errors.extend(dist.cpu().numpy())
    return np.mean(errors) if len(errors) > 0 else float('inf')

def main():
    set_seed(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 确保保存目录存在
    save_dir = "./checkpoints/itoloc"
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 1. 数据加载与划分 (修改点：修复数据泄漏)
    # ==========================================
    # 实例化完整的源域数据集
    full_src_dataset = LabeledRSSIDataset(
        "train.csv", 
        R_ref=-30.0, eta=2.0, missing_value=-200.0
    )
    
    # 按照 80% 训练，20% 验证的比例进行切分
    train_size = int(0.8 * len(full_src_dataset))
    val_size = len(full_src_dataset) - train_size
    
    # 使用 generator 固定种子，确保每次运行切分出的验证集是一样的
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = random_split(full_src_dataset, [train_size, val_size], generator=generator)

    # 构造源域训练 DataLoader (带 Anchor 采样逻辑)
    src_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=32*100),
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )

    # 构造干净的源域验证 DataLoader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # 目标域无标签数据加载保持不变
    tgt_loader = make_target_loader(
        "test.csv", batch_size=32, num_workers=2,
        R_ref=-30.0, eta=2.0, missing_value=-200.0
    )

    # ==========================================
    # 2. 模型初始化与预训练权重加载
    # ==========================================
    model = DANNModel(in_ch=1, feat_dim=128, grl_lambda=0.0).to(device)
    try:
        sd = torch.load("./checkpoints/DANN/best_model.pth", map_location=device)
        model.load_state_dict(sd)
        print("-> Successfully loaded DANN pre-trained weights.")
    except Exception as e:
        print(f"-> Warning: Load checkpoint failed: {e}. Training from scratch.")

    # ==========================================
    # 3. 优化器配置
    # ==========================================
    # 初始状态：冻结特征提取器
    for p in model.extractor.parameters():
        p.requires_grad = False
    
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    mse_none = nn.MSELoss(reduction='none')

    # ==========================================
    # 4. 超参数
    # ==========================================
    epochs_stage1 = 20   # 仅回归头微调
    epochs_stage2 = 20   # 轻微解冻 extractor.head
    total_epochs = epochs_stage1 + epochs_stage2
    eps_start, eps_end = 0.5, 2.5
    diversity_w = 0.05
    anchor_w = 0.5       # 增加锚点权重以保持稳定性
    smear_sigma = 0.05
    
    best_error = float('inf')

    def masked_mse(pred, tgt, m):
        l = mse_none(pred, tgt).mean(dim=1)
        if m.sum() < 1: return pred.new_tensor(0.0)
        return (l * m).sum() / (m.sum() + 1e-9)

    # ==========================================
    # 5. 训练循环
    # ==========================================
    src_iter = cycle(src_loader)
    
    for epoch in range(total_epochs):
        # 阶段切换：解冻 Extractor 的高层网络
        if epoch == epochs_stage1:
            print("-> Stage 2: Unfreezing extractor head layers...")
            for p in model.extractor.head.parameters():
                p.requires_grad = True
        
        model.train()
        eps = eps_start + (eps_end - eps_start) * (epoch / float(total_epochs - 1))
        total_used, total_seen, loss_sum = 0, 0, 0.0

        for tgt_img, _ in tgt_loader:
            tgt_img = tgt_img.to(device)

            # --- 目标域：三头一致性筛选 ---
            with torch.no_grad():
                avg, (y1, y2, y3) = model.predict_heads(tgt_img)
            maxd = pairwise_max_dist(y1, y2, y3)
            mask = (maxd < eps).float()
            total_used += int(mask.sum().item())
            total_seen += mask.numel()

            pseudo = avg + torch.randn_like(avg) * smear_sigma

            # --- 前向传播 ---
            feats = model.extractor(tgt_img)
            p1 = model.regressor.m1(feats)
            p2 = model.regressor.m2(feats)
            p3 = model.regressor.m3(feats)

            loss_pseudo = (masked_mse(p1, pseudo, mask) +
                           masked_mse(p2, pseudo, mask) +
                           masked_mse(p3, pseudo, mask)) / 3.0

            div = ((p1 - p2).pow(2).mean() + (p2 - p3).pow(2).mean() + (p1 - p3).pow(2).mean()) / 3.0
            loss_div = - diversity_w * div

            # --- 源域：锚点监督防止漂移 ---
            src_img, src_xy, _ = next(src_iter)
            src_img, src_xy = src_img.to(device), src_xy.to(device)
            fs = model.extractor(src_img)
            s1, s2, s3 = model.regressor.m1(fs), model.regressor.m2(fs), model.regressor.m3(fs)
            loss_anchor = ((s1 - src_xy).pow(2).mean() + 
                           (s2 - src_xy).pow(2).mean() + 
                           (s3 - src_xy).pow(2).mean()) / 3.0

            loss = loss_pseudo + loss_div + (anchor_w * loss_anchor)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.item()

        # --- 验证与保存最佳模型 ---
        # 修改点：使用干净的 val_loader 进行评估
        current_error = evaluate(model, val_loader, device) 
        use_rate = 100.0 * total_used / max(1, total_seen)
        
        print(f"Epoch [{epoch:02d}/{total_epochs}] | Val Error: {current_error:.3f}m | Used: {use_rate:.1f}% | Loss: {loss_sum/len(tgt_loader):.4f}")

        if current_error < best_error:
            best_error = current_error
            save_checkpoint(model, os.path.join(save_dir, "best_model.pth"))
            print(f" >> Best model saved (Val Error: {best_error:.4f})")

    print(f"Training finished. Best Val Error: {best_error:.4f}")

if __name__ == "__main__":
    main()