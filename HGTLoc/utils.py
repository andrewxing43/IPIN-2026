# === utils.py ===
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

def mse_loss():
    return torch.nn.MSELoss()

def euclidean_rmse(preds, targets):
    """
    计算欧几里得 RMSE：sqrt(mean(||pred - target||^2))
    preds, targets: [N, 2] tensor
    """
    return torch.sqrt(torch.mean(torch.sum((preds - targets) ** 2, dim=1))).item()

def ale(preds, targets):
    """
    计算 ALE（Average Localization Error）：mean(||pred - target||)
    """
    return torch.mean(torch.norm(preds - targets, dim=1)).item()

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def plot_loss_curve(train_loss_list, val_ale_list, save_path=None):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_ale_list, label='Val ALE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('ALE')
    plt.title('Validation ALE')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

import torch
import torch.nn.functional as F

def ale_loss(pred, target, eps: float = 1e-8):
    # pred,target: [N,2]
    # 加 eps 保证数值稳定
    return torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, dim=1) + eps))

def combined_loss(pred, target, lambda_ale: float = 0.5):
    # RMSE 的无开方形式就是 MSELoss，更平滑更稳
    mse = F.mse_loss(pred, target)
    ale = ale_loss(pred, target)
    return lambda_ale * ale + (1.0 - lambda_ale) * mse
