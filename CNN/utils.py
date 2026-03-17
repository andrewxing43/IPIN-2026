import torch
import torch.nn as nn
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# ----------------------------
# Loss Functions
# ----------------------------
def get_loss_fn(name="smoothl1"):
    if name == "mse":
        return nn.MSELoss()
    elif name == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")

# ----------------------------
# RMSE Metric
# ----------------------------
def compute_rmse(preds, targets):
    """
    preds: (N, 2)
    targets: (N, 2)
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

# ----------------------------
# ALE Metric (Average L2 Error)
# ----------------------------
def compute_ale(preds, targets):
    """
    preds: (N, 2)
    targets: (N, 2)
    """
    return torch.norm(preds - targets, dim=1).mean().item()

# ----------------------------
# Save / Load Model
# ----------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

# ----------------------------
# Random Seed Control
# ----------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Loss Curve Plotting
# ----------------------------
def plot_losses(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
