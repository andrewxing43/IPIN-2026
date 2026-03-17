# utils.py
import os, random, math
import torch
import numpy as np
from torch import nn
from torch.autograd import Function

# ---------- Repro ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Metrics ----------
@torch.no_grad()
def ale_meters(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> float:
    # pred_xy, gt_xy: (N,2), 单位同数据（米）
    return torch.linalg.vector_norm(pred_xy - gt_xy, dim=1).mean().item()

# ---------- Checkpoint ----------
def save_checkpoint(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    return model

# ---------- Gradient Reversal Layer ----------
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return _GradReverse.apply(x, self.lambd)

# ---------- Fingerprint Image Transform ----------
def rssi_to_fingerprint_image(
    rssi_1d: np.ndarray,
    R_ref: float = -30.0,     # 1m 参考 RSSI
    eta: float = 2.0,         # 路损指数
    missing_value: float = -200.0,
    clip_max: float = 5.0     # 对 (d_i/d_j - 1) 上限裁剪，稳定训练
):
    """
    输入: rssi_1d shape (N,) -> 输出: fp_img shape (N,N)
    x_{i,j} = d_i/d_j - 1, 其中 d = 10^{(R - f)/(10*eta)}
    """
    f = rssi_1d.astype(np.float32).copy()
    # 缺失值处理：将极小 RSSI 映射为很大的距离
    f = np.where(f <= missing_value + 1e-6, -110.0, f)  # 把缺失近似为 -110 dBm
    d = 10 ** ((R_ref - f) / (10.0 * eta))              # 相对传播“距离”
    d = np.clip(d, 1e-6, 1e6)
    ratio = (d[:, None] / d[None, :]) - 1.0            # (N,N)
    # 数值稳定：裁剪过大值，主对角设为0
    ratio = np.clip(ratio, -1.0, clip_max)
    np.fill_diagonal(ratio, 0.0)
    return ratio.astype(np.float32)