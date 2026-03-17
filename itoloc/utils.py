# utils.py
import os, random, torch, numpy as np
from torch import nn
from torch.autograd import Function

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def ale_meters(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> float:
    return torch.linalg.vector_norm(pred_xy - gt_xy, dim=1).mean().item()

def save_checkpoint(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    return model

# GRL (训练 DANN 时会用到；iToLoc 阶段默认不再用)
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.lambd * grad_output, None

class GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0): super().__init__(); self.lambd = lambd
    def forward(self, x): return _GradReverse.apply(x, self.lambd)

# RSSI → “指纹图像”（pairwise 距离比）
def rssi_to_fingerprint_image(
    rssi_1d: np.ndarray,
    R_ref: float = -30.0,
    eta: float = 2.0,
    missing_value: float = -200.0,
    clip_max: float = 5.0
):
    f = rssi_1d.astype(np.float32).copy()
    f = np.where(f <= missing_value + 1e-6, -110.0, f)
    d = 10 ** ((R_ref - f) / (10.0 * eta))
    d = np.clip(d, 1e-6, 1e6)
    ratio = (d[:, None] / d[None, :]) - 1.0
    ratio = np.clip(ratio, -1.0, clip_max)
    np.fill_diagonal(ratio, 0.0)
    return ratio.astype(np.float32)
