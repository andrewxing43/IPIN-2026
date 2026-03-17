# draw.py — plot CDF curves for all methods in one figure (run inside `results/`)
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy import series

# ======= Config (sizes & style) =======
SAVE_PATH    = "cdf_comparison.png"   # 设 "" 不保存
SHOW_FIG     = True
XLABEL       = "Localization error (m)"
YLABEL       = "Cumulative Distiribution (CDF)"
XLIM_MAX     = None                   # 例如 10.0；None 自动
LINEWIDTH    = 3.0                    # 曲线粗细

# 字体与字号（Times New Roman）
mpl.rcParams["font.family"] = "Times New Roman"   # 需要系统已安装该字体
TITLE_SIZE   = 0   # 不需要标题
LABEL_SIZE   = 18
TICK_SIZE    = 16
LEGEND_SIZE  = 16

# 绘制顺序；其余方法按字母序追加
METHODS_ORDER = ["HGTLoc", "iToLoc", "DANN", "CNNLoc", "KNN"]
FIG_SIZE      = (7.5, 6.0)            # inches
GRID_ALPHA    = 0.5
GRID_LW       = 0.8
LEGEND_LOC    = "lower right"
# ======================================

def find_error_files():
    files = glob.glob("results/errors_*.npy")
    if not files:
        raise FileNotFoundError("No errors_*.npy found in current folder.")
    label_map = {}
    for f in files:
        base = os.path.basename(f)
        label = base.replace("errors_", "").replace(".npy", "")
        label_map[label] = f
    ordered = []
    for m in METHODS_ORDER:
        if m in label_map:
            ordered.append((m, label_map.pop(m)))
    for m in sorted(label_map.keys()):
        ordered.append((m, label_map[m]))
    return ordered

def load_errors(path):
    e = np.load(path).astype(np.float64).reshape(-1)
    return e[np.isfinite(e)]

def plot_cdf_curves(series):
    plt.figure(figsize=FIG_SIZE)
    for label, errs in series:
        if errs.size == 0:
            print(f"[WARN] {label}: empty error array, skip.")
            continue
        
        # 1. 基础排序和计算
        xs = np.sort(errs)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        
        # 2. 核心修改：手动添加一个远端点，让曲线水平延伸至 XLIM_MAX
        # 如果你没设置 XLIM_MAX，可以取一个比当前最大值大一点的值
        current_max_x = xs[-1]
        extended_x = XLIM_MAX if (XLIM_MAX is not None) else (current_max_x * 1.1)
        
        # 拼接数组：在末尾追加一个 (extended_x, 1.0) 的点
        plot_xs = np.append(xs, 16)
        plot_ys = np.append(ys, 1.0)
        
        # 3. 绘图
        plt.plot(plot_xs, plot_ys, label=label, linewidth=LINEWIDTH)

    
    plt.xlabel(XLABEL, fontsize=LABEL_SIZE)
    plt.ylabel(YLABEL, fontsize=LABEL_SIZE)
    if XLIM_MAX is not None:
        plt.xlim(0, XLIM_MAX)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(True, linestyle="--", alpha=GRID_ALPHA, linewidth=GRID_LW)
    plt.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=True)
    plt.tight_layout()

def main():
    pairs = find_error_files()
    data = [(label, load_errors(path)) for label, path in pairs]
    plot_cdf_curves(data)
    if SAVE_PATH:
        plt.savefig(SAVE_PATH, dpi=400, bbox_inches="tight")
        print(f"[Saved] {SAVE_PATH}")
    if SHOW_FIG:
        plt.show()

if __name__ == "__main__":
    main()
