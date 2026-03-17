import numpy as np
import pandas as pd

def compute_per_ap_stability_variance(rssi_data, group_size=5):
    num_aps = rssi_data.shape[1]
    num_samples = rssi_data.shape[0]
    num_groups = num_samples // group_size

    V = np.zeros(num_aps)

    for ap_idx in range(num_aps):
        group_vars = []
        for g in range(num_groups):
            group = rssi_data[g * group_size:(g + 1) * group_size, ap_idx]
            valid = group[group > -200]
            if len(valid) > 1:
                group_vars.append(np.var(valid))
        if len(group_vars) > 0:
            V[ap_idx] = np.mean(group_vars)
        else:
            V[ap_idx] = 1e6  # 缺失严重
    return V

def select_top_k_ap_indices(train_raw_csv, k=100, save_index_path=None):
    # 读取无表头数据 (header=None 是必须的)
    data = pd.read_csv(train_raw_csv, header=None).values
    
    # 【核心修改 1】: 取除了最后两列之外的所有列作为 AP 特征
    rssi_data = data[:, :-2]
    N, num_aps = rssi_data.shape

    C = (rssi_data > -200).sum(axis=0) / N
    V = compute_per_ap_stability_variance(rssi_data, group_size=5)
    max_V = np.max(V) + 1e-8
    scores = C * (1 - V / max_V)

    top_k_indices = np.argsort(scores)[-k:][::-1]
    if save_index_path:
        np.save(save_index_path, top_k_indices)
        print(f"Saved top-{k} AP indices to {save_index_path}")
    return top_k_indices

def apply_ap_selection_and_save(raw_csv_path, ap_indices, save_path):
    data = pd.read_csv(raw_csv_path, header=None).values
    
    # 提取得分最高的 K 个 AP 的数据
    selected_rssi = data[:, ap_indices]            # shape: [N, k]
    
    # 【核心修改 2】: 提取最后两列作为标签 (X, Y)
    labels = data[:, -2:]                          
    
    # 拼接到一起
    final_data = np.hstack([selected_rssi, labels])
    
    # 保存结果 (由于是纯数字，所以不要 header 和 index)
    pd.DataFrame(final_data).to_csv(save_path, index=False, header=False)
    print(f"Saved reduced data to {save_path}")

if __name__ == "__main__":
    # 请将这里的 csv 名字换成你实际的无表头数据文件名
    input_csv = "train.csv"
    
    # 1. 从数据集中计算并选择 top 100 个 AP，保存它们的列号
    top_k_ap = select_top_k_ap_indices(
        train_raw_csv=input_csv,
        k=175,
        save_index_path=False
    )

    # 2. 生成全新的、只有 100 个特征 + 2 个坐标列的文件
    apply_ap_selection_and_save(input_csv, top_k_ap, "train_final.csv")
    input_csv = "test.csv"
    apply_ap_selection_and_save(input_csv, top_k_ap, "test_final.csv")
     