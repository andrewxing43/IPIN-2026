import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

METHOD_NAME = "KNN"  

def load_data(train_csv, test_csv):
    # 自动识别维度，不再写死 100
    train_df = pd.read_csv(train_csv, header=None).values  
    test_df  = pd.read_csv(test_csv,  header=None).values  
    
    # 假设你的 CSV 结构是：[AP1, AP2, ..., X, Y] (没有 floor)
    # 如果你的 CSV 结构是：[AP1, AP2, ..., X, Y, Floor]，请改成 :-3 和 -3:-1
    X_train = train_df[:, :-2]
    y_train = train_df[:, -2:]
    
    X_test  = test_df[:, :-2]
    y_test  = test_df[:, -2:]
    
    # 【可选优化】：将极端的 -200 惩罚拉回到一个合理范围，防止欧式距离爆炸
    # X_train[X_train == -200] = -100
    # X_test[X_test == -200] = -100
    
    return X_train, y_train, X_test, y_test

def compute_errors(y_true, y_pred, eps=1e-12):
    diff = y_true - y_pred
    dists = np.sqrt(np.sum(diff * diff, axis=1) + eps)
    return dists.astype(np.float32)

if __name__ == "__main__":
    # 注意：确保这里传入的是 AP 对齐后的文件，如果是不同场地或 AP 不同的原始文件，需要先做对齐
    train_csv = "train.csv"
    test_csv  = "test.csv"
    k = 11

    print(f"Loading data from {train_csv} and {test_csv}...")
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)
    
    print(f"[Model] Input feature dim = {X_train.shape[1]}")

    print(f"Fitting KNN regressor with k={k} and minkowski metric...")
    model = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance')
    model.fit(X_train, y_train)

    print("Predicting...")
    y_pred = model.predict(X_test)

    errors = compute_errors(y_test, y_pred)  # (N,)
    ale = float(errors.mean())
    p50 = float(np.percentile(errors, 50))
    p90 = float(np.percentile(errors, 90))   # ✅ 修复了 P90 的百分比

    print(f"Test ALE: {ale:.4f} m")
    print(f"Test P50: {p50:.4f} m")
    print(f"Test P90: {p90:.4f} m")

    os.makedirs("results", exist_ok=True)
    np.save(f"results/errors_{METHOD_NAME}.npy", errors)
    print(f"[Saved] results/errors_{METHOD_NAME}.npy")

    # ✅ 修复了可能由于 CDF 文件夹不存在导致的崩溃
    summary_path = "CDF/summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True) 
    
    header = "method,ALE_m,P50_m,P90_m\n"
    row = f"{METHOD_NAME},{ale:.6f},{p50:.6f},{p90:.6f}\n"
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write(header)
    with open(summary_path, "a") as f:
        f.write(row)
    print(f"[Append] {summary_path} ({METHOD_NAME}: ALE={ale:.3f}, P50={p50:.3f}, P90={p90:.3f})")