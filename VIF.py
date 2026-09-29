import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 设置工作路径与文件名
data_dir = r"C:/Users/吴佳琪/Desktop/Serbia_ML/第三次尝试（更新数据）/VIF检验"
native_file = os.path.join(data_dir, "native_data.csv")
learner_file = os.path.join(data_dir, "learner_data.csv")

# 2. 24 个特征参数
feature_cols = [
    'max_pos', 'min_pos', 'f0_max', 'f0_min', 'slope',
    'curve', 'BLslope1', 'BLslope2', 'Duration', 'Location'
]

def load_csv_safely(file_path):
    """尝试用多种编码读取 CSV 文件，避免 UnicodeDecodeError"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"❌ 无法读取文件 {file_path}，请检查文件编码！")

def calculate_vif_features(file_path, group_name):
    if not os.path.exists(file_path):
        print(f"❌ 未找到文件: {file_path}")
        return None

    # 🌟 自动识别编码读取数据
    df = load_csv_safely(file_path)

    # 检查表格中是否存在所有列
    actual_cols = [col for col in feature_cols if col in df.columns]
    if len(actual_cols) < len(feature_cols):
        missing = set(feature_cols) - set(actual_cols)
        print(f"⚠️ 警告 [{group_name}]：当前 CSV 文件缺失列: {missing}")

    X = df[actual_cols].copy()

    for col in actual_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X_clean = X.dropna()

    # 🌟 核心步骤：加入常数截距项（Constant）
    X_with_const = sm.add_constant(X_clean)

    # 计算 VIF 和 TOL
    vif_df = pd.DataFrame()
    vif_df["Feature"] = actual_cols

    vif_list = []
    tol_list = []

    # 从 1 开始遍历，跳过第 0 列的 const 截距
    for i in range(1, X_with_const.shape[1]):
        vif = variance_inflation_factor(X_with_const.values, i)
        vif_list.append(vif)
        tol_list.append(1.0 / vif if vif != 0 else np.nan)

    vif_df["VIF"] = vif_list
    vif_df["TOL (Tolerance)"] = 1 / vif_df["VIF"]
    vif_df["Pass_Check (VIF<10)"] = vif_df["VIF"] < 10

    print(f"\n=================== {group_name} VIF 结果 ===================")
    print(vif_df.to_string(index=False))

    # 保存结果到当前目录
    out_file = os.path.join(data_dir, f"VIF_Result_{group_name}.csv")
    vif_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存: {out_file}")

    return vif_df

if __name__ == "__main__":
    print("--- 开始计算 Native 组 ---")
    calculate_vif_features(native_file, "Native")

    print("\n--- 开始计算 Learner 组 ---")
    calculate_vif_features(learner_file, "Learner")
