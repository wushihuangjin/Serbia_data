import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    auc,
)
from matplotlib import pyplot as plt

# 工作路径与数据文件
work_dir = r"C:/Users/吴佳琪/Desktop/Serbia_ML/第三次尝试（更新数据）"
data_dir = os.path.join(work_dir, "RF")
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(work_dir, "native_data.csv")
group_name = "Native"

# 十个声学特征
feature_cols = [
    "max_pos", "min_pos", "f0_max", "f0_min", "slope",
    "curve", "BLslope1", "BLslope2", "Duration", "Location",
]
tone_mapping = {"Tone2": 0, "Tone3": 1}

# 读取数据，并在脚本内完成二声、三声编码
df = pd.read_csv(data_file, encoding="utf-8-sig")
df["Tone_Label"] = df["Tone"].map(tone_mapping)

model_data = df[feature_cols + ["Tone_Label"]].copy()
for col in feature_cols:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
model_data = model_data.dropna()

# 相关性分析
plt.figure(figsize=(11, 9))
sns.heatmap(model_data.corr(), cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
plt.title(f"Variable Correlation - {group_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(data_dir, f"RF_Correlation_Heatmap_{group_name}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 划分自变量和目标变量：70%训练集，30%测试集
X = model_data[feature_cols]
y = model_data["Tone_Label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# 设置随机森林模型的参数
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# 使用class_weight='balanced'处理类别差异
rf = RandomForestClassifier(random_state=42, class_weight="balanced")

# 五折交叉验证结合网格搜索确定最佳参数
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=1,
)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Score: ", grid_search.best_score_)

# 保存网格搜索准确率图
grid_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 8))
plt.plot(
    range(1, len(grid_results) + 1),
    grid_results["mean_test_score"],
    color="#d7191c",
    linestyle="--",
    label="Accuracy",
)
plt.title(f"Random Forest Grid Search Accuracy - {group_name}", fontsize=16)
plt.xlabel("Parameter Set", fontsize=14)
plt.ylabel("Accuracy Score", fontsize=14)
plt.grid(False)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(
    os.path.join(data_dir, f"RF_Grid_Search_Accuracy_{group_name}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 使用最佳模型预测测试集
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_test, predictions)
oa = accuracy_score(y_test, predictions)
kappa = cohen_kappa_score(y_test, predictions)

print("Accuracy of the best model: ", accuracy)
print("Classification Report:\n", classification_report(
    y_test,
    predictions,
    target_names=["Tone2", "Tone3"],
    zero_division=0,
))
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Matthews Correlation Coefficient (MCC): ", mcc)
print("Overall Accuracy (OA): ", oa)
print("Cohen's Kappa: ", kappa)

# 保存模型评价指标
metrics_df = pd.DataFrame([{
    "Group": group_name,
    "Model": "RF",
    "Best Parameters": str(grid_search.best_params_),
    "Best Cross-Validation Score": grid_search.best_score_,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "MCC": mcc,
    "OA": oa,
    "Kappa": kappa,
    "AUC": roc_auc,
}])
metrics_df.to_csv(
    os.path.join(data_dir, f"RF_Metrics_{group_name}.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Tone2", "Tone3"],
    yticklabels=["Tone2", "Tone3"],
)
plt.title(f"Confusion Matrix - {group_name}", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.tight_layout()
plt.savefig(
    os.path.join(data_dir, f"RF_Confusion_Matrix_{group_name}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 随机森林特征重要性及相对重要性百分比
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": best_rf.feature_importances_,
})
importance_df["Relative Importance (%)"] = importance_df["Importance"] * 100
importance_df = importance_df.sort_values(
    by="Relative Importance (%)",
    ascending=False,
).reset_index(drop=True)
importance_df.insert(0, "Rank", np.arange(1, len(importance_df) + 1))

importance_df.to_csv(
    os.path.join(data_dir, f"RF_Feature_Importance_{group_name}.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\nRandom Forest Feature Importance Ranking:")
print(importance_df.to_string(index=False))

# 特征相对重要性排序图
plot_df = importance_df.iloc[::-1]
colors = plt.cm.Oranges(np.linspace(0.25, 0.85, len(plot_df)))
plt.figure(figsize=(12, 7))
bars = plt.barh(
    plot_df["Feature"],
    plot_df["Relative Importance (%)"],
    color=colors,
    edgecolor="black",
)
for bar, value in zip(bars, plot_df["Relative Importance (%)"]):
    plt.text(
        value + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center",
        fontsize=11,
    )
plt.title(f"Random Forest Feature Importance - {group_name} Group", fontsize=16)
plt.xlabel("Relative Importance (%)", fontsize=14)
plt.ylabel("Acoustic Feature", fontsize=14)
plt.xlim(0, importance_df["Relative Importance (%)"].max() * 1.22)
plt.tight_layout()
plt.savefig(
    os.path.join(data_dir, f"RF_Feature_Importance_{group_name}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 前100个测试样本的真实值与预测值
sample_count = min(100, len(y_test))
plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(sample_count),
    y_test.to_numpy()[:sample_count],
    "go--",
    label="True value",
)
plt.plot(
    np.arange(sample_count),
    predictions[:sample_count],
    "ro-",
    label="Predicted value",
)
plt.title(f"True vs Predicted Labels - {group_name}")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(
    os.path.join(data_dir, f"RF_True_vs_Predicted_{group_name}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ROC曲线和AUC值
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Receiver Operating Characteristic - {group_name}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    os.path.join(data_dir, f"RF_ROC_Curve_{group_name}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(f"\nAll results have been saved to: {data_dir}")
