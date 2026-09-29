import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


work_dir = r"C:/Users/吴佳琪/Desktop/Serbia_ML/第三次尝试（更新数据）"
data_file = os.path.join(work_dir, "learner_data.csv")
output_dir = os.path.join(work_dir, "modelcomparison_ROC")
os.makedirs(output_dir, exist_ok=True)

group_name = "Learner"
feature_cols = [
    "max_pos", "min_pos", "f0_max", "f0_min", "slope",
    "curve", "BLslope1", "BLslope2", "Duration", "Location",
]
tone_mapping = {"Tone2": 0, "Tone3": 1}


def save_metric_table(result_df):
    display_df = result_df.copy()
    metric_cols = [
        "Precision", "Recall", "F1 Score", "MCC", "OA", "Kappa", "AUC"
    ]
    for col in metric_cols:
        display_df[col] = display_df[col].map(lambda value: f"{value:.3f}")

    fig, ax = plt.subplots(figsize=(11, 3.3), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.16, 0.98, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("white")
        cell.set_text_props(fontfamily="DejaVu Serif")
        if col == 0:
            cell.get_text().set_ha("left")

    ax.plot([0.01, 0.99], [0.87, 0.87], color="black", linewidth=1.4,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.01, 0.99], [0.68, 0.68], color="black", linewidth=0.9,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0.01, 0.99], [0.13, 0.13], color="black", linewidth=1.4,
            transform=ax.transAxes, clip_on=False)

    table_path = os.path.join(
        output_dir, f"Three_Models_Results_{group_name}.png"
    )
    plt.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table_path


df = pd.read_csv(data_file, encoding="utf-8-sig")
model_data = df[feature_cols + ["Tone"]].copy()
for col in feature_cols:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
model_data["Tone_Label"] = model_data["Tone"].map(tone_mapping)
model_data = model_data.dropna(subset=feature_cols + ["Tone_Label"])

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

models = {
    "SVM": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42, probability=True)),
        ]),
        "param_grid": {
            "model__C": [0.1, 1, 10],
            "model__gamma": [0.1, 0.01],
            "model__kernel": ["rbf", "linear"],
        },
        "color": "#6A5ACD",
    },
    "KNN": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 10],
            "model__weights": ["uniform", "distance"],
        },
        "color": "#FF7F50",
    },
    "RF": {
        "estimator": RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "color": "#ADD8E6",
    },
}

metric_rows = []
parameter_rows = []
roc_results = {}

print(f"Group: {group_name}")
print(f"Total samples: {len(model_data)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

for model_name, config in models.items():
    print(f"\nTraining {model_name}...")
    grid_search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        cv=5,
        scoring="accuracy",
        verbose=2,
        n_jobs=1,
    )
    grid_search.fit(X_train, y_train)

    # All metrics below come from this same fitted best_model.
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probability, pos_label=1)
    model_auc = roc_auc_score(y_test, y_probability)

    metric_rows.append({
        "Model": model_name,
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions),
        "OA": accuracy_score(y_test, predictions),
        "Kappa": cohen_kappa_score(y_test, predictions),
        "AUC": model_auc,
    })
    parameter_rows.append({
        "Model": model_name,
        "Best Parameters": json.dumps(
            grid_search.best_params_, ensure_ascii=False
        ),
        "Best Cross-Validation Score": grid_search.best_score_,
    })
    roc_results[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": model_auc,
        "color": config["color"],
    }

    print("Best parameters:", grid_search.best_params_)
    print(f"Best five-fold CV score: {grid_search.best_score_:.4f}")
    print(f"Test AUC: {model_auc:.4f}")

result_df = pd.DataFrame(metric_rows)
parameter_df = pd.DataFrame(parameter_rows)
result_csv = os.path.join(output_dir, f"Three_Models_Results_{group_name}.csv")
parameter_csv = os.path.join(
    output_dir, f"Three_Models_Best_Parameters_{group_name}.csv"
)
result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
parameter_df.to_csv(parameter_csv, index=False, encoding="utf-8-sig")
table_path = save_metric_table(result_df)

plt.figure(figsize=(8, 6), dpi=300)
for model_name in ["RF", "KNN", "SVM"]:
    result = roc_results[model_name]
    display_name = "Random Forest" if model_name == "RF" else model_name
    plt.plot(
        result["fpr"],
        result["tpr"],
        color=result["color"],
        linewidth=2,
        label=f"{display_name} (AUC = {result['auc']:.3f})",
    )
plt.plot(
    [0, 1], [0, 1], color="#2E8B57", linestyle="--", linewidth=1.5,
    label="Chance level",
)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for SVM, KNN, and Random Forest - Learner Group")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(output_dir, "Three_Models_ROC_Learner.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nModel comparison:")
print(result_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"Results saved to: {result_csv}")
print(f"Parameters saved to: {parameter_csv}")
print(f"Metric table saved to: {table_path}")
print(f"ROC figure saved to: {roc_path}")
