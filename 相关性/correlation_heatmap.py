import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Working directory and input files
DATA_DIR = r"C:\Users\吴佳琪\Desktop\Serbia_ML\第三次尝试（更新数据）"
NATIVE_FILE = os.path.join(DATA_DIR, "native_data.csv")
LEARNER_FILE = os.path.join(DATA_DIR, "learner_data.csv")

FEATURES = [
    "max_pos",
    "min_pos",
    "f0_max",
    "f0_min",
    "slope",
    "curve",
    "BLslope1",
    "BLslope2",
    "Duration",
    "Location",
]

# High-contrast ColorBrewer-style diverging palette:
# negative correlations = blue, zero = white, positive correlations = red.
CORRELATION_CMAP = plt.get_cmap("RdBu_r")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_correlation_matrix(file_path):
    """Read the selected variables and calculate their Pearson correlations."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    missing_columns = [column for column in FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {file_path}: {missing_columns}")

    values = df[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No complete numeric rows are available in {file_path}")

    return values.corr(method="pearson")


def draw_heatmap(ax, corr_matrix, title, cbar=True, cbar_ax=None):
    """Draw one consistently styled correlation heatmap."""
    heatmap = sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=CORRELATION_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=False,
        linewidths=0.45,
        linecolor="white",
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws={
            "ticks": np.linspace(-1, 1, 9),
            "shrink": 0.82,
            "label": "Pearson's r",
        }
        if cbar
        else None,
    )

    # Use regular-weight labels; switch to white only on strongly colored cells.
    for row in range(corr_matrix.shape[0]):
        for column in range(corr_matrix.shape[1]):
            value = corr_matrix.iloc[row, column]
            text_color = "white" if abs(value) >= 0.50 else "#222222"
            ax.text(
                column + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="normal",
                color=text_color,
            )

    ax.set_title(title, fontsize=12, fontweight="normal", pad=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=9.5,
        fontweight="normal",
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=9.5,
        fontweight="normal",
    )

    if cbar:
        colorbar = heatmap.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=9, width=0.6, length=3)
        colorbar.ax.yaxis.label.set_size(9.5)
        colorbar.ax.yaxis.label.set_weight("normal")


def save_single_heatmap(corr_matrix, group_label, output_name):
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    draw_heatmap(ax, corr_matrix, f"Variable Correlation - {group_label}")
    fig.savefig(
        os.path.join(DATA_DIR, output_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_combined_heatmap(learner_corr, native_corr):
    fig = plt.figure(figsize=(15.0, 6.2))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.035],
        left=0.06,
        right=0.94,
        bottom=0.18,
        top=0.90,
        wspace=0.28,
    )
    learner_ax = fig.add_subplot(grid[0, 0])
    native_ax = fig.add_subplot(grid[0, 1])
    colorbar_ax = fig.add_subplot(grid[0, 2])

    draw_heatmap(learner_ax, learner_corr, "L2 learners", cbar=False)
    draw_heatmap(
        native_ax,
        native_corr,
        "Native speakers",
        cbar=True,
        cbar_ax=colorbar_ax,
    )

    fig.savefig(
        os.path.join(DATA_DIR, "Correlation_Heatmaps_Combined.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(DATA_DIR, "Correlation_Heatmaps_Combined.pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    learner_corr = read_correlation_matrix(LEARNER_FILE)
    native_corr = read_correlation_matrix(NATIVE_FILE)

    save_single_heatmap(
        learner_corr,
        "L2 Learners",
        "Correlation_Heatmap_Learner.png",
    )
    save_single_heatmap(
        native_corr,
        "Native Speakers",
        "Correlation_Heatmap_Native.png",
    )
    save_combined_heatmap(learner_corr, native_corr)

    print("Correlation heatmaps have been saved to:")
    print(DATA_DIR)


if __name__ == "__main__":
    main()
