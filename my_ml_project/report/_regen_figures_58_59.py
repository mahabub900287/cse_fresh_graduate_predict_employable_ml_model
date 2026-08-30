"""
Regenerates Figures 5.8 and 5.9 (confusion matrix heatmaps) with the correct
cell values 7391/417/1413/779 (previously stale 7395/413/1415/777), matching
the classification report / Table 5.9 / Figure 5.7. Same visual style as the
originals (navy Blues colormap, bold white/black annotations).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = r"d:\laragon\www\cse_fresh_graduate_predict_employable_ml_model\my_ml_project\report"

CM = np.array([[7391, 417], [1413, 779]])
LABELS = ["Employable", "Not Employable"]


def render(figsize, out_path):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    im = ax.imshow(CM, cmap="Blues", vmin=0, vmax=CM.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.set_yticklabels(LABELS, fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix - XGBoost (tuned, threshold=0.50)", fontsize=13)

    thresh = CM.max() / 2
    for i in range(2):
        for j in range(2):
            val = CM[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                     fontsize=17, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


render((7.5, 6.2), f"{OUT_DIR}\\figure_5_8_confusion_matrix.png")
render((6.2, 5.2), f"{OUT_DIR}\\figure_5_9_confusion_matrix.png")
