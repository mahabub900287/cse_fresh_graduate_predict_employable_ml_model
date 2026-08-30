"""
Regenerates Figure 4.1 (ML model workflow, 9 stages) with the correct
dataset size (50,000 records, not 5,000+), matching the existing visual
style (navy-to-teal gradient boxes, same as the other regenerated figures).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = r"d:\laragon\www\cse_fresh_graduate_predict_employable_ml_model\my_ml_project\report\figure_4_1_workflow.png"

LINE = "#333333"
GREY = "#444444"

STAGES = [
    ("1. Data Collection\n(Kaggle dataset, 50,000 records)", "#16233f"),
    ("2. Data Preprocessing", "#223a63"),
    ("3. Feature Engineering", "#2a4a78"),
    ("4. Exploratory Data Analysis (EDA)", "#2f5a86"),
    ("5. Training / Testing Dataset Split", "#356a90"),
    ("6. Machine Learning Model Training", "#3d7a96"),
    ("7. Model Evaluation", "#42869e"),
    ("8. Model Selection", "#4790a4"),
    ("9. Employability Prediction\n(returned via API / web view)", "#4a9aa6"),
]

fig, ax = plt.subplots(figsize=(9, 12.6), dpi=200)
ax.set_xlim(0, 9)
ax.set_ylim(0, 12.6)
ax.axis("off")

box_w, box_h, gap = 5.6, 1.0, 0.28
x0 = 0.3
n = len(STAGES)
total_h = n * box_h + (n - 1) * gap
y_top = 12.3

ys = []
for i, (text, color) in enumerate(STAGES):
    y = y_top - (i + 1) * box_h - i * gap
    ys.append(y)
    b = FancyBboxPatch(
        (x0, y), box_w, box_h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.0, edgecolor=LINE, facecolor=color, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x0 + box_w / 2, y + box_h / 2, text, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=3, linespacing=1.4)
    if i < n - 1:
        arrow_y_top = y
        arrow_y_bot = y - gap
        a = FancyArrowPatch(
            (x0 + box_w / 2, arrow_y_top), (x0 + box_w / 2, arrow_y_bot),
            arrowstyle="-|>", mutation_scale=14, color=LINE, linewidth=1.1, zorder=2.5,
        )
        ax.add_patch(a)

# side annotation: new student data path (API prediction loop)
api_y_top = ys[0] + box_h * 0.4
api_y_bot = ys[7] + box_h * 0.15
ax.annotate(
    "", xy=(x0 + box_w + 0.15, api_y_bot), xytext=(x0 + box_w + 0.15, api_y_top),
    arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.1,
                     connectionstyle="arc3,rad=0.15"),
)
ax.text(x0 + box_w + 0.3, api_y_top + 0.05, "New student data\n(submitted via API / web view)",
        fontsize=9.5, color=GREY, va="bottom")
ax.text(x0 + box_w + 0.3, api_y_bot - 0.05, "Same preprocessing &\nfeature transformation\napplied to the new record",
        fontsize=9.5, color=GREY, va="top", style="italic")

plt.tight_layout(pad=0.4)
plt.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {OUT_PATH}")
