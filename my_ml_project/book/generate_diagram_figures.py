"""
Regenerates Figures 3.1, 3.2, 4.2 and 4.3 as clean, non-overlapping,
print-quality diagrams (matplotlib), preserving the exact content and
logical relationships already documented in the thesis text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.path import Path

OUT_DIR = r"d:\laragon\www\cse_fresh_graduate_predict_employable_ml_model\my_ml_project\book"

NAVY = "#1b2a4a"
STEEL = "#3d5a80"
TEAL = "#4a8a96"
LIGHT_BLUE = "#eaf2f8"
GREY = "#555555"
LINE = "#333333"


def box(ax, xy, w, h, text, facecolor=NAVY, textcolor="white", fontsize=10, weight="bold", rounding=0.02):
    x, y = xy
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={rounding}",
        linewidth=1.1, edgecolor=LINE, facecolor=facecolor, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=weight, color=textcolor, zorder=3, linespacing=1.4)
    return b


def arrow(ax, start, end, text=None, text_pos=0.5, color=LINE, style="-|>", connectionstyle="arc3,rad=0.0",
          text_offset=(0, 0.12), fontsize=8.5, text_bg=True):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        color=color, linewidth=1.1, connectionstyle=connectionstyle, zorder=2.5,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    if text:
        mx = start[0] + (end[0] - start[0]) * text_pos + text_offset[0]
        my = start[1] + (end[1] - start[1]) * text_pos + text_offset[1]
        bbox = dict(facecolor="white", edgecolor="none", pad=1.0) if text_bg else None
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color=GREY, zorder=4, bbox=bbox)


def new_fig(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


# =============================================================================
# Figure 3.1: The Incremental Development Model
# =============================================================================
fig, ax = new_fig(11, 6.4)

box(ax, (0.3, 2.9), 1.8, 0.9, "Outline\nDescription", facecolor="white", textcolor=NAVY, fontsize=10)

mid_x, mid_w = 3.3, 2.1
box(ax, (mid_x, 0.5), mid_w, 5.4, "", facecolor=LIGHT_BLUE, textcolor=NAVY, rounding=0.03)
ax.text(mid_x + mid_w / 2, 6.15, "Concurrent Activities", ha="center", fontsize=11, fontweight="bold", color=NAVY)

box(ax, (mid_x + 0.25, 4.35), mid_w - 0.5, 0.8, "Specification", facecolor="white", textcolor=NAVY, fontsize=10, rounding=0.4)
box(ax, (mid_x + 0.25, 2.9), mid_w - 0.5, 0.8, "Development", facecolor="white", textcolor=NAVY, fontsize=10, rounding=0.4)
box(ax, (mid_x + 0.25, 1.45), mid_w - 0.5, 0.8, "Validation", facecolor="white", textcolor=NAVY, fontsize=10, rounding=0.4)

arrow(ax, (mid_x + mid_w / 2, 4.35), (mid_x + mid_w / 2, 3.7), text=None)
arrow(ax, (mid_x + mid_w / 2, 3.7), (mid_x + mid_w / 2, 4.35), text=None, connectionstyle="arc3,rad=0.25")
arrow(ax, (mid_x + mid_w / 2, 2.9), (mid_x + mid_w / 2, 2.25), text=None)
arrow(ax, (mid_x + mid_w / 2, 2.25), (mid_x + mid_w / 2, 2.9), text=None, connectionstyle="arc3,rad=0.25")

right_x, right_w = 7.6, 3.1
box(ax, (right_x, 0.5), right_w, 5.4, "", facecolor=LIGHT_BLUE, textcolor=NAVY, rounding=0.03)
box(ax, (right_x + 0.3, 4.4), right_w - 0.6, 0.75, "Initial Version", facecolor="white", textcolor=NAVY, fontsize=9.5)
box(ax, (right_x + 0.3, 3.0), right_w - 0.6, 0.9, "Intermediate\nVersions", facecolor="white", textcolor=NAVY, fontsize=9.5)
box(ax, (right_x + 0.3, 1.5), right_w - 0.6, 0.75, "Final Version", facecolor="white", textcolor=NAVY, fontsize=9.5)

arrow(ax, (2.1, 3.35), (mid_x, 3.3), fontsize=8.5)
arrow(ax, (mid_x + mid_w, 4.85), (right_x + 0.3, 4.85), connectionstyle="arc3,rad=-0.15")
arrow(ax, (right_x + 0.3, 4.55), (mid_x + mid_w, 4.55), connectionstyle="arc3,rad=-0.15")
arrow(ax, (mid_x + mid_w, 3.5), (right_x + 0.3, 3.5), connectionstyle="arc3,rad=-0.15")
arrow(ax, (right_x + 0.3, 3.2), (mid_x + mid_w, 3.2), connectionstyle="arc3,rad=-0.15")
arrow(ax, (mid_x + mid_w, 1.85), (right_x + 0.3, 1.85), connectionstyle="arc3,rad=-0.1")

plt.tight_layout(pad=0.4)
plt.savefig(f"{OUT_DIR}\\figure_3_1_incremental_model.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved Figure 3.1")

# =============================================================================
# Figure 3.2: Incremental Development of the Employability Prediction System
# =============================================================================
fig, ax = new_fig(13, 4.6)

increments = [
    ("Increment 1", "Requirement analysis &\ndataset construction", "#16233f"),
    ("Increment 2", "Data preprocessing &\nexploratory analysis", "#223a63"),
    ("Increment 3", "Baseline model\ndevelopment", "#2f5a86"),
    ("Increment 4", "Comparative model\ntraining & tuning", "#3d7a96"),
    ("Increment 5", "Interpretation & API-\nbased prediction", "#4a9aa6"),
]
n = len(increments)
box_w, gap = 2.15, 0.35
total_w = n * box_w + (n - 1) * gap
start_x = (13 - total_w) / 2
y = 2.5
box_h = 1.55

xs = []
for i, (title, body, color) in enumerate(increments):
    x = start_x + i * (box_w + gap)
    xs.append(x)
    box(ax, (x, y), box_w, box_h, f"{title}\n\n{body}", facecolor=color, fontsize=9.3)
    if i < n - 1:
        arrow(ax, (x + box_w, y + box_h / 2), (x + box_w + gap, y + box_h / 2), fontsize=0)

# feedback loop
loop_y = 1.4
for i, x in enumerate(xs):
    ax.plot([x + 0.15, x + 0.15], [y, loop_y], color=GREY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
ax.plot([xs[0] + 0.15, xs[-1] + 0.15], [loop_y, loop_y], color=GREY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
ax.annotate("", xy=(xs[0] - 0.05, loop_y), xytext=(xs[0] + 0.15, loop_y),
            arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.0))

ax.text(13 / 2, 0.75, "Evaluate increment  \u2192  refine requirements  \u2192  proceed to next increment",
        ha="center", fontsize=10, style="italic", color=GREY)

plt.tight_layout(pad=0.4)
plt.savefig(f"{OUT_DIR}\\figure_3_2_incremental_system.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved Figure 3.2")

# =============================================================================
# Figure 4.2: Context-level Data Flow Diagram (DFD Level 0)
# =============================================================================
fig, ax = new_fig(11, 7.2)

cx, cy, r = 5.5, 3.9, 1.55
circle = Circle((cx, cy), r, facecolor=NAVY, edgecolor=LINE, linewidth=1.2, zorder=2)
ax.add_patch(circle)
ax.text(cx, cy, "0\nEmployability\nPrediction\nSystem", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white", zorder=3, linespacing=1.5)

ent_w, ent_h = 2.6, 1.1
box(ax, (0.4, 5.6), ent_w, ent_h, "Graduate /\nStudent User", facecolor=TEAL, fontsize=10.5)
box(ax, (8.0, 5.6), ent_w, ent_h, "University /\nCareer Office", facecolor=TEAL, fontsize=10.5)
box(ax, (0.4, 0.5), ent_w, ent_h, "Admin /\nResearcher", facecolor=TEAL, fontsize=10.5)

arrow(ax, (2.6, 5.9), (cx - r * 0.75, cy + r * 0.55), text="student data\n(via API / web view)",
      text_pos=0.42, text_offset=(-0.15, 0.35), fontsize=8.7)
arrow(ax, (cx + r * 0.72, cy + r * 0.58), (8.4, 5.9), text="aggregate\nemployability report",
      text_pos=0.5, text_offset=(0.1, 0.35), fontsize=8.7)
arrow(ax, (2.6, 1.1), (cx - r * 0.8, cy - r * 0.45), text="dataset admin,\nmodel retrain request",
      text_pos=0.55, text_offset=(-0.05, -0.55), fontsize=8.7)
arrow(ax, (cx - r * 0.55, cy - r * 0.75), (2.6, 0.75), text="prediction result\n(with gap analysis)",
      text_pos=0.5, text_offset=(0.15, -0.55), fontsize=8.7)

plt.tight_layout(pad=0.4)
plt.savefig(f"{OUT_DIR}\\figure_4_2_dfd_level0.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved Figure 4.2")

# =============================================================================
# Figure 4.3: Data Flow Diagram (DFD Level 1)
# Grid layout: 4 columns x 3 rows, generous gutters so no box or label
# crosses another element's path.
# =============================================================================
fig, ax = new_fig(16.5, 11.5)

col_x = [0.5, 5.0, 9.5, 14.0]   # column left-edges
row_y = [9.3, 5.6, 1.9]         # row bottom-edges (top, middle, bottom)
proc_w, proc_h = 3.1, 1.4
ent_w, ent_h = 3.1, 1.2
store_w, store_h = 3.1, 1.5

# External entities (column 0)
box(ax, (col_x[0], row_y[0] + 0.1), ent_w, ent_h, "Graduate /\nStudent User", facecolor=TEAL, fontsize=10.5)
box(ax, (col_x[0], row_y[2] + 0.1), ent_w, ent_h, "Admin /\nResearcher", facecolor=TEAL, fontsize=10.5)

# Processes (numbered) — row 0: 1.0, 2.0, 3.0 across columns 1-3
box(ax, (col_x[1], row_y[0]), proc_w, proc_h, "1.0\nCollect & validate\nstudent data", facecolor=STEEL, fontsize=9.5)
box(ax, (col_x[2], row_y[0]), proc_w, proc_h, "2.0\nPreprocess &\nengineer features", facecolor=STEEL, fontsize=9.5)
box(ax, (col_x[3], row_y[0]), proc_w, proc_h, "3.0\nTrain / select\nclassification model", facecolor=STEEL, fontsize=9.5)

# Row 1: D1 (col 1), 5.0 (col 2), 4.0 (col 3)
def data_store(ax, xy, w, h, text):
    x, y = xy
    ax.plot([x, x + w], [y + h, y + h], color=LINE, linewidth=1.2, zorder=2)
    ax.plot([x, x + w], [y, y], color=LINE, linewidth=1.2, zorder=2)
    ax.plot([x, x], [y, y + h], color=LINE, linewidth=1.2, zorder=2)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY, zorder=3)

data_store(ax, (col_x[1], row_y[1] - 0.1), store_w, store_h, "D1  Student\nData Store")
box(ax, (col_x[2], row_y[1]), proc_w, proc_h, "5.0\nReturn prediction\n(API response)", facecolor=STEEL, fontsize=9.5)
box(ax, (col_x[3], row_y[1]), proc_w, proc_h, "4.0\nGenerate prediction\n& gap analysis", facecolor=STEEL, fontsize=9.5)

# Row 2: 6.0 (col 0, below Admin), D2 (col 2), D3 (col 3)
box(ax, (col_x[0], row_y[2] - 1.5), proc_w, proc_h, "6.0\nManage dataset &\nmodel retraining", facecolor=STEEL, fontsize=9.5)
data_store(ax, (col_x[2], row_y[2] - 1.6), store_w, store_h, "D2  Trained\nModel Store")
data_store(ax, (col_x[3], row_y[2] - 1.6), store_w, store_h, "D3  Prediction\nLog")

cxs = [x + proc_w / 2 for x in col_x]  # column centre x's (approx, reused loosely below)

# --- Row 0 flows: entity -> 1.0 -> 2.0 -> 3.0 ---
ent_top_mid = (col_x[0] + ent_w / 2, row_y[0] + 0.1 + ent_h)
arrow(ax, (col_x[0] + ent_w, row_y[0] + 0.1 + ent_h * 0.5), (col_x[1], row_y[0] + proc_h * 0.5),
      text="student data\n(API / web view)", text_pos=0.5, text_offset=(0, 0.35), fontsize=8.5)
arrow(ax, (col_x[1] + proc_w, row_y[0] + proc_h * 0.5), (col_x[2], row_y[0] + proc_h * 0.5),
      text="raw data", fontsize=8.5)
arrow(ax, (col_x[2] + proc_w, row_y[0] + proc_h * 0.5), (col_x[3], row_y[0] + proc_h * 0.5),
      text="feature vector", fontsize=8.5)

# 1.0 -> D1 (validated record), straight down
arrow(ax, (col_x[1] + proc_w * 0.3, row_y[0]), (col_x[1] + proc_w * 0.3, row_y[1] + store_h - 0.1),
      text="validated record", text_pos=0.5, text_offset=(1.0, 0), fontsize=8.5)

# 3.0 -> 4.0 (trained model), straight down
arrow(ax, (col_x[3] + proc_w * 0.5, row_y[0]), (col_x[3] + proc_w * 0.5, row_y[1] + proc_h),
      text="trained model", fontsize=8.5)

# --- Row 1 flows ---
# D1 -> 5.0 (stored record, upper lane) and 5.0 -> D1 (API response context, lower lane) kept vertically separated
arrow(ax, (col_x[1] + store_w, row_y[1] + store_h * 0.68), (col_x[2], row_y[1] + proc_h * 0.75),
      text="stored record", text_pos=0.5, text_offset=(0, 0.28), fontsize=8.5)
arrow(ax, (col_x[2], row_y[1] + proc_h * 0.3), (col_x[1] + store_w, row_y[1] + store_h * 0.3),
      text=None)

# 5.0 <-> 4.0 (prediction + gap analysis in, nothing back needed structurally but keep single flow)
arrow(ax, (col_x[3], row_y[1] + proc_h * 0.5), (col_x[2] + proc_w, row_y[1] + proc_h * 0.5),
      text="prediction + gap analysis", text_pos=0.5, text_offset=(0, 0.3), fontsize=8.5)

# 5.0 -> Graduate/Student User (API response), curved above everything
arrow(ax, (col_x[2] + proc_w * 0.15, row_y[1] + proc_h), (col_x[0] + ent_w * 0.7, row_y[0] + 0.1),
      text="API response (result & gap analysis)", text_pos=0.4, text_offset=(0.3, 0.5), fontsize=8.3,
      connectionstyle="arc3,rad=0.2")

# 5.0 -> D3 Prediction Log (prediction record), straight down
arrow(ax, (col_x[2] + proc_w * 0.5, row_y[1]), (col_x[3] + store_w * 0.5, row_y[2] - 1.6 + store_h),
      text="prediction record", text_pos=0.55, text_offset=(1.1, 0.2), fontsize=8.5,
      connectionstyle="arc3,rad=-0.15")

# D3 -> D2 (trained model ref.)
arrow(ax, (col_x[3], row_y[2] - 1.6 + store_h * 0.5), (col_x[2] + store_w, row_y[2] - 1.6 + store_h * 0.5),
      text="trained model ref.", fontsize=8.3)

# --- Row 2 flows ---
# Admin -> 6.0
arrow(ax, (col_x[0] + ent_w * 0.35, row_y[2] + 0.1), (col_x[0] + proc_w * 0.35, row_y[2] - 1.5 + proc_h),
      text="retrain /\nupdate request", text_pos=0.5, text_offset=(1.3, 0), fontsize=8.3)
# 6.0 -> D1 (corrected records)
arrow(ax, (col_x[0] + proc_w * 0.85, row_y[2] - 1.5 + proc_h), (col_x[1] + store_w * 0.3, row_y[1] - 0.1),
      text="corrected records", text_pos=0.45, text_offset=(0.75, -0.15), fontsize=8.3,
      connectionstyle="arc3,rad=-0.15")
# 6.0 -> D2 (updated model)
arrow(ax, (col_x[0] + proc_w, row_y[2] - 1.5 + proc_h * 0.5), (col_x[2], row_y[2] - 1.6 + store_h * 0.5),
      text="updated model", text_pos=0.5, text_offset=(0, -0.3), fontsize=8.5)

plt.tight_layout(pad=0.4)
plt.savefig(f"{OUT_DIR}\\figure_4_3_dfd_level1.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved Figure 4.3")
