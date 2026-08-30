"""
Regenerates Figure 5.7 (console output of the executed pipeline) using the
actual, freshly captured output of src/train_model.py, in the same monospace
terminal-screenshot style as the existing figure. Replaces the previous
version's stale hyperparameters (n_estimators=100, colsample_bytree=0.8,
"Best CV accuracy: 0.8172") which did not match Table 5.7 / the real run.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = r"d:\laragon\www\cse_fresh_graduate_predict_employable_ml_model\my_ml_project\report\figure_5_7_console.png"

LINES = [
    ">>> Loading dataset from: dataset/raw/student_career_success_dataset.xlsx",
    ">>> Loaded 50000 instances and 23 columns (after dropping 6 post-outcome columns).",
    ">>> Label classes (alphabetical): ['Employable', 'Not Employable']",
    ">>> Retained 18 features -> 14 numeric, 4 categorical.",
    ">>> Train: 40000 rows, Test: 10000 rows",
    "",
    ">>> Running RandomizedSearchCV (15 candidates, 3-fold CV)...",
    ">>> Search completed in 53.6 seconds.",
    ">>> Best params: n_estimators=200, learning_rate=0.05, max_depth=3,",
    "                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1",
    ">>> Final fit on 40000 training records took 0.97 seconds.",
    "",
    "=== Classification report (threshold = 0.50, no SMOTE) ===",
    "                precision    recall  f1-score   support",
    "",
    "    Employable       0.84      0.95      0.89      7808",
    "Not Employable       0.65      0.36      0.46      2192",
    "",
    "      accuracy                           0.82     10000",
    "     macro avg       0.75      0.65      0.67     10000",
    "  weighted avg       0.80      0.82      0.80     10000",
    "",
    ">>> Confusion matrix:",
    ">>>  [[7391  417]",
    ">>>   [1413  779]]",
    ">>> Accuracy: 0.8170",
    ">>> ROC-AUC (Not Employable probability): 0.8068",
]

n_lines = len(LINES)
fig_width = 11.2
fig_height = 0.245 * n_lines + 0.35
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

line_h = 1.0 / (n_lines + 0.6)
y = 1.0 - line_h * 0.6
for line in LINES:
    ax.text(0.012, y, line, family="monospace", fontsize=10.3, va="top", ha="left", color="#111111")
    y -= line_h

plt.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {OUT_PATH}")
