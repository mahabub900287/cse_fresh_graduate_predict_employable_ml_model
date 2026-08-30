"""
Regenerates Figures 5.1 and 5.2 (loaded-dataset screenshots) with a
corrected title — the dataset is the raw .xlsx file, cleaned in-pipeline
at load time, not a separate "_cleaned.csv" file.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = r"d:\laragon\www\cse_fresh_graduate_predict_employable_ml_model\my_ml_project\dataset\raw\student_career_success_dataset.xlsx"
OUT_DIR = r"d:\laragon\www\cse_fresh_graduate_predict_employable_ml_model\my_ml_project\report"

df = pd.read_excel(DATA_PATH)
df = df.drop_duplicates()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

N_ROWS = 12

FIG_5_1_COLS = [
    "Age", "Gender", "University_Year", "Major", "Attendance_Percentage",
    "Study_Hours_Per_Week", "CGPA", "Academic_Performance", "Programming_Skill",
    "Projects_Completed", "Certifications", "Hackathons", "GitHub_Profile",
    "Internships", "Leadership_Experience",
]
FIG_5_2_COLS = [
    "Academic_Performance", "Programming_Skill", "Projects_Completed",
    "Certifications", "Hackathons", "GitHub_Profile", "Internships",
    "Leadership_Experience", "Resume_Score", "Communication_Skills",
    "Teamwork", "Problem_Solving", "English_Proficiency", "Interview_Score",
]


def render_table(columns, out_path, title):
    sample = df.loc[:N_ROWS - 1, columns].copy()
    header = ["#"] + list(columns)
    rows = [[str(i + 1)] + [str(v) for v in row] for i, row in enumerate(sample.itertuples(index=False))]

    n_cols = len(header)
    fig_width = max(11, n_cols * 1.05)
    fig_height = 0.42 * (len(rows) + 1) + 0.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1b2a4a")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")
        if col == 0:
            cell.set_width(0.03)

    table.auto_set_column_width(col=list(range(n_cols)))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=14, family="serif")
    plt.tight_layout(pad=0.6)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


render_table(
    FIG_5_1_COLS,
    f"{OUT_DIR}\\figure_5_1_dataset.png",
    "student_career_success_dataset.xlsx (raw, cleaned in-pipeline) \u2014 columns Age through Leadership_Experience",
)
render_table(
    FIG_5_2_COLS,
    f"{OUT_DIR}\\figure_5_2_dataset.png",
    "student_career_success_dataset.xlsx (raw, cleaned in-pipeline) \u2014 columns Academic_Performance through Interview_Score",
)
