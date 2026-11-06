"""One visual grammar for every figure (Lecture 6b, slide 31: one colour per cluster, everywhere)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

INK = "#15171c"
INK2 = "#4d5261"
MUTED = "#8a8f9c"
HAIR = "#d9dce3"
SURFACE = "#f6f7f9"
CLOUD = "#9aa0ad"          # un-clustered data: grey, the first colour of every palette

NAMES = ["Low-risk", "Inflammatory", "Hepatic", "Dysglycaemic"]
# validated with dataviz/scripts/validate_palette.js --pairs all (CVD dE >= 8.6, normal >= 15)
CL = ["#1f78b4", "#d1457f", "#b07a00", "#5a3fc0"]
NEG, MID, POS = "#3b6e8f", "#f1f1ef", "#b5462f"      # diverging: below / at / above cohort mean

LABELS = {
    "glucose": "Fasting glucose", "hba1c": "HbA1c", "insulin": "Fasting insulin",
    "triglycerides": "Triglycerides", "hdl": "HDL cholesterol", "ldl": "LDL cholesterol",
    "crp": "hs-CRP", "alt": "ALT", "ggt": "GGT", "uric_acid": "Uric acid",
    "creatinine": "Creatinine", "albumin": "Albumin", "acr": "Urine albumin/creatinine",
    "wbc": "White blood cells",
    "age": "Age", "female": "Women", "bmi": "BMI", "waist": "Waist", "sbp": "Systolic BP",
    "dbp": "Diastolic BP", "income_poverty_ratio": "Income / poverty line",
    "cotinine": "Cotinine", "dx_diabetes": "Told: diabetes", "on_diabetes_meds": "Diabetes medication",
    "dx_hypertension": "Told: hypertension", "dx_high_chol": "Told: high cholesterol",
    "dx_cvd": "Heart disease or stroke", "dx_liver": "Told: liver condition",
    "told_overweight": "Told: overweight", "smoker": "Smoker (cotinine)", "college": "Some college",
    "uninsured": "Uninsured", "food_insecure": "Food insecure", "no_usual_care": "No usual place of care",
    "fair_poor_health": "Fair / poor self-rated health", "active": "Physically active",
}
UNITS = {
    "glucose": "mg/dL", "hba1c": "%", "insulin": "µU/mL", "triglycerides": "mg/dL", "hdl": "mg/dL",
    "ldl": "mg/dL", "crp": "mg/L", "alt": "U/L", "ggt": "U/L", "uric_acid": "mg/dL",
    "creatinine": "mg/dL", "albumin": "g/dL", "acr": "mg/g", "wbc": "10³/µL",
}


def setup():
    mpl.rcParams.update({
        "font.family": ["Instrument Sans", "DejaVu Sans"],
        "font.size": 12.5,
        "svg.fonttype": "none",
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlesize": 13,
        "axes.titleweight": 600,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "legend.frameon": False,
        "text.color": INK,
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=HAIR, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name, shrink=0.88):
    """Shrinking the canvas (not the type) makes every label ~14% larger on the slide."""
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * shrink, h * shrink)
    path = FIG / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # strip fixed width/height so the SVG scales with its container
    s = path.read_text()
    s = re.sub(r'(<svg[^>]*?) width="[^"]+" height="[^"]+"', r"\1", s, count=1)
    path.write_text(s)
    return path
