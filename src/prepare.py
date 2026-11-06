"""
Step 1 - Build the analytic dataset from the raw NHANES 2017-2018 files.

Two blocks are kept strictly apart (Lecture 6b, slide 3):
  * INTERNAL block  - blood & urine biomarkers. ONLY these build the clusters.
  * EXTERNAL block  - everything the clustering never sees: age, sex, body
                      measures, blood pressure, doctor diagnoses, medication,
                      income, education, insurance, food security, lifestyle.

Output: data/processed/nhanes_adults_fasting.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def read(name, cols):
    df = pd.read_sas(RAW / f"{name}_J.xpt", format="xport")
    return df[["SEQN"] + cols]


def yes(x):
    """NHANES coding: 1 = yes, 2 = no, 7/9 = refused / don't know -> NaN."""
    return np.where(x == 1, 1.0, np.where(x == 2, 0.0, np.nan))


# ---------------------------------------------------------------- raw tables
demo = read("DEMO", ["RIAGENDR", "RIDAGEYR", "DMDEDUC2", "INDFMPIR", "RIDEXPRG"])
glu = read("GLU", ["WTSAF2YR", "LBXGLU"])
ins = read("INS", ["LBXIN"])
tri = read("TRIGLY", ["LBXTR", "LBDLDL"])
hdl = read("HDL", ["LBDHDD"])
ghb = read("GHB", ["LBXGH"])
crp = read("HSCRP", ["LBXHSCRP"])
bio = read("BIOPRO", ["LBXSATSI", "LBXSGTSI", "LBXSUA", "LBXSCR", "LBXSAL"])
acr = read("ALB_CR", ["URDACT"])
cbc = read("CBC", ["LBXWBCSI"])
cot = read("COT", ["LBXCOT"])
bmx = read("BMX", ["BMXBMI", "BMXWAIST"])
bpx = read("BPX", ["BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"])
diq = read("DIQ", ["DIQ010", "DIQ160", "DIQ050", "DIQ070"])
bpq = read("BPQ", ["BPQ020", "BPQ080", "BPQ050A"])
mcq = read("MCQ", ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ160L", "MCQ080"])
hiq = read("HIQ", ["HIQ011"])
huq = read("HUQ", ["HUQ030", "HUQ010"])
fsq = read("FSQ", ["FSDAD"])
paq = read("PAQ", ["PAQ650", "PAQ665"])

df = demo
for t in [glu, ins, tri, hdl, ghb, crp, bio, acr, cbc, cot, bmx, bpx,
          diq, bpq, mcq, hiq, huq, fsq, paq]:
    df = df.merge(t, on="SEQN", how="left")

n_total = len(df)

# --------------------------------------------------------------- population
# adults (>= 20), in the fasting morning subsample, not pregnant
df = df[(df.RIDAGEYR >= 20) & (df.WTSAF2YR > 0)]
n_fasting = len(df)
df = df[df.RIDEXPRG != 1]

# ------------------------------------------------------- internal block
internal = {
    "glucose":       "LBXGLU",    # fasting plasma glucose, mg/dL
    "hba1c":         "LBXGH",     # glycohemoglobin, %
    "insulin":       "LBXIN",     # fasting insulin, uU/mL
    "triglycerides": "LBXTR",     # mg/dL
    "hdl":           "LBDHDD",    # mg/dL
    "ldl":           "LBDLDL",    # Friedewald LDL, mg/dL
    "crp":           "LBXHSCRP",  # high-sensitivity CRP, mg/L
    "alt":           "LBXSATSI",  # alanine aminotransferase, U/L
    "ggt":           "LBXSGTSI",  # gamma-glutamyl transferase, U/L
    "uric_acid":     "LBXSUA",    # mg/dL
    "creatinine":    "LBXSCR",    # serum creatinine, mg/dL
    "albumin":       "LBXSAL",    # serum albumin, g/dL
    "acr":           "URDACT",    # urine albumin/creatinine ratio, mg/g
    "wbc":           "LBXWBCSI",  # white blood cells, 1000 cells/uL
}
for k, v in internal.items():
    df[k] = df[v]

# ------------------------------------------------------- external block
ext = pd.DataFrame(index=df.index)
ext["age"] = df.RIDAGEYR
ext["female"] = (df.RIAGENDR == 2).astype(float)
ext["bmi"] = df.BMXBMI
ext["waist"] = df.BMXWAIST
ext["sbp"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
ext["dbp"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1).replace(0, np.nan)
ext["cotinine"] = df.LBXCOT  # tobacco exposure (smoking) - lifestyle, not metabolism
ext["smoker"] = (df.LBXCOT >= 10).astype(float).where(df.LBXCOT.notna())

# doctor told you...
ext["dx_diabetes"] = np.where(df.DIQ010 == 1, 1.0,
                       np.where(df.DIQ010.isin([2, 3]), 0.0, np.nan))  # 3 = borderline -> not told diabetes
ext["dx_prediabetes"] = yes(df.DIQ160)
ext["dx_hypertension"] = yes(df.BPQ020)
ext["dx_high_chol"] = yes(df.BPQ080)
cvd = pd.DataFrame({c: yes(df[c]) for c in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"]}, index=df.index)
ext["dx_cvd"] = cvd.max(axis=1).where(cvd.notna().any(axis=1))   # CHF / CHD / MI / stroke
ext["dx_liver"] = yes(df.MCQ160L)
ext["told_overweight"] = yes(df.MCQ080)
ext["on_diabetes_meds"] = np.fmax(yes(df.DIQ050), yes(df.DIQ070))

# socioeconomic
ext["income_poverty_ratio"] = df.INDFMPIR  # family income / poverty line (capped at 5)
ext["college"] = np.where(df.DMDEDUC2.isin([4, 5]), 1.0,
                   np.where(df.DMDEDUC2.isin([1, 2, 3]), 0.0, np.nan))
ext["uninsured"] = np.where(df.HIQ011 == 2, 1.0, np.where(df.HIQ011 == 1, 0.0, np.nan))
ext["food_insecure"] = np.where(df.FSDAD.isin([3, 4]), 1.0,
                         np.where(df.FSDAD.isin([1, 2]), 0.0, np.nan))
ext["no_usual_care"] = np.where(df.HUQ030 == 2, 1.0,
                         np.where(df.HUQ030.isin([1, 3]), 0.0, np.nan))
ext["fair_poor_health"] = np.where(df.HUQ010.isin([4, 5]), 1.0,
                            np.where(df.HUQ010.isin([1, 2, 3]), 0.0, np.nan))
ext["active"] = np.fmax(yes(df.PAQ650), yes(df.PAQ665))

out = pd.concat([df[["SEQN", "WTSAF2YR"]], df[list(internal)], ext], axis=1)

n_before_cc = len(out)
out = out.dropna(subset=list(internal))          # complete cases on the clustering block
out.to_csv(OUT / "nhanes_adults_fasting.csv", index=False)

flow = pd.DataFrame({
    "step": ["NHANES 2017-18 participants", "Adults in fasting subsample",
             "Not pregnant", "Complete on 14 biomarkers"],
    "n": [n_total, n_fasting, n_before_cc, len(out)],
})
flow.to_csv(OUT / "flow.csv", index=False)
print(flow.to_string(index=False))
print(out[list(internal)].describe().T.round(2))
print(out[ext.columns].isna().mean().round(3))
