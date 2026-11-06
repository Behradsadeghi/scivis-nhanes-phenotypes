"""
Step 2 - Cluster the biomarker block, then describe and validate the clusters.

Follows the recipe of Lectures 6a / 6b / 7:
  look -> transform -> cluster (k-means, many restarts) + a second method
  -> choose k with stability, not with one score -> check k-1 and k+1
  -> describe (internal block)  -> validate (external block, never seen)
  -> tests: Kruskal-Wallis / chi2, Benjamini-Hochberg across variables,
     pairwise Wilcoxon / Fisher with Holm inside each variable, effect sizes
  -> surrogate models: random forest (permutation importance on held-out folds)
     and multinomial L1 logistic regression (odds ratios vs a reference cluster)
  -> one patient explained with SHAP.

Everything is written to results/ so that figures.py never recomputes anything.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

d = pd.read_csv(ROOT / "data" / "processed" / "nhanes_adults_fasting.csv")

INTERNAL = ["glucose", "hba1c", "insulin", "triglycerides", "hdl", "ldl", "crp",
            "alt", "ggt", "uric_acid", "creatinine", "albumin", "acr", "wbc"]
LOGGED = ["glucose", "hba1c", "insulin", "triglycerides", "hdl", "crp", "alt",
          "ggt", "creatinine", "acr", "wbc"]
EXT_CONT = ["age", "bmi", "waist", "sbp", "dbp", "income_poverty_ratio", "cotinine"]
EXT_BIN = ["female", "dx_diabetes", "on_diabetes_meds", "dx_hypertension", "dx_high_chol",
           "dx_cvd", "dx_liver", "told_overweight", "smoker", "college", "uninsured",
           "food_insecure", "no_usual_care", "fair_poor_health", "active"]

# ------------------------------------------------------------------ transform
X = d[INTERNAL].copy()
for c in LOGGED:
    X[c] = np.log(X[c])
scaler = StandardScaler().fit(X)
Z = np.clip(scaler.transform(X), -4, 4)   # log + z-score; clip the 0.3% beyond 4 SD
pd.DataFrame(Z, columns=INTERNAL).to_csv(RES / "z_internal.csv", index=False)

# ------------------------------------------------------------- choosing k
rng = np.random.default_rng(SEED)
rows = []
labels_by_k = {}
for k in range(2, 9):
    km = KMeans(k, n_init=100, random_state=SEED).fit(Z)
    lab = km.labels_
    labels_by_k[k] = lab
    gmm = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=SEED).fit(Z).predict(Z)
    ward = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    boot = []
    for b in range(30):
        idx = rng.choice(len(Z), len(Z), replace=True)
        kb = KMeans(k, n_init=10, random_state=b).fit(Z[idx])
        boot.append(adjusted_rand_score(lab, kb.predict(Z)))
    rows.append(dict(k=k, silhouette=silhouette_score(Z, lab, random_state=SEED),
                     inertia=km.inertia_,
                     ari_gmm=adjusted_rand_score(lab, gmm), ari_ward=adjusted_rand_score(lab, ward),
                     boot_mean=np.mean(boot), boot_lo=np.percentile(boot, 5),
                     boot_hi=np.percentile(boot, 95), min_size=np.bincount(lab).min()))
kdf = pd.DataFrame(rows)
kdf.to_csv(RES / "k_selection.csv", index=False)
print(kdf.round(3).to_string(index=False))

K = 4
lab = labels_by_k[K]
cent = pd.DataFrame(KMeans(K, n_init=100, random_state=SEED).fit(Z).cluster_centers_, columns=INTERNAL)

# ---- name clusters from their pattern (deterministic, seed-independent)
g = pd.DataFrame(Z, columns=INTERNAL).groupby(lab).mean()
dys = g.hba1c.idxmax()
low = g.drop(index=dys).insulin.idxmin()
hep = g.drop(index=[dys, low]).alt.idxmax()
inf = [c for c in g.index if c not in (dys, low, hep)][0]
order = [low, inf, hep, dys]
NAMES = ["Low-risk", "Inflammatory", "Hepatic", "Dysglycaemic"]
remap = {old: new for new, old in enumerate(order)}
d["cluster"] = pd.Series(lab).map(remap).values
d["cluster_name"] = d.cluster.map(dict(enumerate(NAMES)))


# k-1 / k / k+1 for the alluvial
d["k3"] = labels_by_k[3]
d["k5"] = labels_by_k[5]

# ---- a second method that fails differently: mean-shift looks for density peaks
bw = estimate_bandwidth(Z, quantile=0.2, random_state=SEED)
ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(Z)
ms_sizes = np.bincount(ms.labels_)
print("mean-shift modes:", len(ms_sizes), "sizes:", ms_sizes[:10])

# ------------------------------------------------------- embeddings (display only)
pca = PCA(n_components=len(INTERNAL), random_state=SEED).fit(Z)
P = pca.transform(Z)
import umap
U = umap.UMAP(n_neighbors=30, min_dist=0.35, random_state=SEED).fit_transform(Z)
emb = pd.DataFrame({"pc1": P[:, 0], "pc2": P[:, 1], "u1": U[:, 0], "u2": U[:, 1],
                    "cluster": d.cluster, "seqn": d.SEQN})
emb.to_csv(RES / "embedding.csv", index=False)
pd.DataFrame({"pc": np.arange(1, len(INTERNAL) + 1),
              "explained": pca.explained_variance_ratio_}).to_csv(RES / "pca_scree.csv", index=False)
pd.DataFrame(pca.components_[:3].T, index=INTERNAL, columns=["PC1", "PC2", "PC3"]).to_csv(RES / "pca_loadings.csv")

# ------------------------------------------------------- lab-defined conditions
d["lab_diabetes"] = ((d.hba1c >= 6.5) | (d.glucose >= 126)).astype(int)
d["lab_prediabetes"] = (((d.hba1c >= 5.7) & (d.hba1c < 6.5)) |
                        ((d.glucose >= 100) & (d.glucose < 126))).astype(int) * (1 - d.lab_diabetes)
# AASLD upper limits of normal for ALT: 33 U/L men, 25 U/L women
d["alt_high"] = np.where(d.female == 1, d.alt > 25, d.alt > 33).astype(int)
d["undiagnosed_diabetes"] = ((d.lab_diabetes == 1) & (d.dx_diabetes == 0)).astype(int)
d.to_csv(RES / "patients.csv", index=False)

# ------------------------------------------------------------ statistics


def eps2_kw(h, n):
    return h / ((n ** 2 - 1) / (n + 1))      # epsilon-squared for Kruskal-Wallis


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))


def test_block(cols, binary, block):
    out = []
    for v in cols:
        s = d[[v, "cluster"]].dropna()
        if binary:
            ct = pd.crosstab(s.cluster, s[v])
            if ct.shape[1] < 2:
                continue
            p = stats.chi2_contingency(ct)[1]
            eff = cramers_v(ct)
            test = "chi2"
        else:
            groups = [s[v][s.cluster == k] for k in range(K)]
            h, p = stats.kruskal(*groups)
            eff = eps2_kw(h, len(s))
            test = "Kruskal-Wallis"
        # one-vs-rest: standardised difference of the cluster that departs most
        sd = s[v].std()
        diffs = [(s[v][s.cluster == k].mean() - s[v][s.cluster != k].mean()) / sd for k in range(K)]
        kmax = int(np.argmax(np.abs(diffs)))
        out.append(dict(variable=v, block=block, type="binary" if binary else "continuous",
                        test=test, p=p, effect=eff, n=len(s),
                        top_cluster=kmax, top_std_diff=diffs[kmax]))
    return out


ext = pd.DataFrame(test_block(EXT_CONT, False, "external") + test_block(EXT_BIN, True, "external"))
ext["q"] = multipletests(ext.p, method="fdr_bh")[1]
ext["n_tests_family"] = len(ext)
internal = pd.DataFrame(test_block(INTERNAL, False, "internal"))
internal["q"] = multipletests(internal.p, method="fdr_bh")[1]
internal["n_tests_family"] = len(internal)
tests = pd.concat([ext, internal]).sort_values(["block", "effect"], ascending=[True, False])
tests.to_csv(RES / "tests_global.csv", index=False)
print(tests[["variable", "block", "test", "effect", "q", "top_cluster", "top_std_diff"]].round(4).to_string(index=False))

# pairwise, Holm inside each variable
pair_rows = []
for v in EXT_CONT + EXT_BIN + INTERNAL:
    s = d[[v, "cluster"]].dropna()
    binary = v in EXT_BIN
    ps, pairs = [], []
    for a in range(K):
        for b in range(a + 1, K):
            x, y = s[v][s.cluster == a], s[v][s.cluster == b]
            if binary:
                tab = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
                p = stats.fisher_exact(tab)[1]
            else:
                p = stats.mannwhitneyu(x, y, alternative="two-sided")[1]
            ps.append(p)
            pairs.append((a, b))
    adj = multipletests(ps, method="holm")[1]
    for (a, b), p, pa in zip(pairs, ps, adj):
        pair_rows.append(dict(variable=v, a=a, b=b, p=p, p_holm=pa))
pd.DataFrame(pair_rows).to_csv(RES / "tests_pairwise.csv", index=False)

# ------------------------------------------------------ descriptive tables
desc = d.groupby("cluster")[INTERNAL + EXT_CONT + EXT_BIN +
                            ["lab_diabetes", "undiagnosed_diabetes", "alt_high"]].mean().T
desc.columns = NAMES
desc["cohort"] = d[desc.index].mean()
desc.to_csv(RES / "cluster_means.csv")
cent_z = pd.DataFrame(Z, columns=INTERNAL).groupby(d.cluster.values).mean()
cent_z.index = NAMES
cent_z.to_csv(RES / "centroids_z.csv")
sizes = d.cluster.value_counts().sort_index()
sizes.index = NAMES
sizes.to_csv(RES / "cluster_sizes.csv")


def prop_ci(x):
    """Wilson 95% interval for a proportion."""
    x = x.dropna()
    n, k = len(x), x.sum()
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    z = 1.96
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h, n


ci_rows = []
for v in EXT_BIN + ["lab_diabetes", "undiagnosed_diabetes", "alt_high"]:
    for k in range(K):
        p, lo, hi, n = prop_ci(d[v][d.cluster == k])
        ci_rows.append(dict(variable=v, cluster=k, p=p, lo=lo, hi=hi, n=n))
pd.DataFrame(ci_rows).to_csv(RES / "proportions_ci.csv", index=False)

# --------------------------------------------- the hidden findings, in numbers
hid = {}
dm = d[d.lab_diabetes == 1]
hid["lab_diabetes_n"] = int(len(dm))
hid["lab_diabetes_not_told"] = float((dm.dx_diabetes == 0).mean())
w = dm.WTSAF2YR
hid["lab_diabetes_not_told_weighted"] = float(((dm.dx_diabetes == 0) * w).sum() / w.sum())
hid["undiagnosed_by_cluster"] = {NAMES[k]: dict(
    n_lab=int((d.lab_diabetes[d.cluster == k] == 1).sum()),
    n_undx=int(d.undiagnosed_diabetes[d.cluster == k].sum()))
    for k in range(K)}
hp = d[d.cluster == 2]
hid["hepatic_alt_high"] = float(hp.alt_high.mean())
hid["hepatic_told_liver"] = float(hp.dx_liver.mean())
hid["hepatic_alt_high_told_liver"] = float(hp.dx_liver[hp.alt_high == 1].mean())
hid["meanshift_modes"] = int(len(ms_sizes))
hid["meanshift_largest_share"] = float(ms_sizes.max() / ms_sizes.sum())
hid["silhouette_k4"] = float(kdf.silhouette[kdf.k == 4].iloc[0])
hid["sizes"] = {n: int(s) for n, s in sizes.items()}
pir = d.dropna(subset=["income_poverty_ratio"])
hid["below_130pct_poverty_by_cluster"] = {NAMES[k]: float((pir.income_poverty_ratio[pir.cluster == k] < 1.3).mean())
                                          for k in range(K)}
hid["n"] = int(len(d))

# ----------------------------------------------------------- surrogate models
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
y = d.cluster.values
cands = {
    "Majority class": None,
    "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED),
    "Logistic (L1)": LogisticRegression(penalty="l1", C=0.5, solver="saga", max_iter=5000, class_weight="balanced"),
    "Random forest": RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
}
cv_rows, perm_rows = [], []
for fold, (tr, te) in enumerate(skf.split(Z, y)):
    for name, m in cands.items():
        if m is None:
            pred = np.full(len(te), np.bincount(y[tr]).argmax())
        else:
            m.fit(Z[tr], y[tr])
            pred = m.predict(Z[te])
        cv_rows.append(dict(model=name, fold=fold, accuracy=(pred == y[te]).mean(),
                            balanced_accuracy=balanced_accuracy_score(y[te], pred)))
        if name == "Random forest":
            pi = permutation_importance(m, Z[te], y[te], n_repeats=10, random_state=SEED,
                                        scoring="balanced_accuracy", n_jobs=-1)
            for j, v in enumerate(INTERNAL):
                for r in pi.importances[j]:
                    perm_rows.append(dict(variable=v, fold=fold, importance=r))
cv = pd.DataFrame(cv_rows)
cv.to_csv(RES / "surrogate_cv.csv", index=False)
perm = pd.DataFrame(perm_rows)
perm.to_csv(RES / "permutation_importance.csv", index=False)
print(cv.groupby("model")[["accuracy", "balanced_accuracy"]].agg(["mean", "std"]).round(3))

# ------------------- clinical description: multinomial L1 on the EXTERNAL block
desc_vars = ["age", "female", "bmi", "sbp", "income_poverty_ratio", "college", "smoker",
             "dx_hypertension", "dx_cvd", "dx_liver", "food_insecure", "uninsured", "active"]
E = d[desc_vars + ["cluster"]].dropna()
XE = StandardScaler().fit_transform(E[desc_vars])
yE = E.cluster.values
best = None
for C in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    sc = []
    for tr, te in skf.split(XE, yE):
        m = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=5000, class_weight="balanced")
        m.fit(XE[tr], yE[tr])
        sc.append(balanced_accuracy_score(yE[te], m.predict(XE[te])))
    if best is None or np.mean(sc) > best[1]:
        best = (C, np.mean(sc))
print("external-block L1 multinomial: C =", best[0], "balanced acc =", round(best[1], 3))


def ref_contrast(model):
    co = model.coef_
    return co - co[0]      # contrast every cluster with the Low-risk reference


m = LogisticRegression(penalty="l1", C=best[0], solver="saga", max_iter=5000, class_weight="balanced").fit(XE, yE)
point = ref_contrast(m)
boots = []
for b in range(200):
    idx = rng.choice(len(XE), len(XE), replace=True)
    mb = LogisticRegression(penalty="l1", C=best[0], solver="saga", max_iter=3000,
                            class_weight="balanced").fit(XE[idx], yE[idx])
    boots.append(ref_contrast(mb))
boots = np.array(boots)
or_rows = []
for k in range(1, K):
    for j, v in enumerate(desc_vars):
        or_rows.append(dict(cluster=k, variable=v, log_or=point[k, j],
                            lo=np.percentile(boots[:, k, j], 2.5),
                            hi=np.percentile(boots[:, k, j], 97.5)))
pd.DataFrame(or_rows).to_csv(RES / "odds_ratios_external.csv", index=False)
hid["external_model_balanced_acc"] = float(best[1])
hid["external_model_C"] = best[0]
hid["external_model_n"] = int(len(E))

# ------------------------------------------------------------ one patient, SHAP
import shap
rf = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced", n_jobs=-1,
                            random_state=SEED).fit(Z, y)
cand = d.index[(d.cluster == 3) & (d.undiagnosed_diabetes == 1)]
pid = int(cand[np.argsort(d.loc[cand, "hba1c"].values)[len(cand) // 2]])   # a typical one, not the extreme
expl = shap.TreeExplainer(rf)
sv = expl.shap_values(Z[[pid]])
sv = np.array(sv)
vals = sv[0, :, 3] if sv.ndim == 3 else sv[3][0]
base = expl.expected_value[3]
patient = dict(index=pid, seqn=int(d.SEQN[pid]), base=float(base),
               prob=float(rf.predict_proba(Z[[pid]])[0, 3]),
               shap={v: float(s) for v, s in zip(INTERNAL, vals)},
               raw={v: float(d.loc[pid, v]) for v in INTERNAL},
               age=float(d.age[pid]), female=int(d.female[pid]), bmi=float(d.bmi[pid]),
               told_diabetes=int(d.dx_diabetes[pid]))
json.dump(patient, open(RES / "shap_patient.json", "w"), indent=2)

json.dump(hid, open(RES / "headline_numbers.json", "w"), indent=2)
print(json.dumps(hid, indent=2))
