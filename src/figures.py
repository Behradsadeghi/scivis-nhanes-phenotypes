"""
Step 3 - Draw every figure from results/ (no statistics are computed here).

Each function draws one figure for one claim. The claim is the slide title;
the figure itself carries no title, so that the two never disagree.
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import gaussian_kde
from style import (setup, save, grid, INK, INK2, MUTED, HAIR, CLOUD, NAMES, CL, NEG, MID, POS,
                   LABELS, UNITS, ROOT)

setup()
RES = ROOT / "results"
d = pd.read_csv(RES / "patients.csv")
Z = pd.read_csv(RES / "z_internal.csv")
INTERNAL = list(Z.columns)
LOGGED = ["glucose", "hba1c", "insulin", "triglycerides", "hdl", "crp", "alt",
          "ggt", "creatinine", "acr", "wbc"]
DIV = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
H = json.load(open(RES / "headline_numbers.json"))

# clinical reference lines for the raw-data view (what a clinician would mark)
REF = {"glucose": (126, "126 diabetes"), "hba1c": (6.5, "6.5 diabetes"), "triglycerides": (150, "150"),
       "hdl": (40, "40"), "ldl": (130, "130"), "crp": (3, "3 high risk"), "alt": (33, "33"),
       "ggt": (60, "60"), "uric_acid": (7, "7"), "creatinine": (1.3, "1.3"), "albumin": (3.5, "3.5"),
       "acr": (30, "30 albuminuria"), "wbc": (11, "11"), "insulin": (25, "25")}


def nice_log_ticks(lo, hi):
    best = None
    for seq in ([1, 10, 100, 1000, 10000, 0.1], [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000],
                [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]):
        t = sorted(c for c in seq if lo <= c <= hi)
        while len(t) > 3:
            t = t[::2]
        if len(t) >= 3:
            return t
        if len(t) == 2 and best is None:
            best = t
    return best or [round(lo, 1), round(hi, 1)]


def head(ax, title, sub=None):
    """Panel heading: title + one quiet line of how-to-read, never overlapping."""
    ax.text(0, 1.13 if sub else 1.04, title, transform=ax.transAxes, fontsize=13, weight=600,
            va="bottom", color=INK)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")


SHORT = {"acr": "Urine ACR", "wbc": "White cells", "hdl": "HDL", "ldl": "LDL",
         "glucose": "Glucose", "insulin": "Insulin", "uric_acid": "Uric acid"}


# --------------------------------------------------------------- 1 raw data
def f_distributions():
    order = ["glucose", "hba1c", "insulin", "triglycerides", "hdl", "ldl", "crp",
             "alt", "ggt", "uric_acid", "creatinine", "albumin", "acr", "wbc"]
    fig, axes = plt.subplots(2, 7, figsize=(15, 5.6))
    for ax, v in zip(axes.flat, order):
        x = d[v].values
        logv = v in LOGGED
        xx = np.log10(x) if logv else x
        lo, hi = np.percentile(xx, [0.2, 99.8])
        bins = np.linspace(lo, hi, 34)
        cnt, _, _ = ax.hist(np.clip(xx, lo, hi), bins=bins, color=CLOUD, edgecolor="white", linewidth=0.6)
        top = cnt.max() * 1.32
        ax.set_ylim(0, top)
        med = np.median(xx)
        ax.plot([med, med], [0, cnt.max() * 1.05], color=INK, lw=1.3)
        if v in REF:
            r, lab = REF[v]
            rr = np.log10(r) if logv else r
            if lo < rr < hi:
                ax.plot([rr, rr], [0, top * 0.86], color=POS, lw=1.0)
                ax.text(rr, top * 0.88, lab, color=POS, fontsize=10, va="bottom",
                        ha="left" if rr < (lo + hi) / 2 else "right")
        if logv:
            t = nice_log_ticks(10 ** lo, 10 ** hi)
            ax.set_xticks(np.log10(t))
            ax.set_xticklabels([f"{c:g}" for c in t])
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(0, 1.2, SHORT.get(v, LABELS[v]), transform=ax.transAxes, fontsize=12.5,
                weight=600, va="bottom", color=INK)
        ax.text(0, 1.07, f"{UNITS[v]}{' · log scale' if logv else ''}", transform=ax.transAxes,
                fontsize=10, color=MUTED, va="bottom", ha="left")
        ax.tick_params(labelsize=10.5)
    fig.subplots_adjust(hspace=0.85, wspace=0.2)
    save(fig, "f02_distributions")


def f_correlation():
    S = d[INTERNAL].corr(method="spearman")
    order = leaves_list(linkage(1 - S.abs().values[np.triu_indices(len(S), 1)], "average"))
    S = S.iloc[order, order]
    n = len(S)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    M = S.values.copy()
    mask = np.triu(np.ones_like(M, bool), 0)
    Mm = np.ma.array(M, mask=mask)
    ax.imshow(Mm, cmap=DIV, norm=TwoSlopeNorm(0, -0.8, 0.8))
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                continue
            val = M[i, j]
            ax.text(j, i, f"{val:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=9.5, color="white" if abs(val) > 0.45 else INK2)
    labs = [LABELS[c] for c in S.columns]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labs, rotation=50, ha="right", fontsize=10.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # white gaps between cells
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=1.5)
        ax.axvline(k - 0.5, color="white", lw=1.5)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    ax.set_xticks(range(n - 1))
    ax.set_xticklabels(labs[:-1], rotation=50, ha="right", fontsize=10.5)
    cax = fig.add_axes([0.6, 0.8, 0.25, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -0.8, 0.8), cmap=DIV), cax=cax,
                      orientation="horizontal", ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10, length=0)
    cb.ax.set_title("Spearman ρ", fontsize=10.5, color=INK2, loc="left")
    save(fig, "f03_correlation")
    return list(S.columns)


def f_cloud():
    emb = pd.read_csv(RES / "embedding.csv")
    scree = pd.read_csv(RES / "pca_scree.csv")
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_axes([0.0, 0.0, 0.5, 1])
    ax.scatter(emb.pc1, emb.pc2, s=6, color=CLOUD, alpha=0.55, linewidths=0)
    kde = gaussian_kde(np.vstack([emb.pc1, emb.pc2]))
    gx, gy = np.mgrid[-6:8:160j, -6:6:160j]
    gz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, gz, levels=6, colors=INK2, linewidths=0.7)
    ax.set_xlim(-5.5, 7.5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(f"PC1 · {scree.explained[0]:.0%} of variance")
    ax.set_ylabel(f"PC2 · {scree.explained[1]:.0%}")
    ax.set_aspect("equal")
    ax.text(0.02, 0.98, "2,290 adults · density contours", transform=ax.transAxes, va="top",
            fontsize=11, color=MUTED)
    # scree
    ax2 = fig.add_axes([0.6, 0.12, 0.38, 0.78])
    ax2.bar(scree.pc, scree.explained, color=CLOUD, width=0.72)
    ax2.plot(scree.pc, scree.explained.cumsum(), color=INK, lw=1.6, marker="o", ms=4)
    ax2.set_xticks(scree.pc)
    ax2.tick_params(labelsize=10.5)
    grid(ax2)
    ax2.set_ylim(0, 1.02)
    ax2.set_yticks([0, .25, .5, .75, 1])
    ax2.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("Principal component")
    k80 = int(np.argmax(scree.explained.cumsum().values >= 0.8)) + 1
    ax2.annotate(f"{k80} components for 80%", (k80, scree.explained.cumsum()[k80 - 1]),
                 (k80 + 1.2, 0.6), fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax2.text(2.5, 0.15, "variance per component", fontsize=10.5, color=MUTED)
    ax2.text(3.2, 0.98, "cumulative", fontsize=10.5, color=INK, va="top")
    save(fig, "f04_cloud")


# --------------------------------------------------------------- 2 clustering
def f_ksel():
    k = pd.read_csv(RES / "k_selection.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    specs = [
        ("Separation", "Silhouette (higher = better separated)", [("silhouette", INK, "k-means")]),
        ("Stability", "ARI vs. resampled data (30 bootstraps)", [("boot_mean", INK, None)]),
        ("Agreement", "ARI with methods that fail differently", [("ari_gmm", INK, "Gaussian mixture"),
                                                                ("ari_ward", INK2, "Ward linkage")]),
    ]
    for ax, (t, sub, series) in zip(axes, specs):
        ax.axvspan(3.6, 4.4, color=HAIR, alpha=0.6, lw=0)
        for col, c, lab in series:
            ls = "-" if c == INK else (0, (1, 0))
            ax.plot(k.k, k[col], color=c, lw=2, marker="o", ms=6, mec="white", mew=1.2,
                    alpha=1 if c == INK else 0.65)
            if lab and len(series) > 1:
                ax.text(8.15, k[col].iloc[-1], lab, color=c, fontsize=10.5, va="center")
        if t == "Stability":
            ax.fill_between(k.k, k.boot_lo, k.boot_hi, color=CLOUD, alpha=0.35, lw=0)
            ax.text(8.15, k.boot_mean.iloc[-1], "mean", fontsize=10.5, va="center", color=INK)
            ax.text(8.15, k.boot_lo.iloc[-1], "5th pct", fontsize=10.5, va="center", color=MUTED)
            ax.set_ylim(0.4, 1)
        if t == "Separation":
            ax.set_ylim(0, 0.2)
        if t == "Agreement":
            ax.set_ylim(0.2, 0.7)
        head(ax, t, sub)
        ax.set_xticks(range(2, 9))
        ax.set_xlabel("number of clusters k")
        grid(ax)
    axes[1].annotate(f"k = 4: {k.boot_mean[k.k == 4].iloc[0]:.2f}", (4, k.boot_mean[k.k == 4].iloc[0]),
                     (4.6, 0.97), fontsize=11, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[1].annotate(f"k = 5: {k.boot_mean[k.k == 5].iloc[0]:.2f}", (5, k.boot_mean[k.k == 5].iloc[0]),
                     (5.9, 0.86), fontsize=11, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.subplots_adjust(wspace=0.42)
    save(fig, "f05_k_selection")


def f_alluvial():
    cols = [("k3", {1: ("Low-risk", CL[0]), 0: ("Insulin-resistant", MUTED), 2: ("Dysglycaemic", CL[3])}, [1, 0, 2]),
            ("cluster", {i: (NAMES[i], CL[i]) for i in range(4)}, [0, 1, 2, 3]),
            ("k5", {2: ("Low-risk", CL[0]), 3: ("Inflammatory", CL[1]), 1: ("Hepatic", CL[2]),
                    4: ("Dysglycaemic", CL[3]), 0: ("Renal, older", MUTED)}, [2, 3, 1, 4, 0])]
    N = len(d)
    gap = 110
    xs = [0, 1, 2]
    w = 0.06
    fig, ax = plt.subplots(figsize=(13, 6))
    pos = []
    for (col, names, order), x in zip(cols, xs):
        y = 0
        p = {}
        for g in order:
            n = (d[col] == g).sum()
            p[g] = [y, y + n, y]   # top, bottom, cursor (outgoing)
            nm, c = names[g]
            ax.add_patch(Rectangle((x - w / 2, y), w, n, color=c, lw=0))
            lab = f"{nm}  {n:,}"
            if x == 0:
                ax.text(x - w, y + n / 2, lab, ha="right", va="center", fontsize=12)
            elif x == 2:
                ax.text(x + w, y + n / 2, lab, ha="left", va="center", fontsize=12)
            else:
                ax.text(x, y - 22, lab, ha="center", va="bottom", fontsize=12, color=INK, weight=600)
            y += n + gap
        pos.append(p)
    # flows
    for li, (left, right) in enumerate([(0, 1), (1, 2)]):
        lc, ln, lo = cols[left]
        rc, rn, ro = cols[right]
        pl, pr = pos[left], pos[right]
        incur = {g: pr[g][0] for g in ro}
        for a in lo:
            for b in ro:
                n = ((d[lc] == a) & (d[rc] == b)).sum()
                if n == 0:
                    continue
                y0 = pl[a][2]
                y1 = incur[b]
                pl[a][2] += n
                incur[b] += n
                x0, x1 = xs[left] + w / 2, xs[right] - w / 2
                xm = (x0 + x1) / 2
                verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + n), (xm, y1 + n),
                         (xm, y0 + n), (x0, y0 + n), (x0, y0)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                         MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                # colour of the flow = the k = 4 cluster it belongs to
                c = CL[b] if li == 0 else CL[a]
                ax.add_patch(PathPatch(MPath(verts, codes), facecolor=c, alpha=0.28 if n > 30 else 0.18, lw=0))
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(N + 4.2 * gap, -110)
    ax.axis("off")
    for x, t in zip(xs, ["k = 3", "k = 4  (chosen)", "k = 5"]):
        ax.text(x, N + 4.15 * gap, t, ha="center", va="top", fontsize=12.5, color=INK2,
                weight=600 if "chosen" in t else 400)
    save(fig, "f06_alluvial")


def f_umap():
    emb = pd.read_csv(RES / "embedding.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    idx = np.random.default_rng(1).permutation(len(emb))
    e = emb.iloc[idx]
    ax.scatter(e.u1, e.u2, s=9, c=[CL[c] for c in e.cluster], alpha=0.75, linewidths=0.3, edgecolors="white")
    for k in range(4):
        s = emb[emb.cluster == k]
        mx, my = s.u1.median(), s.u2.median()
        ax.text(mx, my, f"{NAMES[k]}\n{len(s):,}", ha="center", va="center", fontsize=13, weight=600,
                color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    save(fig, "f07_umap")


# --------------------------------------------------------------- 3 describe
def column_order():
    t = pd.read_csv(RES / "tests_global.csv")
    t = t[t.block == "internal"].sort_values("effect", ascending=False)
    return list(t.variable), dict(zip(t.variable, t.effect)), dict(zip(t.variable, t.q))


def f_centroids():
    cz = pd.read_csv(RES / "centroids_z.csv", index_col=0)
    sizes = pd.read_csv(RES / "cluster_sizes.csv", index_col=0).iloc[:, 0]
    order, eff, q = column_order()
    M = cz[order].values
    fig, ax = plt.subplots(figsize=(15, 4.6))
    norm = TwoSlopeNorm(0, -2.5, 2.5)
    ax.imshow(M, cmap=DIV, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=12,
                    color="white" if abs(v) > 1.2 else INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[c] for c in order], rotation=35, ha="right", fontsize=11.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels([])
    for i, nm in enumerate(NAMES):
        ax.add_patch(Rectangle((-1.05, i - 0.42), 0.18, 0.84, color=CL[i], clip_on=False, lw=0))
        ax.text(-1.15, i - 0.08, nm, ha="right", va="center", fontsize=13, weight=600)
        ax.text(-1.15, i + 0.24, f"n = {sizes[nm]:,}", ha="right", va="center", fontsize=10.5, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for k in range(5):
        ax.axhline(k - 0.5, color="white", lw=2.5)
    for k in range(len(order) + 1):
        ax.axvline(k - 0.5, color="white", lw=2.5)
    for j, c in enumerate(order):
        ax.text(j, -0.62, f"ε² {eff[c]:.2f}", ha="center", va="bottom", fontsize=9.5, color=MUTED)
    ax.text(-1.15, -0.62, "separation (KW ε²) →", ha="right", va="bottom", fontsize=9.5, color=MUTED)
    save(fig, "f08_centroids")


def f_patients():
    order, eff, q = column_order()
    zz = Z[order].copy()
    zz["cluster"] = d.cluster
    emb = pd.read_csv(RES / "embedding.csv")
    zz["pc1"] = emb.pc1
    zz = zz.sort_values(["cluster", "pc1"])
    M = zz[order].values
    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_axes([0.08, 0.15, 0.82, 0.8])
    ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -2.5, 2.5), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[c] for c in order], rotation=55, ha="right", fontsize=10.5)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    y = 0
    for k in range(4):
        n = (zz.cluster == k).sum()
        ax.add_patch(Rectangle((-1.4, y - 0.5), 0.55, n, color=CL[k], clip_on=False, lw=0))
        ax.text(-1.6, y + n / 2, f"{NAMES[k]}\n{n:,}", ha="right", va="center", fontsize=11.5)
        if k:
            ax.axhline(y - 0.5, color="white", lw=3)
        y += n
    ax.set_xlim(-0.5, len(order) - 0.5)
    cax = fig.add_axes([0.92, 0.55, 0.018, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -2.5, 2.5), cmap=DIV), cax=cax,
                      ticks=[-2, -1, 0, 1, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10, length=0)
    cb.ax.set_title("SD from\ncohort mean", fontsize=10, color=INK2, loc="left")
    save(fig, "f09_patients")


# --------------------------------------------------------------- 4 validate
def repel(items, min_gap):
    """items: list of [y, ...]; nudge y so neighbours are at least min_gap apart (1-D)."""
    items.sort(key=lambda r: r[0])
    for _ in range(200):
        moved = False
        for a, b in zip(items, items[1:]):
            if b[0] - a[0] < min_gap:
                m = (b[0] - a[0] - min_gap) / 2
                a[0] += m
                b[0] -= m
                moved = True
        if not moved:
            break
    return items


def f_volcano():
    t = pd.read_csv(RES / "tests_global.csv")
    t = t[t.block == "external"].copy()
    t["nlq"] = -np.log10(t.q.clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_yscale("log")
    ax.axvspan(-0.2, 0.2, color=HAIR, alpha=0.55, lw=0)
    ax.axhline(-np.log10(0.05), color=MUTED, lw=0.9)
    ax.text(-1.22, -np.log10(0.05) * 1.08, "q = 0.05", fontsize=10.5, color=MUTED, ha="left", va="bottom")
    ax.text(0, 300, "too small\nto matter", ha="center", va="top", fontsize=10.5, color=MUTED)
    for _, r in t.iterrows():
        c = CL[int(r.top_cluster)]
        mk = "o" if r.type == "continuous" else "s"
        ax.scatter(r.top_std_diff, r.nlq, s=70, marker=mk, color=c, edgecolors="white", linewidths=1.2, zorder=3)
    ax.set_xlim(-1.25, 2.6)
    ax.set_ylim(0.2, 400)
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    sides = {"L": [], "R": []}
    for _, r in t.iterrows():
        xa, ya = inv.transform(ax.transData.transform((r.top_std_diff, r.nlq)))
        side = "L" if r.top_std_diff < -0.2 or r.variable in ("uninsured",) and False else "R"
        sides[side].append([ya, xa, r])
    for side, items in sides.items():
        # group labels whose points are horizontally close, then repel vertically
        repel(items, 0.045)
        for ya, xa, r in items:
            px, py = r.top_std_diff, r.nlq
            tx = xa + (0.018 if side == "R" else -0.018)
            ax.annotate(LABELS[r.variable], (px, py), xytext=(tx, ya), textcoords="axes fraction",
                        fontsize=10.5, ha="left" if side == "R" else "right", va="center", color=INK,
                        arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.7, shrinkA=0, shrinkB=4))
    ax.set_xlabel("Standardised difference: the cluster that departs most vs. all others (SD)")
    ax.set_ylabel("−log₁₀ q  (BH-adjusted across 22 tests · log scale)")
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.set_yticklabels(["1", "3", "10", "30", "100", "300"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    grid(ax, "both")
    for i, nm in enumerate(NAMES):
        ax.scatter([], [], color=CL[i], s=60, label=nm)
    ax.scatter([], [], marker="o", color="white", edgecolors=INK2, s=50, label="continuous · Kruskal–Wallis")
    ax.scatter([], [], marker="s", color="white", edgecolors=INK2, s=50, label="binary · χ²")
    ax.legend(loc="lower right", fontsize=10.5, ncol=2, handletextpad=0.3, columnspacing=1.2)
    save(fig, "f10_volcano_external")


def f_external_multiples():
    pc = pd.read_csv(RES / "proportions_ci.csv")
    t = pd.read_csv(RES / "tests_global.csv").set_index("variable")
    binv = ["dx_diabetes", "dx_hypertension", "dx_cvd", "fair_poor_health", "dx_liver", "female",
            "food_insecure", "college"]
    contv = ["age", "bmi", "sbp", "income_poverty_ratio"]
    fig, axes = plt.subplots(2, 6, figsize=(16, 7.2))
    flat = axes.flat
    for v in contv:
        ax = next(flat)
        data = [d[v][d.cluster == k].dropna() for k in range(4)]
        bp = ax.boxplot(data, notch=True, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED))
        for k, b in enumerate(bp["boxes"]):
            b.set_facecolor(CL[k])
            b.set_alpha(0.28)
            b.set_edgecolor(CL[k])
        rng = np.random.default_rng(k)
        for k, x in enumerate(data):
            xs = k + 1 + rng.uniform(-0.18, 0.18, len(x))
            ax.scatter(xs, x, s=1.5, color=CL[k], alpha=0.35, lw=0, zorder=0)
        ax.set_xticks([])
        unit = {"age": "years", "bmi": "kg/m²", "sbp": "mmHg", "income_poverty_ratio": "ratio, capped at 5"}[v]
        head(ax, LABELS[v], f"{unit} · q {t.q[v]:.0e}".replace("e-0", "e−").replace("e-", "e−"))
        grid(ax)
        ax.tick_params(labelsize=10)
    for v in binv:
        ax = next(flat)
        s = pc[pc.variable == v].sort_values("cluster")
        for _, r in s.iterrows():
            k = int(r.cluster)
            ax.plot([k, k], [r.lo, r.hi], color=CL[k], lw=2.2, solid_capstyle="round")
            ax.scatter(k, r.p, s=48, color=CL[k], edgecolors="white", linewidths=1.2, zorder=3)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, max(0.15, s.hi.max() * 1.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_xticks([])
        q = t.q[v]
        qs = "q < 1e−10" if q < 1e-10 else f"q {q:.0e}".replace("e-0", "e−").replace("e-", "e−")
        head(ax, {"fair_poor_health": "Fair / poor health", "dx_liver": "Told: liver",
                  "dx_hypertension": "Told: hypertension", "dx_cvd": "Heart or stroke"}.get(v, LABELS[v]),
             f"% · 95% CI · {qs}")
        grid(ax)
        ax.tick_params(labelsize=10)
    fig.subplots_adjust(hspace=0.5, wspace=0.38)
    # shared legend row
    handles = [plt.Line2D([], [], marker="o", ls="", color=CL[i], ms=8) for i in range(4)]
    fig.legend(handles, NAMES, loc="lower center", ncol=4, fontsize=11.5, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=1.6)
    save(fig, "f11_external_multiples")


def f_undiagnosed():
    fig, ax = plt.subplots(figsize=(15, 5.6))
    per_row = 12
    x0 = 0
    for k in range(4):
        s = d[(d.cluster == k) & (d.lab_diabetes == 1)].sort_values("dx_diabetes", ascending=False)
        n = len(s)
        told = (s.dx_diabetes == 1).values
        for i, tl in enumerate(told):
            cx = x0 + (i % per_row)
            cy = i // per_row
            if tl:
                ax.scatter(cx, cy, s=58, color=CL[k], lw=0)
            else:
                ax.scatter(cx, cy, s=58, facecolor="white", edgecolor=CL[k], lw=1.6)
        nund = int((~told).sum())
        ax.text(x0 - 0.4, -3.3, NAMES[k], fontsize=13.5, weight=600, va="bottom")
        ax.text(x0 - 0.4, -2.0, f"{n} meet lab criteria", fontsize=11, color=INK2, va="bottom")
        ax.text(x0 - 0.4, -0.85, f"{nund} never told · {nund / n:.0%}", fontsize=11, color=INK, va="bottom",
                weight=600)
        x0 += per_row + 3.2
    ax.set_xlim(-1, x0 - 2.5)
    ax.set_ylim(21, -4.6)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.scatter([], [], s=58, color=INK2, label="told by a doctor")
    ax.scatter([], [], s=58, facecolor="white", edgecolor=INK2, lw=1.6, label="never told")
    ax.legend(loc="lower left", fontsize=11.5, ncol=1, bbox_to_anchor=(0.0, 0.12))
    save(fig, "f12_undiagnosed")


def f_liver():
    pc = pd.read_csv(RES / "proportions_ci.csv")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k in range(4):
        a = pc[(pc.variable == "alt_high") & (pc.cluster == k)].iloc[0]
        b = pc[(pc.variable == "dx_liver") & (pc.cluster == k)].iloc[0]
        y = 3 - k
        ax.plot([b.p, a.p], [y, y], color=CL[k], lw=2.5, alpha=0.45, solid_capstyle="round")
        ax.plot([a.lo, a.hi], [y + 0.16, y + 0.16], color=CL[k], lw=1.6, solid_capstyle="round")
        ax.plot([b.lo, b.hi], [y - 0.16, y - 0.16], color=MUTED, lw=1.6, solid_capstyle="round")
        ax.scatter(a.p, y, s=110, color=CL[k], zorder=3, edgecolors="white", linewidths=1.5)
        ax.scatter(b.p, y, s=110, facecolor="white", edgecolor=CL[k], lw=2, zorder=3)
        ax.text(-0.01, y, NAMES[k], ha="right", va="center", fontsize=13, weight=600)
        ax.text(a.p + 0.012, y + 0.02, f"{a.p:.0%}", va="center", fontsize=11.5, color=INK)
        ax.text(b.p - 0.012, y + 0.02, f"{b.p:.0%}", va="center", ha="right", fontsize=11.5, color=INK2)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.6, 3.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    grid(ax, "x")
    ax.text(0.375, 3.55, "● ALT above the upper limit of normal (blood)", fontsize=11, color=INK, ha="left")
    ax.text(0.375, 3.25, "○ ever told of a liver condition (interview)", fontsize=11, color=INK2, ha="left")
    ax.set_xlabel("share of the cluster · bars are 95% Wilson intervals")
    save(fig, "f13_liver")


# --------------------------------------------------------------- 5 explain
def f_cv():
    cv = pd.read_csv(RES / "surrogate_cv.csv")
    order = ["Majority class", "Decision tree (depth 3)", "Random forest", "Logistic (L1)"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, m in enumerate(order):
        s = cv[cv.model == m].balanced_accuracy
        y = len(order) - 1 - i
        ax.scatter(s, [y] * len(s), s=34, color=MUTED if m == "Majority class" else INK2, alpha=0.7, lw=0)
        ax.plot([s.mean()] * 2, [y - 0.25, y + 0.25], color=INK, lw=2.4)
        ax.text(s.mean() + 0.015, y + 0.28, f"{s.mean():.2f}", fontsize=12, weight=600, va="bottom")
        ax.text(-0.02, y, m, ha="right", va="center", fontsize=12.5)
    ax.axvline(0.25, color=MUTED, lw=0.9)
    ax.text(0.255, 3.55, "chance for 4 balanced classes", fontsize=10.5, color=MUTED)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.6, 3.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("balanced accuracy on held-out patients · 5 stratified folds (dots)")
    grid(ax, "x")
    save(fig, "f14_surrogate_cv")


def f_uni_multi():
    t = pd.read_csv(RES / "tests_global.csv")
    t = t[t.block == "internal"].set_index("variable")
    pi = pd.read_csv(RES / "permutation_importance.csv")
    g = pi.groupby("variable").importance
    imp, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
    fig, ax = plt.subplots(figsize=(11, 6))
    for v in t.index:
        x, y = t.effect[v], imp[v]
        ax.plot([x, x], [lo[v], hi[v]], color=HAIR, lw=2, zorder=1)
        ax.scatter(x, y, s=70, color=INK, edgecolors="white", lw=1.2, zorder=3)
        off = {"hba1c": (-8, 4), "glucose": (8, 2), "insulin": (-8, -2), "alt": (8, -6), "crp": (8, 2),
               "hdl": (8, 6), "triglycerides": (8, -8), "ggt": (8, -2), "uric_acid": (8, 4), "wbc": (-8, -2),
               "albumin": (-8, 4), "acr": (6, 10), "creatinine": (-6, 10), "ldl": (8, 0)}[v]
        ax.annotate(SHORT.get(v, LABELS[v]), (x, y), xytext=off, textcoords="offset points", fontsize=11,
                    ha="left" if off[0] > 0 else "right", va="center")
    ax.set_xlabel("Univariate: how strongly the clusters differ on it (Kruskal–Wallis ε²)")
    ax.set_ylabel("Multivariate: drop in balanced accuracy\nwhen shuffled (random forest, held-out)")
    grid(ax, "both")
    ax.set_xlim(0, 0.42)
    ymax = hi.max() * 1.1
    ax.set_ylim(-0.005, ymax)
    xm, ym = 0.2, ymax * 0.33
    ax.axvline(xm, color=MUTED, lw=0.6)
    ax.axhline(ym, color=MUTED, lw=0.6)
    ax.text(0.415, ymax * 0.97, "differs and is needed", ha="right", va="top", fontsize=10.5, color=MUTED)
    ax.text(0.415, ym * 0.08, "differs, but a correlated\nneighbour carries it", ha="right", va="bottom",
            fontsize=10.5, color=MUTED)
    ax.text(0.005, ymax * 0.97, "needed only in combination", ha="left", va="top", fontsize=10.5, color=MUTED)
    save(fig, "f15_uni_vs_multi")


def f_forest():
    o = pd.read_csv(RES / "odds_ratios_external.csv")
    vars_ = ["age", "female", "bmi", "sbp", "income_poverty_ratio", "college", "smoker", "active",
             "dx_hypertension", "dx_cvd", "dx_liver", "food_insecure", "uninsured"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    for ax, k in zip(axes, [1, 2, 3]):
        s = o[o.cluster == k].set_index("variable").loc[vars_]
        ys = np.arange(len(vars_))[::-1]
        ax.axvline(1, color=MUTED, lw=0.9)
        for y, (v, r) in zip(ys, s.iterrows()):
            orr, lo, hi = np.exp(r.log_or), np.exp(r.lo), np.exp(r.hi)
            zero = abs(r.log_or) < 1e-6
            ax.plot([lo, hi], [y, y], color=CL[k] if not zero else HAIR, lw=2, solid_capstyle="round")
            if zero:
                ax.scatter(orr, y, s=40, facecolor="white", edgecolor=MUTED, lw=1.2, zorder=3)
            else:
                ax.scatter(orr, y, s=55, color=CL[k], edgecolors="white", lw=1.2, zorder=3)
                if abs(r.log_or) > np.log(1.35):
                    ax.text(hi * 1.06, y, f"{orr:.1f}×", va="center", fontsize=10.5, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(0.18, 6.5)
        ax.set_title(f"{NAMES[k]} vs. Low-risk")
        grid(ax, "x")
        ax.set_yticks(ys)
        ax.set_yticklabels([LABELS[v] for v in vars_], fontsize=11.5)
        ax.tick_params(axis="y", length=0)
        if k == 2:
            ax.set_xlabel("odds ratio per SD (continuous) or yes vs. no (binary) · 95% bootstrap CI, 200 resamples")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "f16_forest_external")


def f_shap():
    p = json.load(open(RES / "shap_patient.json"))
    sv = pd.Series(p["shap"]).sort_values(key=np.abs, ascending=True)
    sv = sv.iloc[-9:]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for y, (v, s) in enumerate(sv.items()):
        ax.barh(y, s, color=CL[3] if s > 0 else CLOUD, height=0.62)
        raw = p["raw"][v]
        lab = f"{LABELS[v]}  {raw:g} {UNITS[v]}"
        ax.text(-0.004 if s > 0 else 0.004, y, lab, ha="right" if s > 0 else "left", va="center", fontsize=11.5)
        ax.text(s + (0.004 if s > 0 else -0.004), y, f"{s:+.2f}".replace("-", "−"), va="center",
                ha="left" if s > 0 else "right", fontsize=10.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("SHAP contribution to P(Dysglycaemic) · random forest")
    lim = max(abs(sv.min()), sv.max()) * 1.25
    ax.set_xlim(-0.16, lim)
    grid(ax, "x")
    save(fig, "f17_shap_patient")


def export_deck_data():
    """Small JSON for the animated / interactive slides (no identifiers)."""
    emb = pd.read_csv(RES / "embedding.csv")
    pat = json.load(open(RES / "shap_patient.json"))

    def norm(v):
        return (v - v.min()) / (v.max() - v.min())
    u1, u2 = norm(emb.u1), norm(emb.u2)
    # PCA view: same scale on both axes, clipped to the central 99.5% so the cloud fills the frame
    lo = np.percentile(np.r_[emb.pc1, emb.pc2], 0.25)
    hi = np.percentile(np.r_[emb.pc1, emb.pc2], 99.75)
    p1 = ((emb.pc1.clip(lo, hi) - lo) / (hi - lo))
    p2 = ((emb.pc2.clip(lo, hi) - lo) / (hi - lo))
    rows = []
    for i in range(len(d)):
        rows.append([round(float(u1[i]), 4), round(float(u2[i]), 4), round(float(p1[i]), 4),
                     round(float(p2[i]), 4), int(d.cluster[i]), int(d.lab_diabetes[i]),
                     int(d.dx_diabetes[i] == 1), round(float(d.hba1c[i]), 1), int(d.glucose[i]),
                     int(d.alt[i]), round(float(d.crp[i]), 1), int(d.age[i]), int(d.female[i])])
    out = {"fields": ["ux", "uy", "px", "py", "cluster", "lab_diabetes", "told_diabetes", "hba1c",
                      "glucose", "alt", "crp", "age", "female"],
           "rows": rows, "names": NAMES, "colors": CL, "patient": int(pat["index"])}
    (ROOT / "results" / "deck_points.json").write_text(json.dumps(out, separators=(",", ":")))


ALL = [export_deck_data, f_distributions, f_correlation, f_cloud, f_ksel, f_alluvial, f_umap, f_centroids,
       f_patients, f_volcano, f_external_multiples, f_undiagnosed, f_liver, f_cv, f_uni_multi,
       f_forest, f_shap]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if not want or any(w in f.__name__ for w in want):
            f()
            print("drew", f.__name__)
