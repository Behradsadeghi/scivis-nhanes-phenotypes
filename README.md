# What a blood test knows that its patient doesn't

**Latent metabolic phenotypes in 2,290 US adults (NHANES 2017–2018)**
Scientific Visualization · MSc Computer Science · Università degli Studi di Milano · 2026–27
Behrad Sadeghi · Mohammad Ali Noroozshad · Saba Shafiei Sabet

**Presentation:** [`presentation/index.html`](presentation/index.html) — a single self-contained file (25 slides + 4 appendix). A static backup is in [`presentation/slides.pdf`](presentation/slides.pdf).
With GitHub Pages enabled on this repo, it is served at `https://<user>.github.io/<repo>/`.

| Key | Action |
|---|---|
| ← → / Space | previous / next slide (click or swipe also work) |
| **P** | presenter view in a second window: current + next slide, speaker notes, timer — stays in sync with the audience window |
| **N** | speaker notes under the slide |
| **F** | full screen |

Slides 16 and 17 are interactive (hover a dot for that adult's values; switch the view on slide 17).

---

## The question

Fourteen routine blood and urine biomarkers are measured on every adult in the NHANES fasting subsample. **Do they contain groups of people that matter beyond the blood** — groups that differ on diagnoses, blood pressure, self-rated health and money, none of which the clustering is allowed to see?

## The one rule

A variable either **builds** the clusters or **judges** them — never both (Lecture 6b, slide 3).

| Block | Variables | Role |
|---|---|---|
| Internal (14) | fasting glucose, HbA1c, fasting insulin, triglycerides, HDL, LDL, hs-CRP, ALT, GGT, uric acid, creatinine, albumin, urine albumin/creatinine, white blood cells | the only input to k-means |
| External (22) | age, sex, BMI, waist, blood pressure, doctor-told diabetes / hypertension / high cholesterol / heart disease or stroke / liver condition, diabetes medication, income-to-poverty ratio, education, insurance, food security, usual place of care, smoking (cotinine), physical activity, self-rated health | held out; used only to validate and describe |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `src/download_data.py` | downloads the 20 public NHANES 2017–18 `.xpt` files from the CDC |
| 1 | `src/prepare.py` | merges files; adults ≥ 20 in the fasting subsample, not pregnant, complete on the 14 biomarkers (n = 2,290); recodes the external block |
| 2 | `src/analysis.py` | log + z-score; k-means (k = 2…8, 100 restarts) with silhouette, 30-bootstrap stability (ARI), agreement with a Gaussian mixture and Ward linkage; mean-shift; PCA and UMAP (display only); Kruskal–Wallis / χ² with Benjamini–Hochberg; pairwise Mann–Whitney / Fisher with Holm; effect sizes (ε², Cramér's V, standardised differences); surrogate models with 5-fold stratified CV vs. a majority baseline; permutation importance on held-out folds; multinomial L1 logistic on the external block with bootstrap CIs; TreeSHAP for one patient |
| 3 | `src/figures.py` | draws every figure from `results/` (no statistics computed here) |
| 4 | `src/build_deck.py` | inlines the SVG figures and point data into `presentation/template.html` → `presentation/index.html` |
| 5 | `src/export_pdf.py` | optional: one PDF page per slide, animations in their final state |

Reproduce everything:

```bash
pip install -r requirements.txt
./run_all.sh
```

## Main results

* **A hypothesis checked on the raw data first** — from domain knowledge, HbA1c should rise with age. It does (Spearman ρ 0.48 [0.45, 0.51]; 0.43 without people on diabetes medication), and the raw data already shows a gap between diabetes in the blood and diabetes diagnosed, growing with age. One variable shows that the gap exists; the clustering later shows where it is.
* **Four stable phenotypes** — Low-risk (810), Inflammatory (581), Hepatic (660), Dysglycaemic (239). Bootstrap ARI 0.89 at k = 4 vs 0.77 at k = 5. At k = 3 Inflammatory and Hepatic merge; at k = 5 an older renal group splits off.
* **The data is a continuum** (one density mode holds 95% of people, silhouette 0.11): the clusters are regions, not islands — said explicitly.
* **They differ on 21 of 22 variables they never saw** (BH-adjusted), with large effects on diagnosis, BMI, sex and age, and small but real ones on income and food insecurity.
* **31% of lab-defined diabetes is undiagnosed** (143 of 454; 31% survey-weighted), and **93 of the 143 sit in the Inflammatory and Hepatic clusters**, not in the Dysglycaemic one.
* **39% of the Hepatic cluster have a raised ALT; 8% were ever told about their liver.**
* The clusters have a consistent signature (held-out balanced accuracy 0.98, baseline 0.25). CRP, HbA1c, glucose and ALT are both different and needed; LDL barely matters.

## Design choices (and the course material behind them)

* **One object through the whole talk** — the same 2,290 dots (one per adult) open each of the five parts and change state with the argument: a grey PCA cloud (Look), coloured UMAP regions (Cluster), named regions (Describe), the 454 lab-defined diabetics filled/hollow (Validate), one patient (Explain). The motion is the data moving, never decoration; `prefers-reduced-motion` gets the final states directly.
* Slide-design practice drawn from widely used community skills — Anthropic `frontend-design`, `frontend-slides` (fixed 16:9 stage, presenter-first), `impeccable` (one authored motion moment, browser surfaces themed), Emil Kowalski's design-engineering notes (fast ease-out, no slow keyboard-driven transitions), `taste-skill` (two or three type sizes do the work) — always subordinate to the course rules below.

* **One scale for every variable** — the validation slide sets each phenotype against the Low-risk group (Low-risk = 1, ratio of shares or of medians, bootstrap intervals), following Few's advice to divide by the reference rather than draw small multiples on different scales. The per-variable distributions stay in the appendix.
* **One colour per cluster, everywhere** — the four hues are the only colours in the deck; they were validated for colour-vision deficiency (all-pairs ΔE ≥ 8.6). Grey is "not yet clustered".
* **Titles state the finding** (presentation task, Keim via Nagel); each figure answers one question.
* **Show the sample, not only its averages** — patient-level heatmap, notched boxes over the raw points, 95% intervals on every proportion, a unit chart where one dot is one person.
* **No stars**; q-values and effect sizes printed instead. Diverging colour only where zero means something (the cohort mean).
* **Log axes labelled in original units**; clinical thresholds marked on the raw distributions.
* **UMAP for display only**, with the caveat on the slide; clustering runs in the full 14-D space.
* Sources: Tufte, *The Visual Display of Quantitative Information*; Few, *Show Me the Numbers*; Ware, *Information Visualization: Perception for Design*; course Lectures 1–7.

## Limitations

Cross-sectional data (no follow-up outcome); self-reported diagnoses; unweighted clustering, so cluster shares are not US prevalence; CRP is hormonally higher in women, which partly explains the sex balance of the Inflammatory group.

## Repository layout

```
src/            analysis code (Python)
data/processed/ analytic dataset + cohort flow (raw .xpt files are downloaded, not committed)
results/        every number the figures and slides use
figures/        every figure, SVG + PNG
presentation/   template.html (source) and index.html (built, self-contained)
```

Data: CDC / NCHS, National Health and Nutrition Examination Survey 2017–2018, public-use files.
