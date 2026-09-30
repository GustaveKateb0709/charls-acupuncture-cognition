# charls-acupuncture-cognition

Analysis code for:

> Acupuncture use and cognitive change among middle-aged and older Chinese adults with pain: a longitudinal analysis of three waves of the China Health and Retirement Longitudinal Study (2013–2018)

## What this repository contains

Python code implementing the pre-locked statistical analysis plan (SAP v1.2):

- `01_harmonize.py` — loads CHARLS waves 2013/2015/2018, harmonises variable names
  (including the `da042_w2_1s3` vs `da042_w2_1_s3` spelling difference), constructs the
  pain-reporter sample and the time-varying acupuncture exposure (structural missingness
  outside pain waves), counts word-recall scores, applies the published crosswalk of
  Wu et al. (2024) to the revised 2018 word-recall instrument, and builds the analysis panel.
- `02_analysis.py` — descriptive statistics, linear mixed model with a time-by-exposure
  interaction, GEE sensitivity model, inverse-probability-of-treatment weighting with
  balance diagnostics, E-value for unmeasured confounding, and pre-specified sensitivity analyses.
- `03_heterogeneity.py` — exploratory causal-forest estimates of individual-level effects (CATE) with SHAP summaries.
- `04_figures.py` — publication figures (TIFF 600 dpi + PDF + PNG).
- `config/rename_map.csv` — cross-wave variable-name mapping verified against the CHARLS codebooks.

## Data

The CHARLS data are publicly available from the official CHARLS website
(https://charls.pku.edu.cn) after online registration and signing a data use agreement.
**The data files are not included in this repository and must not be redistributed** —
place your downloaded waves in `data_raw/` as follows:

```
data_raw/
├── CHARLS2013_Dataset/Health_Status_and_Functioning.dta  (+ Demographic_Background.dta)
├── CHARLS2015r/Health_Status_and_Functioning.dta         (+ Demographic_Background.dta)
└── CHARLS2018r/Health_Status_and_Functioning.dta         (+ Cognition.dta, Demographic_Background.dta)
```

The word-recall crosswalk table (`eqt_crosswalk_table.xlsx`) is downloaded from the
official repository of Wu et al. (2024): https://github.com/Yingyan-Wu/charls_memory_equating
(place it in `data_raw/crosswalk/`).

## Requirements

Python 3.11+ with pandas, numpy, statsmodels, scikit-learn, matplotlib.
Optional: econml and shap for the exploratory causal-forest analysis.

## Citation

If you use this code, please cite the associated paper and the CHARLS dataset:

> Zhao Y, Hu Y, Smith JP, Strauss J, Yang G. Cohort profile: the China Health and
> Retirement Longitudinal Study (CHARLS). Int J Epidemiol. 2014;43(1):61–68.

We thank the China Center for Economic Research, the National School of Development of
Peking University for providing the data.
