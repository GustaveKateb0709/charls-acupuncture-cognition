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
- `05_w6_followup.py` — Tier-2 long-term follow-up layer (Wave 6): anchored on the 2013
  baseline cohort, links acupuncture exposure history accumulated in the pain-reporting
  waves (2013–2018) to the Wave 6 word-recall outcome (`xscorewr`), with hard frequency
  gates that abort before model fitting on any mismatch, and an inverse-probability-of-censoring
  sensitivity analysis.
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

## Wave 6 long-term follow-up layer (Tier-2)

`05_w6_followup.py` implements the second tier of the two-tier design: it re-anchors on
the 2013 baseline cohort and tests whether acupuncture exposure accumulated during the
2013–2018 pain waves predicts word recall measured in CHARLS Wave 6. Wave 6 is one survey
round fielded across 2021–2023; each respondent is assigned to exactly one year via
`res_wave` in the 2023 `Sample_Infor` file, and the three releases are never stacked as
three time points. Because the Wave 6 word list is new and has no official equating, the
outcome is analysed on the within-Wave-6 scale only.

To run it, additionally place the Wave 6 releases under `data_raw/`:

```
data_raw/
├── W6_2021/CHARLS-W6-2021/  (Sample_Infor.dta, Health_Status_and_Functioning.dta, Cognition_Depression.dta)
├── W6_2022/CHARLS-W6-2022/  (same files)
└── W6_2023/CHARLS-W6-2023/  (same files)
```

The script requires the analysis panel built by `01_harmonize.py` (`data_out/analysis_panel.pkl`)
and writes `data_out/flow_w6.json`, `data_out/results_summary_w6.json` and
`data_out/w6_followup_sample.csv`. Hard-coded frequency gates (wave roster, exposure counts,
minimum cognitive coverage) are checked before any model is fitted; the script aborts on any
mismatch. Results quoted in the associated paper for this layer can only be reproduced by
re-running this script on the licensed CHARLS Wave 6 data.

## Requirements

Python 3.11+ with pandas, numpy, statsmodels, scikit-learn, matplotlib.
Optional: econml and shap for the exploratory causal-forest analysis.

## Citation

If you use this code, please cite the associated paper and the CHARLS dataset:

> Zhao Y, Hu Y, Smith JP, Strauss J, Yang G. Cohort profile: the China Health and
> Retirement Longitudinal Study (CHARLS). Int J Epidemiol. 2014;43(1):61–68.

We thank the China Center for Economic Research, the National School of Development of
Peking University for providing the data.
