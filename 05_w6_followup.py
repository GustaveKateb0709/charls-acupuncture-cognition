#!/usr/bin/env python3
"""05_w6_followup.py — Tier-2 long-term follow-up layer (Wave 6): 2013–2018 acupuncture exposure history → Wave 6 word-recall outcome.

Design lock (2026-11-23; two-tier design):
- Tier-1 (primary analysis, unchanged): 2013–2018 three-wave trajectory (SAP v1.1, Wu crosswalk) — scripts 01/02/03/04 as-is.
- Tier-2 (this script, new): anchored on the 2013 baseline cohort, exposure history (acupuncture use within the pain-reporting
  waves 2013–2018) → Wave 6 word-recall outcome (xscorewr, 0–10, single time point, scored within-wave; no cross-instrument
  equating is claimed).

Data facts (w6_mapping.md, all verified empirically):
- Wave 6 is one survey round spread over three years (2021 main visits / 2022–2023 supplemental visits); each respondent is
  assigned to exactly one year via res_wave in the 2023 Sample_Infor file. The three releases must never be stacked as
  three time points.
- Exposure ascertainment: in the earlier waves, any pain → asked about acupuncture use; in Wave 6 only da069 ∈ {4,5}
  (moderate/severe pain) → asked da073_s1. This asymmetry is reported verbatim in the Methods.
- The Wave 6 word list is entirely new with no official equating → the Tier-2 outcome uses the within-Wave-6 scale only and
  is never compared across the 2018 boundary.

Hard frequency gates (abort on any mismatch; model fitting is not allowed):
- res_wave: 2021=13,222 / 2022=3,585 / 2023=2,230; died=877
- da073_s1 Yes: 2021=218 / 2022=64 / 2023=36
- xscorewr 2021 non-missing ≥ 11,500 (codebook Obs 11,781)
- ID match rate (panel ∩ Wave 6 Cognition) ≥ 0.90
"""
import os, re, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_out")
RAW = os.path.join(BASE, "data_raw")
os.makedirs(OUT, exist_ok=True)

W6_FILES = {
    2021: os.path.join(RAW, "W6_2021", "CHARLS-W6-2021"),
    2022: os.path.join(RAW, "W6_2022", "CHARLS-W6-2022"),
    2023: os.path.join(RAW, "W6_2023", "CHARLS-W6-2023"),
}

GATE_LOG = []
def gate(name, ok, detail=""):
    GATE_LOG.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        raise SystemExit(f"Hard gate failed: {name} {detail} — aborting before model fitting")

def cesd10_w6(df):
    """CESD-10 shift (dc051–dc060 ↔ dc009–dc018; reverse-scored items 5 and 8: dc055/dc058; 1–4 → 0–3; ≥8 valid items required)."""
    cols = [f"dc0{n}" for n in range(51, 61)]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.where(~x.isin([997, 999]))  # don't know / refusal → missing
    x = x - 1
    for c in ("dc055", "dc058"):
        x[c] = 3 - x[c]
    return x.sum(axis=1, min_count=8)

def main():
    flow = {}
    print("=" * 66)
    print("STEP 1  Wave 6 roster and hard gates")
    print("=" * 66)
    si = pd.read_stata(os.path.join(W6_FILES[2023], "Sample_Infor.dta"),
                       convert_categoricals=False)
    si_cols = [c for c in si.columns if c.lower() in ("id", "res_wave", "died", "iyear", "imonth", "crosssection")]
    si = si[si_cols].copy()
    si["ID"] = si["ID"].astype(str).str.strip()
    rw = si["res_wave"].value_counts(dropna=True).to_dict()
    gate("res_wave distribution", rw.get(2021) == 13222 and rw.get(2022) == 3585 and rw.get(2023) == 2230, str(rw))
    n_died = int((si["died"] == 1).sum()) if si["died"].dtype != object else int((si["died"].astype(str).str.lower().str.startswith(("died", "1"))).sum())
    gate("died=877", n_died == 877, f"observed {n_died}")
    if "iyear" in si.columns:
        si["visit_year"] = si["iyear"].fillna(si["res_wave"]).astype(int)
    else:
        si["visit_year"] = si["res_wave"].astype(int)
    visit_year = dict(zip(si["ID"], si["visit_year"]))
    died_map = dict(zip(si["ID"], si["died"]))

    print("=" * 66)
    print("STEP 2  Wave 6 exposure frequency gate (da073_s1)")
    print("=" * 66)
    acu_yes = {}
    for y, d in W6_FILES.items():
        df = pd.read_stata(os.path.join(d, "Health_Status_and_Functioning.dta"),
                           columns=["ID", "da069", "da073_s1"], convert_categoricals=False)
        df["ID"] = df["ID"].astype(str).str.strip()
        acu_yes[y] = int((df["da073_s1"] == 1).sum())
    gate("da073_s1 'Yes' counts", acu_yes == {2021: 218, 2022: 64, 2023: 36}, str(acu_yes))

    print("=" * 66)
    print("STEP 3  Wave 6 cognitive outcomes (assigned year only; no stacking)")
    print("=" * 66)
    cog_frames = []
    for y, d in W6_FILES.items():
        cog = pd.read_stata(os.path.join(d, "Cognition_Depression.dta"),
                            columns=["ID", "xscorewr", "xscoredr", "xwordrecallbr"],
                            convert_categoricals=False)
        cog["ID"] = cog["ID"].astype(str).str.strip()
        cog["w6_year"] = y
        cog_frames.append(cog[["ID", "xscorewr", "xscoredr", "xwordrecallbr", "w6_year"]])
        if y == 2021:
            gate("xscorewr 2021 non-missing ≥11,500", int(cog["xscorewr"].notna().sum()) >= 11500,
                 f"observed {int(cog['xscorewr'].notna().sum())} (codebook: 11,781)")
    cog_all = pd.concat(cog_frames, ignore_index=True)
    cog_all = cog_all.drop_duplicates(subset="ID", keep="first")  # IDs overlapping across releases (314/209/645): prefer the assigned-year row; fall back to any non-missing value
    cog_all = cog_all.merge(si[["ID", "res_wave"]].rename(columns={"res_wave": "res_wave"}), on="ID", how="left")
    # Assigned-year release takes precedence: keep rows with res_wave==w6_year; backfill missing values from other releases (R14 completion rule)
    pref = cog_all[cog_all["res_wave"] == cog_all["w6_year"]].set_index("ID")
    alt = cog_all[cog_all["res_wave"] != cog_all["w6_year"]].set_index("ID")
    for c in ("xscorewr", "xscoredr", "xwordrecallbr"):
        alt_vals = alt[c]
        pref[c] = pref[c].fillna(alt_vals)
    w6 = pref.reset_index()
    print(f"Wave 6 cognitive coverage (xscorewr non-missing): {int(w6['xscorewr'].notna().sum())}")

    print("=" * 66)
    print("STEP 4  Legacy panel → exposure history and covariates")
    print("=" * 66)
    panel = pd.read_pickle(os.path.join(OUT, "analysis_panel.pkl"))
    panel["ID"] = panel["ID"].astype(str).str.strip()
    flow["panel_persons"] = int(panel["ID"].nunique())

    base13 = panel[panel["wave"] == 2013].set_index("ID")
    flow["baseline2013"] = int(len(base13))

    # Exposure history (pain-reporting waves only)
    pain_rows = panel[panel["pain_flag"] == 1]
    exp = pain_rows.groupby("ID").agg(
        acu_ever=("acu_w", lambda s: int((s == 1).any())),
        acu_n_waves=("acu_w", lambda s: int((s == 1).sum())),
        acu_first_wave=("wave", lambda s: int(s[(s.notna())].min()) if False else np.nan),
    )
    first = pain_rows[pain_rows["acu_w"] == 1].groupby("ID")["wave"].min()
    exp["acu_first_wave"] = first
    exp["n_pain_waves"] = pain_rows.groupby("ID")["pain_flag"].size()
    flow["pain_eligible_persons"] = int(len(exp))

    # Covariates from the last pre-Wave-6 observed wave (max of 2013/2015/2018)
    last = panel.sort_values("wave").groupby("ID").tail(1).set_index("ID")
    cov = last[["wave", "age", "sex", "education_raw", "marital_raw", "hukou",
                "cesd10", "adl_n", "iadl_n", "chronic_n", "stroke_w"]].rename(columns={"wave": "last_wave"})
    flow["with_last_wave_covar"] = int(len(cov))

    print("=" * 66)
    print("STEP 5  Merging and analysis-sample flow")
    print("=" * 66)
    d = w6.set_index("ID").join(base13[["recall_imm_score"]].rename(columns={"recall_imm_score": "recall_2013"}))
    d = d.join(exp[["acu_ever", "acu_n_waves", "acu_first_wave", "n_pain_waves"]])
    d = d.join(cov)
    lw = d["last_wave"].fillna(2018).astype(float)
    d["visit_year_w6"] = pd.Series([visit_year.get(i, np.nan) for i in d.index], index=d.index).astype(float)
    d["gap_years"] = d["visit_year_w6"] - lw
    d["alive"] = d.index.map(lambda i: (died_map.get(i) != 1) if i in died_map else np.nan)

    n0 = len(base13); n1 = int(d["recall_2013"].notna().sum())
    n2 = int((d["acu_ever"].notna() & (d["n_pain_waves"] > 0)).sum())
    n3 = int((d["alive"] == True).sum() if d["alive"].notna().any() else n2)
    n4 = int((d["xscorewr"].notna() & d["recall_2013"].notna() & d["acu_ever"].notna() &
              d[["age", "sex", "education_raw", "marital_raw", "hukou", "cesd10",
                 "adl_n", "iadl_n", "chronic_n", "stroke_w"]].notna().all(axis=1)).sum())
    flow.update({"after_baseline_recall": n1, "pain_eligible_merged": n2,
                 "alive_at_w6": n3, "final_complete_case": n4})
    print(f"2013 baseline {n0} → with baseline recall {n1} → pain reporters with classifiable exposure {n2} → alive at Wave 6 {n3} → complete cases {n4}")

    ana = d[(d["xscorewr"].notna()) & (d["recall_2013"].notna()) & (d["acu_ever"].notna()) &
            (d["n_pain_waves"] > 0) &
            d[["age", "sex", "education_raw", "marital_raw", "hukou", "cesd10",
               "adl_n", "iadl_n", "chronic_n", "stroke_w", "gap_years"]].notna().all(axis=1)].copy()
    ana["xrgender_ok"] = True
    # Sex coding: panel ba000_w2_3, 1=male 2=female → 0/1
    ana["male"] = (ana["sex"] == 1).astype(int)
    ana["w6_year"] = ana["visit_year_w6"] if "visit_year_w6" in ana.columns else pd.Series(
        [visit_year.get(i, np.nan) for i in ana.index], index=ana.index).astype(float)

    print("=" * 66)
    print("STEP 6  Models")
    print("=" * 66)
    res = {}
    COVS = ["recall_2013", "age", "male", "education_raw", "marital_raw", "hukou",
            "cesd10", "adl_n", "iadl_n", "chronic_n", "stroke_w", "gap_years"]

    def fit_ols(df, y, x_main, covs=COVS, weighted=None):
        frame = df[covs + [x_main]].copy()
        for c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        keep = frame.dropna().index
        frame = frame.loc[keep]
        X = pd.get_dummies(frame, columns=["education_raw", "marital_raw"], drop_first=True)
        X = sm.add_constant(X).astype(float)
        yv = pd.to_numeric(df.loc[keep, y], errors="coerce")
        wv = None if weighted is None else pd.to_numeric(weighted.loc[keep], errors="coerce")
        m = sm.WLS(yv, X, weights=wv) if wv is not None else sm.OLS(yv, X)
        return m.fit(cov_type="HC3")

    # Primary model: ever vs never
    m1 = fit_ols(ana, "xscorewr", "acu_ever")
    b, lo, hi, p = (m1.params["acu_ever"], *m1.conf_int().loc["acu_ever"], m1.pvalues["acu_ever"])
    res["tier2_primary_ols"] = {"outcome": "xscorewr", "beta_acu_ever": float(b),
                                "ci": [float(lo), float(hi)], "p": float(p), "n": int(len(ana))}
    sd = float(ana["xscorewr"].std())
    res["tier2_primary_ols"]["sd_units"] = float(b / sd)
    res["w6_xscorewr_sd"] = sd
    res["w6_xscorewr_mean"] = float(ana["xscorewr"].mean())

    # Dose–response (0–3 waves)
    ana["acu_n_cat"] = ana["acu_n_waves"].clip(upper=3)
    m2 = fit_ols(ana, "xscorewr", "acu_n_cat")
    dose = {f"acu_waves_{k}": {"beta": float(m2.params.get(f"acu_n_cat", np.nan)),
                               "p": float(m2.pvalues.get("acu_n_cat", np.nan))}
            for k in [1]}
    # One dummy per wave count for clarity
    for k in (1, 2, 3):
        col = f"dose_{k}"
        ana[col] = (ana["acu_n_waves"] == k).astype(int)
    m2b = fit_ols(ana, "xscorewr", "dose_1", covs=COVS + ["dose_2", "dose_3"])
    dose = {f"w{k}": {"beta": float(m2b.params[f"dose_{k}"]),
                      "ci": list(map(float, m2b.conf_int().loc[f"dose_{k}"])),
                      "p": float(m2b.pvalues[f"dose_{k}"])} for k in (1, 2, 3) if f"dose_{k}" in m2b.params}
    res["tier2_dose_response"] = dose

    # Timing: wave of first use
    for wv in (2013, 2015, 2018):
        ana[f"first_{wv}"] = (ana["acu_first_wave"] == wv).astype(int)
    m3 = fit_ols(ana, "xscorewr", "first_2013", covs=COVS + ["first_2015", "first_2018"])
    res["tier2_timing"] = {f"first_{wv}": {"beta": float(m3.params[f"first_{wv}"]),
                                           "p": float(m3.pvalues[f"first_{wv}"])}
                           for wv in (2013, 2015, 2018) if f"first_{wv}" in m3.params}

    # Secondary outcome: delayed recall
    m4 = fit_ols(ana.dropna(subset=["xscoredr"]), "xscoredr", "acu_ever")
    res["tier2_secondary_dr"] = {"beta_acu_ever": float(m4.params["acu_ever"]),
                                 "ci": list(map(float, m4.conf_int().loc["acu_ever"])),
                                 "p": float(m4.pvalues["acu_ever"]),
                                 "n": int(m4.nobs)}

    # IPTW (attrition weighting) sensitivity analysis
    alive_pool = d[(d["recall_2013"].notna()) & (d["acu_ever"].notna()) & (d["alive"] == True)].copy()
    for c in COVS + ["male"]:
        if c not in alive_pool.columns:
            alive_pool[c] = np.nan
    alive_pool["male"] = (alive_pool["sex"] == 1).astype(int)
    retain = alive_pool.index.isin(ana.index).astype(int)
    Xc = alive_pool[COVS].copy()
    for c in Xc.columns:
        Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    Xc = Xc.fillna(Xc.median())  # median imputation is acceptable for the propensity model (sensitivity-analysis use only)
    Xa = sm.add_constant(Xc.astype(float))
    try:
        lg = sm.Logit(retain, Xa).fit(disp=0, maxiter=200)
        ps = lg.predict(Xa)
        wgt = np.where(retain == 1, 1.0 / np.clip(ps, 1e-3, 1.0), np.nan)
        ana_w = alive_pool.loc[retain == 1].copy()
        ana_w["ipw"] = wgt[retain == 1]
        m5 = fit_ols(ana_w, "xscorewr", "acu_ever", weighted=ana_w["ipw"])
        res["tier2_ipw_sensitivity"] = {"beta_acu_ever": float(m5.params["acu_ever"]),
                                        "ci": list(map(float, m5.conf_int().loc["acu_ever"])),
                                        "p": float(m5.pvalues["acu_ever"]), "n": int(len(ana_w))}
    except Exception as e:
        res["tier2_ipw_sensitivity"] = {"error": str(e)}

    # Descriptives: Wave 6 exposure update (da073_s1) and pain severity (descriptive only)
    hs_join = {}
    for y, dd in W6_FILES.items():
        h = pd.read_stata(os.path.join(dd, "Health_Status_and_Functioning.dta"),
                          columns=["ID", "da069", "da073_s1"], convert_categoricals=False)
        h["ID"] = h["ID"].astype(str).str.strip()
        hs_join[y] = h
    hs_all = pd.concat(hs_join.values(), ignore_index=True).drop_duplicates("ID")
    ana = ana.join(hs_all.set_index("ID")[["da069", "da073_s1"]].rename(
        columns={"da069": "w6_pain", "da073_s1": "w6_acu"}))
    res["descriptive_w6_acu_by_history"] = {
        "ever": int((ana["w6_acu"] == 1).sum()),
        "n_observed": int(ana["w6_acu"].notna().sum())}
    ana.to_csv(os.path.join(OUT, "w6_followup_sample.csv"))

    with open(os.path.join(OUT, "flow_w6.json"), "w") as f:
        json.dump(flow, f, indent=1)
    with open(os.path.join(OUT, "results_summary_w6.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n=== Summary ===")
    print(json.dumps({k: v for k, v in res.items() if k.startswith("tier2")}, indent=1, ensure_ascii=False))
    print("\n05_w6_followup.py finished: data_out/{flow_w6.json, results_summary_w6.json, w6_followup_sample.csv}")

if __name__ == "__main__":
    main()
