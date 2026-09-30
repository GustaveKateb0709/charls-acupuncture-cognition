#!/usr/bin/env python3
"""02_analysis.py — 主分析（SAP v1.2 §7–§9）。

面板：data_out/analysis_panel.pkl（01_harmonize.py 产出，验证闸 27/27 PASS）。
产出：table1–table4.csv、flow.json、fig_data/、results_summary.json。

口径要点（均可在本脚本内溯源复算）：
- 分析人群：pain_flag==1 至少一波；cog_total 非缺失 ≥2 波；首次进面板时 age≥45。
- 模型行：纳入者 × 其疼痛波行（pain_flag==1，暴露才有定义——非疼痛波为结构缺失）。
- 基线：每人首次进面板波次；教育跨波解析（bd001_w2_4==12 为『未变化』，按
  Harmonized 惯例取该人任一波次的实测教育；2018 波无 12，天然兜底）。
- 主模型 LMM：cog_total ~ time*acu_w + 基线协变量 + (1|ID)，time=year_since_baseline。
- 所有输出为关联性数字，无因果措辞（行文由 team-lead 负责）。
"""
import json
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_out")
FIGD = os.path.join(OUT, "fig_data")
os.makedirs(FIGD, exist_ok=True)

RNG = 20260930

# ---------------------------------------------------------------- 面板与人群
panel = pd.read_pickle(os.path.join(OUT, "analysis_panel.pkl"))
panel = panel.sort_values(["ID", "wave"]).reset_index(drop=True)

flow = {}
flow["panel_rows"] = len(panel)
flow["panel_individuals"] = int(panel.ID.nunique())

# 教育解析：人内跨波取任一非 12、非缺失实测值（12=『与上波相同』，非等级）
def _resolve_edu(g):
    real = g[(g != 12) & g.notna()]
    return real.iloc[0] if len(real) else np.nan

panel["education_resolved"] = panel.groupby("ID")["education_raw"].transform(_resolve_edu)
panel["education_resolved"] = panel.groupby("ID")["education_resolved"].ffill()
panel["education_resolved"] = panel.groupby("ID")["education_resolved"].bfill()

# 并档（SAP：原编码保留于 education_raw，模型用并档）
def _edu_cat(v):
    if pd.isna(v):
        return np.nan
    if v <= 3:
        return 1          # 文盲/未完成小学/私塾
    if v == 4:
        return 2          # 小学
    if v == 5:
        return 3          # 初中
    if v in (6, 7):
        return 4          # 高中/职校
    return 5              # 大专及以上（8–11）

panel["educ_cat"] = panel["education_resolved"].map(_edu_cat)

def _marital_cat(v):
    if pd.isna(v):
        return np.nan
    if v in (1, 2, 7):
        return 1          # 有配偶/同居
    if v in (3, 4):
        return 2          # 分居/离异
    if v == 5:
        return 3          # 丧偶
    return 4              # 从未结婚

panel["marital_cat"] = panel["marital_raw"].map(_marital_cat)

def _hukou_cat(v):
    if pd.isna(v):
        return np.nan
    if v == 1:
        return 1          # 农业
    if v in (2, 3):
        return 2          # 非农业/居民统一
    return 3              # 无户口

panel["hukou_cat"] = panel["hukou"].map(_hukou_cat)

# 纳入排除逐级计数（flow.json）
per = panel.groupby("ID").agg(
    n_pain=("pain_flag", lambda s: int((s == 1).sum())),
    n_cog=("cog_total", lambda s: int(s.notna().sum())),
    first_wave=("wave", "min"),
)
per["age_first"] = panel.groupby("ID")["age"].first()
flow["n_pain_any"] = int((per.n_pain >= 1).sum())
step2 = per[per.n_pain >= 1]
flow["n_cog2"] = int((step2.n_cog >= 2).sum())
step3 = step2[step2.n_cog >= 2]
flow["n_age45"] = int((step3.age_first >= 45).sum())
included_ids = step3[step3.age_first >= 45].index

df_all = panel[panel.ID.isin(included_ids)].copy()
flow["n_included"] = int(df_all.ID.nunique())

# 基线（首次进面板波次）
base = df_all.sort_values(["ID", "wave"]).groupby("ID").first()
base_cols = {"age": "age_b", "sex": "sex_b", "educ_cat": "educ_b", "marital_cat": "marital_b",
             "hukou_cat": "hukou_b", "cesd10": "cesd_b", "adl_n": "adl_b",
             "chronic_n": "chronic_b", "wave": "wave_b", "pain_flag": "pain_b",
             "acu_w": "acu_b", "education_resolved": "edu_res_b", "stroke_w": "stroke_b"}
base = base[list(base_cols)].rename(columns=base_cols)
# 基线暴露：首次进面板波若非疼痛波（acu 缺失），取首个疼痛波的暴露
first_pain = (df_all[df_all.pain_flag == 1].sort_values(["ID", "wave"]).groupby("ID").first())
base["acu_firstpain"] = first_pain["acu_w"]
base["wave_firstpain"] = first_pain["wave"]

df = df_all[df_all.pain_flag == 1].copy()          # 模型行 = 疼痛波行
df = df.merge(base[["age_b", "sex_b", "educ_b", "marital_b", "hukou_b", "cesd_b",
                    "adl_b", "chronic_b", "wave_b", "acu_firstpain"]],
              left_on="ID", right_index=True, how="left")

flow["model_rows_all"] = len(df)
flow["n_baseline_acu"] = int((base["acu_firstpain"] == 1).sum())

# 完整病例（主分析）
COVS = ["age_b", "C(sex_b)", "C(educ_b)", "C(marital_b)", "C(hukou_b)", "cesd_b", "adl_b", "chronic_b"]
FORM = ("cog_total ~ year_since_baseline * acu_w + age_b + C(sex_b) + C(educ_b) "
        "+ C(marital_b) + C(hukou_b) + cesd_b + adl_b + chronic_b")
need = ["cog_total", "year_since_baseline", "acu_w", "age_b", "sex_b", "educ_b",
        "marital_b", "hukou_b", "cesd_b", "adl_b", "chronic_b"]
cc = df.dropna(subset=[c for c in need])
cc = cc.dropna(subset=["educ_b", "marital_b", "hukou_b", "sex_b"])
flow["model_rows_cc"] = len(cc)
flow["n_cc_persons"] = int(cc.ID.nunique())


def fmt_res(name, res, term):
    """提取 β/95CI/p 三元组（MixedLM/GEE 通用）。"""
    ci = res.conf_int()
    return {"term": term, "model": name, "beta": float(res.params[term]),
            "ci_lo": float(ci.loc[term].iloc[0]), "ci_hi": float(ci.loc[term].iloc[1]),
            "p": float(res.pvalues[term])}


results = {}

# ---------------------------------------------------------------- 1. 主模型 LMM
lmm = smf.mixedlm(FORM, cc, groups=cc["ID"])
lmm_res = lmm.fit(reml=False, method="lbfgs")
INT = "year_since_baseline:acu_w"
main = fmt_res("LMM_main", lmm_res, INT)
main["five_year_diff"] = 5 * main["beta"]
main["five_year_ci"] = [5 * main["ci_lo"], 5 * main["ci_hi"]]
results["lmm_main"] = main
results["lmm_main_resid_sd"] = float(np.sqrt(lmm_res.scale))
results["lmm_main_sd_y"] = float(cc.cog_total.std())
results["n_obs_lmm"] = int(lmm_res.nobs)
results["n_groups_lmm"] = int(cc.ID.nunique())

# ---------------------------------------------------------------- 2. GEE 敏感性
gee = smf.gee(FORM, "ID", cc, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
gee_res = gee.fit()
results["gee_main"] = fmt_res("GEE_robust", gee_res, INT)
results["n_obs_gee"] = int(gee_res.nobs)

# ---------------------------------------------------------------- 3. E-value（EValue 4.1.4 evalues.OLS 官方公式）
def evalue_ols(est, se, sd, delta=1.0):
    """CRAN EValue::evalues.OLS 精确复刻：
    MD = est*delta/sd；RR = exp(0.91*MD)；CI 用 exp(0.91*MD ± 1.78*SE_MD)；
    E-value = x + sqrt(x*(x-1))（x 为>1 化后的 RR）；CI 跨 RR=1 时 CI E-value=1。
    sd 用残差 SD（SD(Y|X,C)，文档推荐），保守口径另报 SD(Y)。"""
    def thr(rr, true=1.0):
        if rr <= 1:
            rr = 1 / rr
            true = 1 / true
        return (rr + math.sqrt(rr * (rr - true))) / true

    md, se_md = est * delta / sd, se * delta / sd
    rr_p = math.exp(0.91 * md)
    lo, hi = math.exp(0.91 * md - 1.78 * se_md), math.exp(0.91 * md + 1.78 * se_md)
    e_p = thr(rr_p)
    if lo > 1 or hi < 1:                      # CI 不跨零：取靠近零的限
        lim = hi if lo > 1 else lo
        e_ci = thr(lim)
    else:
        e_ci = 1.0
    return {"evalue_point": e_p, "evalue_ci": e_ci, "rr_approx": rr_p,
            "rr_lo": lo, "rr_hi": hi}


se_int = float(lmm_res.bse[INT])
results["evalue_int_residSD"] = evalue_ols(main["beta"], se_int, results["lmm_main_resid_sd"])
results["evalue_int_sdY"] = evalue_ols(main["beta"], se_int, results["lmm_main_sd_y"])

# ---------------------------------------------------------------- 4. IPTW（表 3）
ps_covs = pd.get_dummies(cc[["age_b", "sex_b", "educ_b", "marital_b", "hukou_b",
                             "cesd_b", "adl_b", "chronic_b"]],
                         columns=["sex_b", "educ_b", "marital_b", "hukou_b"], drop_first=True)
ps_X = sm.add_constant(ps_covs.astype(float))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=2000, C=1e6)
lr.fit(ps_X, cc["acu_w"].astype(int))
ps = np.clip(lr.predict_proba(ps_X)[:, 1], 1e-6, 1 - 1e-6)
a = cc["acu_w"].astype(int).values
p_marg = a.mean()
w = np.where(a == 1, p_marg / ps, (1 - p_marg) / (1 - ps))
lo_q, hi_q = np.quantile(w, [0.01, 0.99])
w_tr = np.clip(w, lo_q, hi_q)
cc = cc.assign(ps=ps, sw=w_tr)
results["iptw"] = {"n": len(cc), "ps_min": float(ps.min()), "ps_max": float(ps.max()),
                   "w_raw_mean": float(w.mean()), "w_max_raw": float(w.max()),
                   "trunc_q01": float(lo_q), "trunc_q99": float(hi_q),
                   "ess": float(w_tr.sum() ** 2 / (w_tr ** 2).sum())}

# SMD 平衡诊断（加权前后；连续用均值差/合并 SD，二分用 p 差/合并 SD）
def smd(x, a_, w_=None):
    if w_ is None:
        m1, m0 = x[a_ == 1].mean(), x[a_ == 0].mean()
        v1, v0 = x[a_ == 1].var(), x[a_ == 0].var()
    else:
        m1 = np.average(x[a_ == 1], weights=w_[a_ == 1])
        m0 = np.average(x[a_ == 0], weights=w_[a_ == 0])
        v1 = np.average((x[a_ == 1] - m1) ** 2, weights=w_[a_ == 1])
        v0 = np.average((x[a_ == 0] - m0) ** 2, weights=w_[a_ == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0

bal_rows = []
dummies = ps_covs.copy()
dummies["age_b"] = cc["age_b"]
for c in dummies.columns:
    x = dummies[c].astype(float).values
    bal_rows.append({"covariate": c, "smd_unw": float(smd(x, a)),
                     "smd_w": float(smd(x, a, w_tr))})
bal = pd.DataFrame(bal_rows)
results["iptw"]["max_smd_unw"] = float(bal.smd_unw.abs().max())
results["iptw"]["max_smd_w"] = float(bal.smd_w.abs().max())
bal.to_csv(os.path.join(OUT, "table3_balance.csv"), index=False)

gee_w = smf.gee(FORM, "ID", cc, family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Independence(), weights=cc["sw"]).fit()
results["iptw_gee"] = fmt_res("IPTW_GEE", gee_w, INT)

# ---------------------------------------------------------------- 表 1（基线特征按基线针灸使用分组）
def smd_col(x, g):
    x1, x0 = x[g == 1].dropna(), x[g == 0].dropna()
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    return (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2)

tb1 = []
base_df = base.loc[sorted(included_ids)].copy()
base_df["acu_grp"] = base_df["acu_firstpain"]

def add_row(label, series, kind):
    g = base_df["acu_grp"]
    row = {"variable": label}
    if kind == "cont":
        row["acu1_mean_sd"] = f"{series[g == 1].mean():.2f} ({series[g == 1].std():.2f})"
        row["acu0_mean_sd"] = f"{series[g == 0].mean():.2f} ({series[g == 0].std():.2f})"
        row["smd"] = round(smd_col(series.astype(float), g), 3)
        row["missing"] = int(series.isna().sum())
    else:
        lv = series.value_counts()
        s1n, s0n = int((g == 1).sum()), int((g == 0).sum())
        row["acu1_mean_sd"] = f"{int(lv.get(kind, 0))} ({lv.get(kind, 0) / s1n * 100:.1f}%)"
        row["acu0_mean_sd"] = f"{int(lv.get(kind, 0) and 0)} ({0:.1f}%)"  # 占位，下方重算
        # 分类变量逐水平百分比
        p1 = (series[g == 1] == kind).mean() * 100
        p0 = (series[g == 0] == kind).mean() * 100
        row["acu1_mean_sd"] = f"{int((series[g == 1] == kind).sum())} ({p1:.1f}%)"
        row["acu0_mean_sd"] = f"{int((series[g == 0] == kind).sum())} ({p0:.1f}%)"
        b = (series == kind).astype(float)
        row["smd"] = round(smd_col(b, g), 3)
        row["missing"] = int(series.isna().sum())
    tb1.append(row)

add_row("Age, years", base_df["age_b"], "cont")
for s, lab in [(1, "Male"), (2, "Female")]:
    r = {"variable": f"Sex = {lab}"}
    g = base_df["acu_grp"]
    p1 = (base_df["sex_b"] == s)[g == 1].mean() * 100
    p0 = (base_df["sex_b"] == s)[g == 0].mean() * 100
    r["acu1_mean_sd"] = f"{int((base_df['sex_b'][g == 1] == s).sum())} ({p1:.1f}%)"
    r["acu0_mean_sd"] = f"{int((base_df['sex_b'][g == 0] == s).sum())} ({p0:.1f}%)"
    r["smd"] = round(smd_col((base_df["sex_b"] == s).astype(float), g), 3)
    r["missing"] = int(base_df["sex_b"].isna().sum())
    tb1.append(r)
EDU_LAB = {1: "No formal education/sishu", 2: "Primary school", 3: "Middle school",
           4: "High school/vocational", 5: "College or above"}
for k, lab in EDU_LAB.items():
    add_row(f"Education: {lab}", base_df["educ_b"], k)
MAR_LAB = {1: "Married/partnered", 2: "Separated/divorced", 3: "Widowed", 4: "Never married"}
for k, lab in MAR_LAB.items():
    add_row(f"Marital: {lab}", base_df["marital_b"], k)
HUK_LAB = {1: "Agricultural hukou", 2: "Non-agricultural/unified", 3: "No hukou"}
for k, lab in HUK_LAB.items():
    add_row(f"Hukou: {lab}", base_df["hukou_b"], k)
add_row("Baseline wave 2013", (base_df["wave_b"] == 2013).astype(float), 1)
add_row("Baseline wave 2015", (base_df["wave_b"] == 2015).astype(float), 1)
add_row("CES-D 10 score", base_df["cesd_b"], "cont")
add_row("ADL difficulties (n)", base_df["adl_b"], "cont")
add_row("Chronic diseases (n)", base_df["chronic_b"], "cont")
add_row("Stroke (yes)", base_df["stroke_b"], 1)
add_row("cog_total at baseline", base_df.merge(
    panel[panel.cog_total.notna()].groupby("ID").first()[["cog_total"]],
    left_index=True, right_index=True, how="left")["cog_total"], "cont")
table1 = pd.DataFrame(tb1)
table1.insert(0, "n_acu1", int((base_df.acu_grp == 1).sum()))
table1.insert(0, "n_acu0", int((base_df.acu_grp == 0).sum()))
table1.to_csv(os.path.join(OUT, "table1.csv"), index=False)

# ---------------------------------------------------------------- 表 2（主模型 + 次要结局）
rows = []
for term in ["year_since_baseline", "acu_w", INT]:
    rows.append(fmt_res("LMM_main", lmm_res, term))
for term in ["year_since_baseline", "acu_w", INT]:
    rows.append(fmt_res("IPTW_GEE", gee_w, term))
# 次要结局：即时/延迟词回忆（2018 为等值分）
for yv, ylab in [("recall_imm_score", "Immediate recall (equated)"),
                 ("recall_del_score", "Delayed recall (equated)")]:
    f2 = yv + " ~ year_since_baseline * acu_w + age_b + C(sex_b) + C(educ_b) + C(marital_b) " \
         "+ C(hukou_b) + cesd_b + adl_b + chronic_b"
    d2 = cc.dropna(subset=[yv])
    if d2[yv].nunique() > 1:
        r2 = smf.mixedlm(f2, d2, groups=d2["ID"]).fit(reml=False, method="lbfgs")
        rows.append(fmt_res(f"secondary:{ylab}", r2, INT))
        results[f"secondary_{yv}"] = fmt_res(f"secondary:{ylab}", r2, INT)
        results[f"secondary_{yv}_n"] = int(r2.nobs)
table2 = pd.DataFrame(rows)
table2.to_csv(os.path.join(OUT, "table2.csv"), index=False)

# ---------------------------------------------------------------- 5. 敏感性（表 4）
sens = []
def run_lmm(d, label, form=FORM, term=INT):
    d = d.dropna(subset=[c for c in need])
    d = d.dropna(subset=["educ_b", "marital_b", "hukou_b", "sex_b"])
    if d[term.split(":")[0]].nunique() < 2 or d.ID.nunique() < 100:
        return None
    try:
        r = smf.mixedlm(form, d, groups=d["ID"]).fit(reml=False, method="lbfgs")
    except Exception as e1:
        try:
            # 优化器回退（低频暴露Replacement模型在 lbfgs 下易出现 Singular matrix）
            r = smf.mixedlm(form, d, groups=d["ID"]).fit(reml=True, method="powell", maxiter=2000)
        except Exception as e2:
            sens.append({"model": label, "error": (str(e1) + " || " + str(e2))[:300]})
            return None
    rr = fmt_res(label, r, term)
    rr["n_obs"] = int(r.nobs)
    rr["n_persons"] = int(d.ID.nunique())
    sens.append(rr)
    return r

# ① 剔 2013 波
run_lmm(cc[cc.wave != 2013], "sens1_drop2013")
# ③ 2015+2018 持续/偶发/从未
d38 = cc[cc.wave.isin([2015, 2018])]
pat = d38.groupby("ID")["acu_w"].agg(["sum", "count"])
pat = pat[pat["count"] == 2]
pmap = {i: ("sustained" if v["sum"] == 2 else ("occasional" if v["sum"] == 1 else "never"))
        for i, v in pat.iterrows()}
d38 = d38[d38.ID.isin(pmap)].copy()
d38["acu_pat"] = d38.ID.map(pmap)
FORM_PAT = FORM.replace("year_since_baseline * acu_w",
                        "year_since_baseline * C(acu_pat, Treatment('never'))")
r3 = run_lmm(d38, "sens3_pattern2015_18", FORM_PAT,
             "year_since_baseline:C(acu_pat, Treatment('never'))[T.occasional]")
if r3 is not None:
    results["sens3_sustained_vs_never"] = fmt_res(
        "sens3_pattern2015_18", r3, "year_since_baseline:C(acu_pat, Treatment('never'))[T.sustained]")
    results["sens3_occasional_vs_never"] = fmt_res(
        "sens3_pattern2015_18", r3, "year_since_baseline:C(acu_pat, Treatment('never'))[T.occasional]")
# ④ 暴露替换
for alt, lab in [("tcm_w", "sens4_tcm"), ("massage_w", "sens4_massage")]:
    d4 = cc.copy()
    d4["acu_w"] = d4[alt]
    d4 = d4.dropna(subset=["acu_w"])
    run_lmm(d4, lab)
# ⑤ 2018 仍在访子样本
in18 = set(panel[panel.wave == 2018].ID)
run_lmm(cc[cc.ID.isin(in18)], "sens5_surv2018")

# ② MI（IterativeImputer m=20；Rubin 合并交互项）
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

mi_rows_all = df.dropna(subset=["cog_total", "year_since_baseline", "acu_w"])
mi_cols_num = ["cog_total", "year_since_baseline", "acu_w", "age_b", "cesd_b", "adl_b", "chronic_b"]
mi_cat = pd.get_dummies(mi_rows_all[["sex_b", "educ_b", "marital_b", "hukou_b"]],
                        columns=["sex_b", "educ_b", "marital_b", "hukou_b"], drop_first=True).astype(float)
mi_X = pd.concat([mi_rows_all[mi_cols_num].reset_index(drop=True),
                  mi_cat.reset_index(drop=True)], axis=1).astype(float)
imp = IterativeImputer(estimator=BayesianRidge(), max_iter=12, sample_posterior=True,
                       random_state=RNG)
M = 10     # m=10（SAP 允许注明）；m=20 计算成本高时取 10
EDU_D = [c for c in mi_X.columns if c.startswith("educ_b_")]
MAR_D = [c for c in mi_X.columns if c.startswith("marital_b_")]
HUK_D = [c for c in mi_X.columns if c.startswith("hukou_b_")]
betas, vars_ = [], []
for m_ in range(M):
    Ximp = imp.fit_transform(mi_X)
    di = pd.DataFrame(Ximp, columns=mi_X.columns)
    di["ID"] = mi_rows_all["ID"].values
    # 从哑变量重构分类协变量（参照水平 = 1 − 各哑变量，取 >0.5 判定后求和）
    di["sex_b"] = 1 + (di["sex_b_2.0"] > 0.5).astype(float)
    di["educ_b"] = (1 + di[EDU_D].gt(0.5).sum(axis=1)).clip(1, 5)
    di["marital_b"] = (1 + di[MAR_D].gt(0.5).sum(axis=1)).clip(1, 4)
    di["hukou_b"] = (1 + di[HUK_D].gt(0.5).sum(axis=1)).clip(1, 3)
    try:
        rmi = smf.mixedlm(FORM, di, groups=di["ID"]).fit(reml=False, method="lbfgs")
        betas.append(float(rmi.params[INT]))
        vars_.append(float(rmi.bse[INT]) ** 2)
    except Exception as e:
        continue
if betas:
    m_ = len(betas)
    beta_bar = float(np.mean(betas))
    w_within = float(np.mean(vars_))
    b_between = float(np.var(betas, ddof=1))
    total_var = w_within + (1 + 1 / m_) * b_between
    se_mi = math.sqrt(total_var)
    from scipy import stats as sps
    zval = beta_bar / se_mi
    p_mi = 2 * (1 - sps.norm.cdf(abs(zval)))
    results["sens2_MI"] = {"m": m_, "beta": beta_bar, "se": se_mi,
                           "ci_lo": beta_bar - 1.96 * se_mi, "ci_hi": beta_bar + 1.96 * se_mi,
                           "p": p_mi}
    sens.append({"model": "sens2_MI_m10", "beta": beta_bar, "ci_lo": beta_bar - 1.96 * se_mi,
                 "ci_hi": beta_bar + 1.96 * se_mi, "p": p_mi, "n_obs": len(mi_X),
                 "n_persons": int(mi_rows_all.ID.nunique())})

table4 = pd.DataFrame(sens)
table4.to_csv(os.path.join(OUT, "table4.csv"), index=False)
results["sens_list"] = [s.get("model") for s in sens]

# ---------------------------------------------------------------- fig2：调整后边际轨迹
fe = lmm_res.fe_params
cov = lmm_res.cov_params().loc[fe.index, fe.index]
Xref = cc[["age_b", "sex_b", "educ_b", "marital_b", "hukou_b", "cesd_b", "adl_b", "chronic_b"]]
ref = pd.DataFrame({
    "Intercept": 1.0, "year_since_baseline": [0.0, 2.0, 5.0] * 2,
    "acu_w": [0, 0, 0, 1, 1, 1],
    "age_b": Xref.age_b.mean(), "sex_b": 2.0,
    "C(sex_b)[T.2.0]": 1.0,
    "educ_b": 1.0, "C(educ_b)[T.2.0]": 0, "C(educ_b)[T.3.0]": 0,
    "C(educ_b)[T.4.0]": 0, "C(educ_b)[T.5.0]": 0,
    "marital_b": 1.0, "C(marital_b)[T.2.0]": 0, "C(marital_b)[T.3.0]": 0,
    "C(marital_b)[T.4.0]": 0,
    "hukou_b": 1.0, "C(hukou_b)[T.2.0]": 0, "C(hukou_b)[T.3.0]": 0,
    "cesd_b": Xref.cesd_b.mean(), "adl_b": Xref.adl_b.mean(), "chronic_b": Xref.chronic_b.mean(),
})
ref.loc[:, "C(educ_b)[T.2.0]"] = 1.0        # 参照人群：小学
ref = ref[[c for c in fe.index if c in ref.columns]]
ref = ref.reindex(columns=fe.index, fill_value=0)
mu = ref.values @ fe.values
var_ = np.einsum("ij,jk,ik->i", ref.values, cov.values, ref.values)
se_ = np.sqrt(var_)
fig2 = pd.DataFrame({"time": ref.year_since_baseline, "acu_w": ref.acu_w,
                     "pred": mu, "lo": mu - 1.96 * se_, "hi": mu + 1.96 * se_})
fig2.to_csv(os.path.join(FIGD, "fig2_adjusted_trajectories.csv"), index=False)

# ---------------------------------------------------------------- fig3：森林数据（敏感性）
forest = table4[[c for c in ["model", "beta", "ci_lo", "ci_hi", "p", "n_obs", "n_persons"]
                 if c in table4.columns]].copy()
forest = pd.concat([pd.DataFrame([{"model": "LMM main", "beta": main["beta"],
                                   "ci_lo": main["ci_lo"], "ci_hi": main["ci_hi"],
                                   "p": main["p"], "n_obs": results["n_obs_lmm"],
                                   "n_persons": results["n_groups_lmm"]}]), forest],
                   ignore_index=True)
forest.to_csv(os.path.join(FIGD, "fig3_forest.csv"), index=False)

# ---------------------------------------------------------------- flow.json & summary
with open(os.path.join(OUT, "flow.json"), "w") as f:
    json.dump(flow, f, indent=2, ensure_ascii=False)

# CATE（探索性，econml 可用才做）
try:
    from econml.dml import CausalForestDML
    dca = cc.dropna(subset=["cog_total", "acu_w"]).copy()
    dca = dca[dca.wave == 2018]                      # 用 2018 横截面：结局=当期认知，暴露=当期针灸
    Xc = pd.get_dummies(dca[["age_b", "sex_b", "educ_b", "marital_b", "hukou_b",
                             "cesd_b", "adl_b", "chronic_b"]],
                        columns=["sex_b", "educ_b", "marital_b", "hukou_b"], drop_first=True).astype(float)
    Xc = Xc.fillna(Xc.mean())
    cf = CausalForestDML(n_estimators=400, min_samples_leaf=20, random_state=RNG)
    cf.fit(Y=dca.cog_total.values, T=dca.acu_w.values.astype(int), X=Xc.values)
    tau = cf.effect(Xc.values)
    cate = pd.DataFrame({"ID": dca.ID.values, "tau": tau,
                         "lo": cf.effect_interval(Xc.values)[0],
                         "hi": cf.effect_interval(Xc.values)[1]})
    cate.to_csv(os.path.join(FIGD, "fig4_cate.csv"), index=False)
    results["cate"] = {"n": len(cate), "tau_mean": float(tau.mean()),
                       "tau_p10": float(np.quantile(tau, .1)),
                       "tau_p90": float(np.quantile(tau, .9)),
                       "note": "exploratory CausalForestDML, 2018 cross-section"}
except Exception as e:
    results["cate"] = {"status": "NOT IMPLEMENTED", "reason": str(e)[:300]}

results["flow"] = flow
results["table1_ref"] = "table1.csv"
results["note_no_causal_language"] = "所有结果为观察性关联估计"
with open(os.path.join(OUT, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=float)

print("=== 02_analysis.py done ===")
print("flow:", json.dumps(flow, ensure_ascii=False))
print("main INT:", main)
print("GEE:", results["gee_main"])
print("IPTW GEE:", results["iptw_gee"])
print("E-value(residSD):", results["evalue_int_residSD"])
print("maxSMD unw/w:", results["iptw"]["max_smd_unw"], results["iptw"]["max_smd_w"])
print("sens:", [s.get("model", s) for s in sens])
