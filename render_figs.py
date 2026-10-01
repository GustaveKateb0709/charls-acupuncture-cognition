# 渲染四图三件套（TIFF 600dpi RGB LZW + PDF + PNG 200dpi）
# 数字全部来自 flow.json / fig_data/*.csv；图内禁标题；Set2 低饱和配色
import json, io, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PKG = "/Users/wangtianyu/Desktop/25_2026.9.26_针灸神经康复SCI（期刊待定）"
D = os.path.join(PKG, "04_Code_and_Data", "data_out")
OUT = os.path.join(PKG, "02_Figures")
flow = json.load(open(os.path.join(D, "flow.json")))

# Set2 低饱和
C1, C2, CG = "#66c2a5", "#fc8d62", "#8da0cb"   # 主色/次色/灰蓝
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 9, "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "pdf.fonttype": 42,
})

MM = 1 / 25.4
def save3(fig, stem):
    for ext, kw in [("pdf", {}), ("png", dict(dpi=200))]:
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), **kw)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, facecolor="white")
    buf.seek(0)
    im = Image.open(buf)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
    bg.save(os.path.join(OUT, f"{stem}.tiff"), compression="tiff_lzw", dpi=(600, 600))
    print(stem, "saved", bg.size)

# ---------------- Fig 1 STROBE 流程 ----------------
n_panel, n_pain, n_cog, n_inc = (flow["panel_individuals"], flow["n_pain_any"],
                                 flow["n_cog2"], flow["n_included"])
rows_cc, pers_cc, n_acu = flow["model_rows_cc"], flow["n_cc_persons"], flow["n_baseline_acu"]
fig, ax = plt.subplots(figsize=(150*MM, 105*MM))
ax.axis("off")
ax.set_xlim(0, 118); ax.set_ylim(0, 100)

def box(x, y, w, h, text, fs=8.5, lw=0.9, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#f7f7f7",
                               edgecolor="#4d4d4d", lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(x0, y0, y1, x1=None):
    if x1 is None:
        ax.annotate("", xy=(x0, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#4d4d4d", lw=0.9))
    else:
        ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#4d4d4d", lw=0.9))

bx, bw = 12, 52
ex = bx + bw + 12
box(bx, 84, bw, 13, f"CHARLS 2013, 2015, 2018\n{n_panel:,} individuals / 59,174 person-waves")
box(bx, 64, bw, 13, f"Reported pain at ≥1 wave\nn = {n_pain:,}")
box(bx, 44, bw, 13, f"≥2 waves with cognitive assessment\nn = {n_cog:,}")
box(bx, 24, bw, 13, f"Age ≥ 45 at baseline\nn = {n_inc:,} (analysis sample)")
box(bx, 2, bw, 15, f"Complete-case analysis\n{rows_cc:,} person-waves / {pers_cc:,} individuals\nbaseline acupuncture n = {n_acu:,}")
arrow(bx + bw/2, 84, 77)
arrow(bx + bw/2, 64, 57)
arrow(bx + bw/2, 44, 37)
arrow(bx + bw/2, 24, 17)
box(ex, 70, 40, 11, f"Excluded:\nno pain report (n = {n_panel - n_pain:,})", fs=7.5)
box(ex, 50, 40, 11, f"Excluded:\n<2 cognitive waves (n = {n_pain - n_cog:,})", fs=7.5)
box(ex, 30, 40, 11, f"Excluded:\nage < 45 (n = {n_cog - n_inc:,})", fs=7.5)
arrow(bx + bw, 75.5, 75.5, x1=ex)
arrow(bx + bw, 55.5, 55.5, x1=ex)
arrow(bx + bw, 35.5, 35.5, x1=ex)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
save3(fig, "fig1_cohort_flow")

# ---------------- Fig 2 调整后轨迹 ----------------
t2 = pd.read_csv(os.path.join(D, "fig_data", "fig2_adjusted_trajectories.csv"))
year = {0.0: 2013, 2.0: 2015, 5.0: 2018}
fig, ax = plt.subplots(figsize=(115*MM, 88*MM))
for v, c, lab in [(0, C1, "No acupuncture use"), (1, C2, "Acupuncture use")]:
    g = t2[t2.acu_w == v].sort_values("time")
    x = g.time.map(year)
    ax.fill_between(x, g.lo, g.hi, color=c, alpha=0.25, lw=0)
    ax.plot(x, g.pred, "-o", color=c, ms=4, lw=1.6, label=lab)
ax.set_xlabel("Survey year"); ax.set_ylabel("Adjusted cognition score\n(marginal mean, 0–20 scale)")
ax.set_xticks([2013, 2015, 2018]); ax.set_xlim(2012.3, 2018.7)
ax.set_ylim(0, 20); ax.set_yticks(range(0, 21, 5))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(pad=0.6)
save3(fig, "fig2_trajectories")

# ---------------- Fig 3 森林 ----------------
t3 = pd.read_csv(os.path.join(D, "fig_data", "fig3_forest.csv"))
order = ["LMM main", "sens1_drop2013", "sens3_pattern2015_18",
         "sens4_tcm", "sens4_massage", "sens5_surv2018", "sens2_MI_m10"]
t3["k"] = t3.model.map({m: i for i, m in enumerate(order)})
t3 = t3.sort_values("k", ascending=False).reset_index(drop=True)
lab = {"LMM main": "Main LMM (time × acupuncture)",
       "sens1_drop2013": "S1: excluding 2013 wave",
       "sens3_pattern2015_18": "S3: 2015+2018 pattern (occasional)",
       "sens4_tcm": "S4: exposure = TCM use",
       "sens4_massage": "S4: exposure = massage use",
       "sens5_surv2018": "S5: survivors to 2018",
       "sens2_MI_m10": "S2: multiple imputation (m = 10)"}
fig, ax = plt.subplots(figsize=(140*MM, 88*MM))
for i, r in t3.iterrows():
    c = C2 if r.model == "LMM main" else CG
    ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=1.6, solid_capstyle="butt")
    ax.plot(r.beta, i, "o", color=c, ms=5, mec="white", mew=0.6, zorder=3)
ax.axvline(0, color="#4d4d4d", lw=0.8, ls="--")
ax.set_yticks(range(len(t3)))
ax.set_yticklabels([lab[m] for m in t3.model])
ax.set_xlabel("Interaction coefficient β (time × exposure)")
ax.set_ylim(-0.6, len(t3) - 0.4)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.6)
save3(fig, "fig3_forest")

# ---------------- Fig 4 CATE 分布 ----------------
t4 = pd.read_csv(os.path.join(D, "fig_data", "fig4_cate.csv"))
tau = t4.tau.values
mean_, p10, p90 = np.mean(tau), np.percentile(tau, 10), np.percentile(tau, 90)
fig, ax = plt.subplots(figsize=(115*MM, 88*MM))
ax.hist(tau, bins=60, color=C1, alpha=0.85, edgecolor="white", lw=0.3)
ax.set_ylim(0, np.histogram(tau, bins=60)[0].max() * 1.32)   # 顶部净空带
ax.axvline(mean_, color=C2, lw=1.6, ymax=0.82)   # 线在 82% 高度截断，标签悬浮于其上
ax.axvline(p10, color=CG, lw=1.0, ls="--"); ax.axvline(p90, color=CG, lw=1.0, ls="--")
ax.annotate(f"mean = {mean_:.3f}", xy=(mean_, 0.84), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", color="#b3581e", fontsize=8.5)
ax.annotate(f"p10 = {p10:.3f}", xy=(p10, 0.97), xycoords=("data", "axes fraction"),
            xytext=(-6, -14), textcoords="offset points", ha="right", color="#5a6e8c", fontsize=8)
ax.annotate(f"p90 = {p90:.3f}", xy=(p90, 0.97), xycoords=("data", "axes fraction"),
            xytext=(6, -14), textcoords="offset points", color="#5a6e8c", fontsize=8)
ax.set_xlabel("Individual estimated treatment-effect difference (CATE, points)")
ax.set_ylabel("Number of individuals")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.6)
save3(fig, "fig4_cate")
print("CHECK", round(mean_, 4), round(p10, 4), round(p90, 4), len(tau))
