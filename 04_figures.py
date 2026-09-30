#!/usr/bin/env python3
"""04_figures.py — 出图（figure-qa 规范）。数据到位前不得执行（无渲染、无假数据）。

规范（硬性）：
- 输出三件套：TIFF（600 dpi, RGB, LZW）+ PDF（矢量）+ PNG（200 dpi 预览）；禁止 EPS。
- 图内禁标题（caption 信息全在正文图注）；面板字母 (a)(b)(c) 放子图下方、x 轴标题之前。
- 配色低饱和（Okabe-Ito 或 Set2 族），禁高饱和强对比；色盲安全。
- 文件命名 = 正文图注编号一致（fig1_*.png/tif/pdf 三方同名）。
- 图注首句 ≤15 词；图内数字与正文与 CSV 三对照。
"""
import os

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_Figures")

PLANNED = {
    "fig1_cohort_flow": "STROBE 流程图（人数来自 02_analysis 的流程表）",
    "fig2_trajectories": "认知总分调整后轨迹（按针灸使用分组，95%CI 阴影；来自主模型边际均值）",
    "fig3_forest": "主效应与亚组森林图（β 与 95%CI）",
    "fig4_cate": "CATE 分布与分位组描述（来自 03_heterogeneity）",
}

def main():
    panel_ok = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_out", "analysis_panel.pkl"))
    if not panel_ok:
        raise SystemExit("analysis_panel 不存在——先跑 01_harmonize.py；数据未到位前禁止出图")
    # TODO: 按 PLANNED 逐张实现（matplotlib；Agg 后端；全局 rcParams 按 04 文件头规范设置）

if __name__ == "__main__":
    main()
