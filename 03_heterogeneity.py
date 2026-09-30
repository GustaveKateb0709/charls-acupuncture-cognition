#!/usr/bin/env python3
"""03_heterogeneity.py — 因果森林 CATE + SHAP（SAP §10，探索性/假设生成）。

定位红线（写进 Methods/Discussion）：
- 与 Mei 2023（XGBoost+SHAP 横断面预测治疗偏好）明确区分：本文是纵向认知变化速率的
  效应异质性，不做使用偏好预测，不报预测性能指标（AUC 等）。
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_out")
panel = pd.read_pickle(os.path.join(OUT, "analysis_panel.pkl"))

# TODO:
# 1. 效应面构造：个体内认知变化（2013→2018 差值或斜率）为 Y，基线针灸使用/持续使用为 W，
#    基线协变量为 X。
# 2. econml.grf.CausalForestDML 拟合 CATE；报告 CATE 分布与分位组均值差。
# 3. shap.TreeExplainer 解释 CATE 的修饰变量（仅展示，不做预测叙事）。
# 4. 亚组描述按 CATE 分位（Q1–Q4）报告基线特征——检查是否存在"重症疼痛/高共病"等临床可解释亚组。
