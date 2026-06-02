#!/usr/bin/env python3
"""
Generate publication-quality confusion matrix (academic-plotting Workflow 2).
Style: seaborn heatmap, Times-compatible, 重邮学报 双栏全宽.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import json, sys
from pathlib import Path

# ── 设置中文字体 ──────────────────────────────────────────────────────────────
def _set_cjk_font():
    candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [prop.get_name()] + plt.rcParams.get("font.sans-serif", [])
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

_set_cjk_font()

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── 从 results/ 重建混淆矩阵 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
report_path = ROOT / "results" / "classification_report.json"
with report_path.open(encoding="utf-8") as f:
    report = json.load(f)

# 各类别顺序（与 LabelEncoder.classes_ 一致）
CLASS_NAMES = ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]

# 从 processed/test.csv 重新预测以拿到混淆矩阵 —— 或直接用 sklearn 重跑
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

test_df = pd.read_csv(ROOT / "data" / "processed" / "test.csv", encoding="utf-8")
pipeline = joblib.load(ROOT / "models" / "tfidf_svm_model.pkl")
le       = joblib.load(ROOT / "models" / "label_encoder.pkl")

X_test  = test_df["tokens"].astype(str).tolist()
y_true  = test_df["label"].astype(str).tolist()
y_pred_enc = pipeline.predict(X_test)
y_pred  = le.inverse_transform(y_pred_enc)
y_true_enc = le.transform(y_true)
y_true_dec = le.inverse_transform(y_true_enc)

cm = confusion_matrix(y_true_dec, y_pred, labels=CLASS_NAMES)

# ── 行归一化（召回率视角）────────────────────────────────────────────────────
cm_norm = cm.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm /= row_sums

# ── 标注：计数(归一化率) ──────────────────────────────────────────────────────
n = len(CLASS_NAMES)
annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]:.2f})" for j in range(n)]
                   for i in range(n)])

# ── 绘图 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.6))
fig.patch.set_facecolor("#FFFFFF")

cmap = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(
    cm_norm,
    annot=annot, fmt="",
    cmap=cmap,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=1.2,
    linecolor="#FFFFFF",
    vmin=0.0, vmax=1.0,
    annot_kws={"size": 7, "weight": "medium"},
    cbar_kws={"shrink": 0.80, "aspect": 22,
              "label": "归一化召回率"},
    ax=ax,
)

# 强调对角线（正确预测）
for i in range(n):
    ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                fill=False, edgecolor="#E76F51",
                                lw=1.5, zorder=5))

ax.set_title("图2  TF-IDF + LinearSVC 混淆矩阵（测试集）",
             fontsize=10, fontweight="bold", color="#264653", pad=8)
ax.set_xlabel("预测类别", fontsize=9, labelpad=4)
ax.set_ylabel("真实类别", fontsize=9, labelpad=4)
ax.tick_params(axis="both", labelsize=8)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)

plt.tight_layout(pad=0.5)
fig.savefig("figures/fig_confusion.pdf", bbox_inches="tight")
fig.savefig("figures/fig_confusion.png", dpi=300, bbox_inches="tight")
print("✅ fig_confusion saved (PDF + PNG)")
