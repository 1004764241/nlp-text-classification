#!/usr/bin/env python3
"""
Generate publication-quality per-class metrics grouped bar chart.
academic-plotting Workflow 2 — grouped bar, "our method" highlight style.
Palette: Ocean Dusk (Okabe-Ito for bars, coral for F1 ref line).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
from pathlib import Path

# ── CJK 字体 ─────────────────────────────────────────────────────────────────
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

# ── Publication defaults ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
})

# ── 读取指标数据 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "results" / "classification_report.json").open(encoding="utf-8") as f:
    report = json.load(f)["classification_report"]

CLASS_NAMES = ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]

precisions = [report[c]["precision"] for c in CLASS_NAMES]
recalls    = [report[c]["recall"]    for c in CLASS_NAMES]
f1s        = [report[c]["f1-score"]  for c in CLASS_NAMES]
supports   = [int(report[c]["support"]) for c in CLASS_NAMES]
macro_f1   = report["macro avg"]["f1-score"]

# ── Okabe-Ito 无障碍配色 ──────────────────────────────────────────────────────
C_P     = "#0072B2"   # blue — Precision
C_R     = "#009E73"   # green — Recall
C_F1    = "#E76F51"   # coral — F1（"our highlight" 色）
C_LINE  = "#E76F51"   # macro F1 参考线

# ── 绘图 ─────────────────────────────────────────────────────────────────────
n = len(CLASS_NAMES)
x = np.arange(n)
w = 0.24   # bar width

fig, ax = plt.subplots(figsize=(7.0, 3.2))
fig.patch.set_facecolor("#FFFFFF")

bars_p  = ax.bar(x - w,   precisions, w, label="Precision", color=C_P,  alpha=0.88,
                  edgecolor="white", linewidth=0.5)
bars_r  = ax.bar(x,       recalls,    w, label="Recall",    color=C_R,  alpha=0.88,
                  edgecolor="white", linewidth=0.5)
bars_f1 = ax.bar(x + w,   f1s,        w, label="F1-score",  color=C_F1, alpha=0.92,
                  edgecolor="white", linewidth=0.5)

# 柱顶数值标注（仅 F1）
for bar, v in zip(bars_f1, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{v:.2f}", ha="center", va="bottom",
            fontsize=6.5, color="#333333")

# 特别标注 "热水器" — 最低 F1
idx_hot = CLASS_NAMES.index("热水器")
ax.annotate(f"F1={f1s[idx_hot]:.2f}\n(样本少)",
            xy=(x[idx_hot] + w, f1s[idx_hot]),
            xytext=(x[idx_hot] + w + 0.6, f1s[idx_hot] + 0.12),
            fontsize=6.5, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8),
            ha="left")

# 特别标注 "蒙牛" — 满分
idx_mn = CLASS_NAMES.index("蒙牛")
ax.annotate("F1=1.00",
            xy=(x[idx_mn] + w, f1s[idx_mn]),
            xytext=(x[idx_mn] + w + 0.5, f1s[idx_mn] + 0.04),
            fontsize=6.5, color="#264653",
            arrowprops=dict(arrowstyle="->", color="#264653", lw=0.8),
            ha="left")

# Macro-F1 参考线
ax.axhline(macro_f1, color=C_LINE, linestyle="--", linewidth=1.2, zorder=3,
           label=f"Macro-F1 均值 = {macro_f1:.4f}")

# 次 x 轴：样本数
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(x)
ax2.set_xticklabels([f"n={s}" for s in supports], fontsize=6.2, color="#888888")
ax2.tick_params(axis="x", length=0, pad=1)
ax2.spines["top"].set_visible(False)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=8.5)
ax.set_ylabel("指标分数", fontsize=9)
ax.set_ylim(0, 1.18)
ax.legend(loc="lower right", ncol=4, fontsize=7.5,
          bbox_to_anchor=(1.0, -0.01))

ax.set_title("图3  各类别 Precision / Recall / F1 对比（测试集）",
             fontsize=10, fontweight="bold", color="#264653", pad=16)

plt.tight_layout(pad=0.5)
fig.savefig("figures/fig_metrics.pdf", bbox_inches="tight")
fig.savefig("figures/fig_metrics.png", dpi=300, bbox_inches="tight")
print("✅ fig_metrics saved (PDF + PNG)")
