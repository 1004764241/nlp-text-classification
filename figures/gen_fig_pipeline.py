#!/usr/bin/env python3
"""
Generate system pipeline figure (academic-plotting Workflow 2 — matplotlib).
Style: Classic Accent Bar (Style D) — safe for any Chinese journal venue.
Palette: Ocean Dusk
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Publication defaults (重邮学报 双栏, 全宽约 170mm ≈ 6.7in) ──────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "SimHei",
                        "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Ocean Dusk palette ────────────────────────────────────────────────────────
C_DEEP   = "#264653"   # 深青 — stage背景/文字
C_TEAL   = "#2A9D8F"   # 青绿 — 预处理
C_GOLD   = "#E9C46A"   # 金   — 特征工程
C_SANDY  = "#F4A261"   # 橙   — 模型训练
C_CORAL  = "#E76F51"   # 珊瑚 — 评估
C_BG     = "#F7F7F5"   # 浅灰背景
C_ARROW  = "#6B7280"   # 箭头灰
C_WHITE  = "#FFFFFF"

fig, ax = plt.subplots(figsize=(7.0, 1.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.2)
ax.axis("off")
fig.patch.set_facecolor(C_WHITE)

# ── 定义各阶段 ────────────────────────────────────────────────────────────────
stages = [
    # (x_left, label_top, label_bot, accent_color, bg_color)
    (0.10, "原始数据",    "CSV文件",      C_DEEP,  "#EAF0F1"),
    (2.10, "数据预处理",  "jieba分词",    C_TEAL,  "#E8F5F3"),
    (4.10, "TF-IDF",     "特征提取",     C_GOLD,  "#FDF8EC"),
    (6.10, "分类器训练",  "LinearSVC",   C_SANDY, "#FEF5EF"),
    (8.10, "评估&可视化", "指标/图表",    C_CORAL, "#FDF0EC"),
]

BOX_W, BOX_H = 1.75, 1.1
BOX_Y = 0.55  # bottom y of boxes

for (x, top, bot, accent, bg) in stages:
    # 主框
    box = FancyBboxPatch((x, BOX_Y), BOX_W, BOX_H,
                         boxstyle="round,pad=0.04",
                         linewidth=0,
                         facecolor=bg)
    ax.add_patch(box)
    # 左accent条
    bar = plt.Rectangle((x, BOX_Y), 0.10, BOX_H, color=accent, zorder=3)
    ax.add_patch(bar)
    # 文字
    ax.text(x + 0.10 + (BOX_W - 0.10)/2, BOX_Y + BOX_H*0.62,
            top, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_DEEP, zorder=4)
    ax.text(x + 0.10 + (BOX_W - 0.10)/2, BOX_Y + BOX_H*0.28,
            bot, ha="center", va="center",
            fontsize=7.5, color="#555555", zorder=4)

# ── 阶段间箭头 ────────────────────────────────────────────────────────────────
for i in range(len(stages) - 1):
    x_start = stages[i][0] + BOX_W + 0.02
    x_end   = stages[i+1][0] - 0.04
    y_mid   = BOX_Y + BOX_H / 2
    ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                arrowprops=dict(arrowstyle="-|>",
                                color=C_ARROW,
                                lw=1.3,
                                mutation_scale=10))

# ── 底部小标注：输入/输出文件 ─────────────────────────────────────────────────
labels_io = ["", "train/val/test\n.csv", "50k维稀疏\n向量矩阵", "Pipeline\n.pkl", "混淆矩阵\nJSON报告"]
for i, (x, *_) in enumerate(stages):
    if labels_io[i]:
        ax.text(x + BOX_W/2 + 0.05, BOX_Y - 0.26,
                labels_io[i], ha="center", va="top",
                fontsize=6.2, color="#888888", linespacing=1.3)
        # 小虚线连到框底
        ax.plot([x + BOX_W/2 + 0.05, x + BOX_W/2 + 0.05],
                [BOX_Y - 0.04, BOX_Y - 0.18],
                color="#CCCCCC", lw=0.7, linestyle="--")

# ── 图题 ─────────────────────────────────────────────────────────────────────
ax.set_title("图1  系统整体处理流程", fontsize=9, fontweight="bold",
             color=C_DEEP, pad=4, loc="center")

plt.tight_layout(pad=0.3)
fig.savefig("figures/fig_pipeline.pdf", bbox_inches="tight")
fig.savefig("figures/fig_pipeline.png", dpi=300, bbox_inches="tight")
print("✅ fig_pipeline saved (PDF + PNG)")
