"""
evaluate.py — 中文文本分类模型评估脚本

功能：
  1. 加载训练好的模型（models/{model_type}_model.pkl）和标签编码器（models/label_encoder.pkl）
  2. 在 data/processed/test.csv 上进行预测
  3. 输出 accuracy、各类别 precision/recall/F1、macro-F1、weighted-F1
  4. 保存以下结果到 results/{exp_name}/：
       - confusion_matrix.png    混淆矩阵热力图
       - classification_report.json  完整指标 JSON
       - metrics_bar.png         各类别 F1 柱状图

用法：
  cd <项目根目录>
  python code/evaluate.py --config configs/default.yaml
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib

import matplotlib
# 使用非交互式后端，避免无显示器环境报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# 将 code/ 目录加入模块搜索路径，确保能 import utils
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.config import get_project_root, load_config  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 中文字体配置
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> None:
    """
    配置 matplotlib 中文字体。
    优先级：项目字体目录 > 系统常见中文字体 > 回退到英文。
    """
    # 候选字体列表（覆盖 macOS / Linux / Windows 常用中文字体）
    candidates = [
        # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        # Linux（文泉驿、思源等）
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            try:
                prop = fm.FontProperties(fname=font_path)
                plt.rcParams["font.family"] = prop.get_family()
                plt.rcParams["font.sans-serif"] = [prop.get_name()] + plt.rcParams.get(
                    "font.sans-serif", []
                )
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue

    # 若未找到任何中文字体，使用 matplotlib 内置字体并关闭负号警告
    plt.rcParams["axes.unicode_minus"] = False
    print("[警告] 未找到中文字体，图表中文标签可能显示为方块。")


# ---------------------------------------------------------------------------
# 数据与模型加载
# ---------------------------------------------------------------------------

def load_test_data(cfg: dict, root: Path) -> pd.DataFrame:
    """读取测试集 CSV，返回 DataFrame。"""
    processed_dir = root / cfg["data"]["processed_dir"]
    test_file = processed_dir / cfg["data"]["test_file"]

    if not test_file.exists():
        raise FileNotFoundError(f"测试集文件不存在：{test_file}")

    df = pd.read_csv(test_file, encoding="utf-8")
    # 预处理输出列名为 "text"（tokens 已分词），config 里的 text_column 是原始字段名
    # 优先使用已处理的 "tokens" 列（分词后，供模型预测），fallback 到 "text" 或 config 指定列
    if "tokens" in df.columns:
        text_col = "tokens"
    elif "text" in df.columns:
        text_col = "text"
    else:
        text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"] if cfg["data"]["label_column"] in df.columns else "label"

    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"测试集缺少列 '{col}'，当前列：{list(df.columns)}")

    # 去除空行
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    # 将实际使用的列名写回 cfg（供 evaluate() 使用）
    cfg["data"]["_text_col"] = text_col
    cfg["data"]["_label_col"] = label_col
    print(f"[数据] 测试集样本数：{len(df)}  (text列={text_col}, label列={label_col})")
    return df


def load_model_and_encoder(cfg: dict, root: Path):
    """
    加载训练好的流水线模型与标签编码器。

    返回：
        pipeline  sklearn Pipeline 对象（TF-IDF + 分类器）
        le        LabelEncoder 对象
    """
    model_dir = root / cfg["output"]["model_dir"]
    model_type = cfg["train"]["model_type"]

    model_path = model_dir / f"{model_type}_model.pkl"
    le_path = model_dir / "label_encoder.pkl"

    for p in (model_path, le_path):
        if not p.exists():
            raise FileNotFoundError(
                f"模型文件不存在：{p}\n"
                "请先运行 train.py 生成模型后再评估。"
            )

    pipeline = joblib.load(model_path)
    le = joblib.load(le_path)

    print(f"[模型] 已加载：{model_path.name}")
    print(f"[标签编码器] 类别数：{len(le.classes_)}  →  {list(le.classes_)}")
    return pipeline, le


# ---------------------------------------------------------------------------
# 评估逻辑
# ---------------------------------------------------------------------------

def evaluate(pipeline, le, df: pd.DataFrame, cfg: dict) -> dict:
    """
    对测试集做预测并计算所有评估指标。

    返回：
        results dict，包含 y_true、y_pred（字符串标签）及所有指标。
    """
    text_col = cfg["data"].get("_text_col", cfg["data"]["text_column"])
    label_col = cfg["data"].get("_label_col", cfg["data"]["label_column"])

    texts = df[text_col].astype(str).tolist()
    y_true_str = df[label_col].astype(str).tolist()

    # 预测（pipeline 输出为编码后的整数标签）
    y_pred_enc = pipeline.predict(texts)

    # 将真实标签也编码，再统一解码为字符串，保证顺序一致
    # 若真实标签已经是字符串类别名，直接使用；否则先 transform
    try:
        y_true_enc = le.transform(y_true_str)
        y_true_decoded = le.inverse_transform(y_true_enc)
    except Exception:
        # 若测试集标签不在 le 中，直接使用原始字符串
        y_true_decoded = np.array(y_true_str)

    y_pred_decoded = le.inverse_transform(y_pred_enc)

    # 类别列表（按 LabelEncoder 顺序）
    class_names = list(le.classes_)

    # ---- 指标计算 ----
    acc = accuracy_score(y_true_decoded, y_pred_decoded)
    macro_f1 = f1_score(y_true_decoded, y_pred_decoded, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_decoded, y_pred_decoded, average="weighted", zero_division=0)

    # classification_report 返回字典（output_dict=True）
    report_dict = classification_report(
        y_true_decoded,
        y_pred_decoded,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    # 同时保留可读字符串版本
    report_str = classification_report(
        y_true_decoded,
        y_pred_decoded,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true_decoded, y_pred_decoded, labels=class_names)

    print("\n" + "=" * 60)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro-F1       : {macro_f1:.4f}")
    print(f"  Weighted-F1    : {weighted_f1:.4f}")
    print("=" * 60)
    print("\nClassification Report:\n")
    print(report_str)

    return {
        "y_true": y_true_decoded,
        "y_pred": y_pred_decoded,
        "class_names": class_names,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report_dict": report_dict,
        "report_str": report_str,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# 可视化：混淆矩阵
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
    exp_name: str = "",
) -> None:
    """
    绘制并保存混淆矩阵热力图。

    参数：
        cm          混淆矩阵（numpy 二维数组）
        class_names 类别名称列表
        save_path   保存路径
        exp_name    实验名称（显示在标题中）
    """
    n = len(class_names)
    # 根据类别数动态调整图像大小
    fig_size = max(8, n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # 归一化混淆矩阵（行归一化，显示召回率视角）
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_norm = cm_norm / row_sums

    # 热力图：annot 显示原始计数
    annot = np.array(
        [[f"{cm[i, j]}\n({cm_norm[i, j]:.2f})" for j in range(n)] for i in range(n)]
    )

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )

    title = "混淆矩阵"
    if exp_name:
        title = f"{title}  [{exp_name}]"
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] 混淆矩阵图：{save_path}")


# ---------------------------------------------------------------------------
# 可视化：各类别 F1 柱状图
# ---------------------------------------------------------------------------

def plot_metrics_bar(
    report_dict: dict,
    class_names: list,
    save_path: Path,
    exp_name: str = "",
) -> None:
    """
    绘制各类别 precision / recall / F1 柱状图。

    参数：
        report_dict classification_report 的字典输出
        class_names 类别名称列表（不含 accuracy / macro avg 等汇总行）
        save_path   保存路径
        exp_name    实验名称（显示在标题中）
    """
    # 提取每个类别的三项指标
    precisions, recalls, f1s = [], [], []
    valid_names = []

    for name in class_names:
        if name in report_dict:
            row = report_dict[name]
            precisions.append(row.get("precision", 0.0))
            recalls.append(row.get("recall", 0.0))
            f1s.append(row.get("f1-score", 0.0))
            valid_names.append(name)

    if not valid_names:
        print("[警告] report_dict 中未找到任何类别指标，跳过柱状图绘制。")
        return

    n = len(valid_names)
    x = np.arange(n)
    bar_width = 0.25

    fig_width = max(10, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    bars_p = ax.bar(x - bar_width, precisions, bar_width, label="Precision", color="#4C72B0", alpha=0.85)
    bars_r = ax.bar(x,             recalls,    bar_width, label="Recall",    color="#55A868", alpha=0.85)
    bars_f = ax.bar(x + bar_width, f1s,        bar_width, label="F1-score",  color="#C44E52", alpha=0.85)

    # 在柱顶标注数值
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # 添加 macro 平均线（使用 F1）
    macro_avg = report_dict.get("macro avg", {})
    macro_f1_val = macro_avg.get("f1-score", None)
    if macro_f1_val is not None:
        ax.axhline(
            y=macro_f1_val,
            color="#C44E52",
            linestyle="--",
            linewidth=1.2,
            label=f"Macro F1 均值 = {macro_f1_val:.4f}",
        )

    title = "各类别分类指标"
    if exp_name:
        title = f"{title}  [{exp_name}]"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("类别", fontsize=11)
    ax.set_ylabel("分数", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=40, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] 指标柱状图：{save_path}")


# ---------------------------------------------------------------------------
# 保存 JSON 指标
# ---------------------------------------------------------------------------

def save_report_json(
    report_dict: dict,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
    save_path: Path,
) -> None:
    """
    将完整指标写入 JSON 文件。

    JSON 结构：
    {
        "summary": { "accuracy": ..., "macro_f1": ..., "weighted_f1": ... },
        "per_class": { "<类别>": {"precision": ..., "recall": ..., "f1-score": ..., "support": ...}, ... },
        "macro avg": { ... },
        "weighted avg": { ... }
    }
    """
    # 将 numpy float 转换为 Python float，确保 JSON 可序列化
    def _to_python(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _convert_dict(d):
        if isinstance(d, dict):
            return {k: _convert_dict(v) for k, v in d.items()}
        return _to_python(d)

    output = {
        "summary": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "classification_report": _convert_dict(report_dict),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"[保存] 指标 JSON：{save_path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    # ---- 加载配置 ----
    cfg = load_config(config_path)
    root = get_project_root()

    model_type = cfg["train"]["model_type"]
    results_dir = root / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # 从 results_dir 路径中提取实验名称（最后一个目录段）
    exp_name = results_dir.name

    print(f"[配置] 实验名称   : {exp_name}")
    print(f"[配置] 模型类型   : {model_type}")
    print(f"[配置] 结果输出目录: {results_dir}")

    # ---- 设置中文字体 ----
    _setup_chinese_font()

    # ---- 加载数据 ----
    df = load_test_data(cfg, root)

    # ---- 加载模型 ----
    pipeline, le = load_model_and_encoder(cfg, root)

    # ---- 评估 ----
    results = evaluate(pipeline, le, df, cfg)

    # ---- 保存指标 JSON ----
    save_report_json(
        report_dict=results["report_dict"],
        accuracy=results["accuracy"],
        macro_f1=results["macro_f1"],
        weighted_f1=results["weighted_f1"],
        save_path=results_dir / "classification_report.json",
    )

    # ---- 绘制混淆矩阵 ----
    plot_confusion_matrix(
        cm=results["confusion_matrix"],
        class_names=results["class_names"],
        save_path=results_dir / "confusion_matrix.png",
        exp_name=exp_name,
    )

    # ---- 绘制 F1 柱状图 ----
    plot_metrics_bar(
        report_dict=results["report_dict"],
        class_names=results["class_names"],
        save_path=results_dir / "metrics_bar.png",
        exp_name=exp_name,
    )

    print(f"\n[完成] 所有评估结果已保存至：{results_dir}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="评估中文文本分类模型，生成指标报告与可视化图表。"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML 配置文件路径（默认：configs/default.yaml）",
    )
    args = parser.parse_args()
    main(args.config)
