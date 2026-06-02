"""
推理入口：中文文本分类预测

支持两种运行模式：
  1. 单条预测：
       python predict.py --text "这款手机质量很好，性价比超高"
  2. 批量预测（CSV 文件）：
       python predict.py --file data/raw/test.csv

可选参数：
  --config   配置文件路径（默认 configs/default.yaml）
  --model    模型类型，可选 tfidf_svm / tfidf_nb / tfidf_lr
             （默认读取配置文件中的 train.model_type）
  --output   批量预测结果保存路径（默认在输入文件同目录下生成 *_predicted.csv）
  --top-k    单条预测时输出 top-k 类别及其置信度（默认 1）

依赖：
  pip install jieba scikit-learn pyyaml pandas
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 确保从 code/ 目录可以导入 utils，兼容直接运行和包导入两种方式
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import pickle
import warnings

import jieba
import numpy as np
import pandas as pd

from utils.config import get_project_root, load_config

# 屏蔽 sklearn 版本不匹配时的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 文本清洗与分词（与 preprocess.py 保持一致）
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    简单清洗：
      - 统一为字符串
      - 去除 HTML 标签
      - 去除纯 ASCII 标点与控制字符，保留中文、数字、字母
    """
    if not isinstance(text, str):
        text = str(text)
    # 去除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 只保留中文字符、英文字母、数字及常见标点（一-鿿 覆盖常用汉字区）
    text = re.sub(r"[^一-鿿　-〿＀-￯a-zA-Z0-9]", " ", text)
    # 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, use_jieba: bool = True) -> str:
    """
    对单条文本做 jieba 分词，返回空格连接的词串，
    以便直接送入 sklearn 的 TfidfVectorizer（analyzer='word'）。
    """
    text = clean_text(text)
    if not text:
        return ""
    if use_jieba:
        # jieba.cut 返回生成器，join 后作为"分好词的句子"
        tokens = jieba.cut(text, cut_all=False)
        return " ".join(t for t in tokens if t.strip())
    else:
        # 不使用 jieba 时，按字符切分（字符级 n-gram 场景）
        return " ".join(list(text))


# ---------------------------------------------------------------------------
# 模型与编码器加载
# ---------------------------------------------------------------------------

def load_artifacts(model_dir: Path, model_type: str):
    """
    加载训练产物：
      - {model_dir}/{model_type}_model.pkl  ：sklearn Pipeline（含 TF-IDF + 分类器）
      - {model_dir}/label_encoder.pkl       ：LabelEncoder

    Returns
    -------
    pipeline : sklearn Pipeline
    label_encoder : sklearn LabelEncoder
    """
    model_path = model_dir / f"{model_type}_model.pkl"
    encoder_path = model_dir / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}\n"
            f"请先运行 train.py 训练模型，或确认 --model 参数与已保存文件名一致。"
        )
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"找不到标签编码器：{encoder_path}\n"
            "请先运行 train.py 生成 label_encoder.pkl。"
        )

    with model_path.open("rb") as f:
        pipeline = pickle.load(f)
    with encoder_path.open("rb") as f:
        label_encoder = pickle.load(f)

    return pipeline, label_encoder


# ---------------------------------------------------------------------------
# 概率 / 置信度提取
# ---------------------------------------------------------------------------

def get_confidence(pipeline, tokenized_texts: list[str]) -> np.ndarray:
    """
    返回形状 (n_samples, n_classes) 的置信度矩阵。

    - 若分类器支持 predict_proba（NB、LR、SVM with probability=True），
      直接返回概率。
    - 否则使用 decision_function，并用 softmax 归一化为伪概率，
      方便统一解读（SVM 默认情形）。
    """
    # 先尝试 predict_proba
    try:
        scores = pipeline.predict_proba(tokenized_texts)
        return scores  # 真实概率
    except AttributeError:
        pass

    # 回退到 decision_function
    scores = pipeline.decision_function(tokenized_texts)  # (n, n_classes) or (n,)
    if scores.ndim == 1:
        # 二分类时 decision_function 返回 (n,)，扩展为 (n, 2)
        scores = np.column_stack([-scores, scores])

    # softmax 归一化
    scores = scores - scores.max(axis=1, keepdims=True)  # 数值稳定
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs


# ---------------------------------------------------------------------------
# 单条预测
# ---------------------------------------------------------------------------

def predict_single(
    text: str,
    pipeline,
    label_encoder,
    use_jieba: bool = True,
    top_k: int = 1,
) -> None:
    """
    对单条文本进行推理，打印预测类别与置信度。

    Parameters
    ----------
    text        : 原始中文文本
    pipeline    : 已加载的 sklearn Pipeline
    label_encoder : 已加载的 LabelEncoder
    use_jieba   : 是否使用 jieba 分词
    top_k       : 输出前 top_k 个候选类别（按置信度降序）
    """
    tokenized = tokenize(text, use_jieba=use_jieba)
    if not tokenized:
        print("[警告] 文本经清洗后为空，无法预测。")
        return

    # sklearn Pipeline 要求输入为列表
    probs = get_confidence(pipeline, [tokenized])  # (1, n_classes)
    prob_row = probs[0]

    # 获取类别名称（label_encoder.classes_ 顺序与分类器类别顺序一致）
    classes = label_encoder.classes_

    # top-k 排序
    top_k = min(top_k, len(classes))
    top_indices = np.argsort(prob_row)[::-1][:top_k]

    print(f"\n输入文本：{text}")
    print("-" * 50)
    print(f"{'排名':<4} {'类别':<15} {'置信度':>8}")
    print("-" * 50)
    for rank, idx in enumerate(top_indices, start=1):
        label = classes[idx]
        conf = prob_row[idx]
        marker = " <-- 预测结果" if rank == 1 else ""
        print(f"{rank:<4} {label:<15} {conf:>8.4f}{marker}")
    print("-" * 50)


# ---------------------------------------------------------------------------
# 批量预测（CSV）
# ---------------------------------------------------------------------------

def predict_file(
    input_path: Path,
    output_path: Path,
    pipeline,
    label_encoder,
    text_col: str,
    use_jieba: bool = True,
) -> None:
    """
    读取 CSV 文件，逐行预测，将预测标签与置信度写入新列后保存。

    新增列：
      - predicted_label  ：预测的类别名称
      - confidence       ：最高类别的置信度

    Parameters
    ----------
    input_path  : 输入 CSV 路径
    output_path : 输出 CSV 路径
    pipeline    : 已加载的 sklearn Pipeline
    label_encoder : 已加载的 LabelEncoder
    text_col    : 文本列名（从配置读取）
    use_jieba   : 是否使用 jieba 分词
    """
    print(f"读取文件：{input_path}")
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="gbk")

    # 自动探测文本列
    if text_col not in df.columns:
        # 尝试常见列名
        candidates = ["text", "review", "content", "comment", "sentence", "文本", "内容"]
        found = [c for c in candidates if c in df.columns]
        if found:
            text_col = found[0]
            print(f"[提示] 配置中的文本列 '{text_col}' 不存在，自动使用列：'{text_col}'")
        else:
            raise ValueError(
                f"找不到文本列。CSV 列名：{list(df.columns)}\n"
                f"请通过配置文件的 data.text_column 指定正确的列名。"
            )

    print(f"共 {len(df)} 条样本，文本列：'{text_col}'")

    # 分词
    print("正在分词……")
    tokenized = df[text_col].fillna("").apply(
        lambda t: tokenize(str(t), use_jieba=use_jieba)
    ).tolist()

    # 推理
    print("正在推理……")
    probs = get_confidence(pipeline, tokenized)   # (n, n_classes)
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = label_encoder.classes_[pred_indices]
    confidences = probs[np.arange(len(probs)), pred_indices]

    df["predicted_label"] = pred_labels
    df["confidence"] = np.round(confidences, 6)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")  # utf-8-sig 兼容 Excel
    print(f"预测完成，结果已保存至：{output_path}")

    # 简要统计
    print("\n--- 预测分布统计 ---")
    print(df["predicted_label"].value_counts().to_string())
    print(f"\n平均置信度：{confidences.mean():.4f}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="中文文本分类推理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单条预测
  python predict.py --text "这款手机拍照效果非常棒，物超所值"

  # 单条预测，输出 top-3 候选类别
  python predict.py --text "送货速度很快，包装完好" --top-k 3

  # 批量预测
  python predict.py --file data/raw/test.csv

  # 指定模型类型与配置文件
  python predict.py --text "xxx" --model tfidf_lr --config configs/default.yaml
        """,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML 配置文件路径（默认：configs/default.yaml）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型类型，可选 tfidf_svm / tfidf_nb / tfidf_lr（默认读取配置文件）",
    )

    # 互斥组：--text 和 --file 只能二选一
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--text",
        metavar="TEXT",
        help="单条中文文本（用引号包裹）",
    )
    mode_group.add_argument(
        "--file",
        metavar="CSV_PATH",
        help="批量预测的 CSV 文件路径",
    )

    parser.add_argument(
        "--output",
        default=None,
        metavar="OUTPUT_PATH",
        help="批量预测结果保存路径（默认：输入文件同目录下的 *_predicted.csv）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        dest="top_k",
        help="单条预测时展示前 k 个候选类别（默认：1）",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 加载配置
    # ------------------------------------------------------------------
    root = get_project_root()
    cfg = load_config(args.config)

    model_type = args.model or cfg["train"]["model_type"]
    model_dir = root / cfg["output"]["model_dir"]
    text_col = cfg["data"].get("text_column", "review")
    use_jieba = cfg["preprocess"].get("use_jieba", True)

    print(f"模型类型：{model_type}")
    print(f"模型目录：{model_dir}")

    # ------------------------------------------------------------------
    # 加载模型与标签编码器
    # ------------------------------------------------------------------
    pipeline, label_encoder = load_artifacts(model_dir, model_type)
    print(f"类别列表：{list(label_encoder.classes_)}\n")

    # ------------------------------------------------------------------
    # 执行预测
    # ------------------------------------------------------------------
    if args.text is not None:
        # 模式一：单条预测
        predict_single(
            text=args.text,
            pipeline=pipeline,
            label_encoder=label_encoder,
            use_jieba=use_jieba,
            top_k=args.top_k,
        )

    else:
        # 模式二：批量 CSV 预测
        input_path = Path(args.file)
        if not input_path.is_absolute():
            input_path = root / input_path
        if not input_path.exists():
            print(f"[错误] 输入文件不存在：{input_path}", file=sys.stderr)
            sys.exit(1)

        if args.output is not None:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = root / output_path
        else:
            # 默认保存在输入文件同目录，文件名加 _predicted 后缀
            output_path = input_path.parent / (input_path.stem + "_predicted.csv")

        predict_file(
            input_path=input_path,
            output_path=output_path,
            pipeline=pipeline,
            label_encoder=label_encoder,
            text_col=text_col,
            use_jieba=use_jieba,
        )


if __name__ == "__main__":
    main()
