"""
中文文本分类模型训练入口。

支持三种模型（通过 config 中 train.model_type 指定）：
  - tfidf_nb  : TF-IDF + MultinomialNB（朴素贝叶斯）
  - tfidf_svm : TF-IDF + LinearSVC（支持向量机）
  - tfidf_lr  : TF-IDF + LogisticRegression（逻辑回归）

用法示例：
  cd <项目根目录>
  python code/train.py --config configs/default.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# 将项目 code/ 目录加入 sys.path，使 utils 模块可被直接导入
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.config import get_project_root, load_config  # noqa: E402


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    读取 CSV 文件，校验必要列是否存在，并做基础清洗。

    期望列：text、tokens、label
    tokens 列格式：分词后以空格分隔的字符串，例如 "手机 质量 不错 很 满意"
    """
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    df = pd.read_csv(file_path, encoding="utf-8")

    required_columns = {"text", "tokens", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"数据文件缺少列：{missing}，当前列：{list(df.columns)}")

    # 去除空行（tokens 或 label 为空的行）
    before = len(df)
    df = df.dropna(subset=["tokens", "label"])
    after = len(df)
    if before != after:
        print(f"  [警告] 删除了 {before - after} 行含空值的数据")

    # tokens 转为字符串（防止类型异常）
    df["tokens"] = df["tokens"].astype(str)

    print(f"  加载完成：{after} 条样本，{df['label'].nunique()} 个类别")
    return df


# ──────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────

def build_pipeline(model_type: str, max_features: int, ngram_range: tuple) -> Pipeline:
    """
    根据 model_type 构建 sklearn Pipeline（TF-IDF + 分类器）。

    Parameters
    ----------
    model_type   : "tfidf_nb" | "tfidf_svm" | "tfidf_lr"
    max_features : TF-IDF 词表最大大小
    ngram_range  : TF-IDF n-gram 范围，例如 (1, 2)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    # TF-IDF 向量化器
    # analyzer="word" + 输入已是空格分隔的 tokens，无需内部分词
    tfidf = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",   # 以空白符分割 token，支持中文词
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # 使用 1+log(tf) 平滑，缓解高频词主导
    )

    if model_type == "tfidf_nb":
        # MultinomialNB 要求非负输入；TF-IDF 默认输出非负，可直接使用
        classifier = MultinomialNB(alpha=1.0)
    elif model_type == "tfidf_svm":
        # LinearSVC 速度快、效果好，适合高维稀疏文本特征
        classifier = LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
        )
    elif model_type == "tfidf_lr":
        # LogisticRegression 提供概率输出，便于后续分析
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"不支持的模型类型：{model_type}。"
            f"请从 [tfidf_nb, tfidf_svm, tfidf_lr] 中选择。"
        )

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", classifier),
    ])
    return pipeline


# ──────────────────────────────────────────────
# 训练与评估
# ──────────────────────────────────────────────

def train_and_evaluate(
    pipeline: Pipeline,
    X_train: list,
    y_train: np.ndarray,
    X_val: list,
    y_val: np.ndarray,
    label_names: list,
) -> float:
    """
    训练 pipeline 并在验证集上评估，打印详细报告。

    Returns
    -------
    val_accuracy : float
    """
    print("\n[训练] 开始训练...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[训练] 完成，耗时 {elapsed:.2f}s")

    # 验证集预测
    y_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)

    print(f"\n[评估] 验证集 Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
    print("\n[评估] 分类报告：")
    print(classification_report(y_val, y_pred, target_names=label_names, digits=4))

    return val_accuracy


# ──────────────────────────────────────────────
# 模型保存
# ──────────────────────────────────────────────

def save_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    model_dir: Path,
    model_type: str,
) -> None:
    """
    将训练好的 pipeline 与 label_encoder 序列化到磁盘。

    保存路径：
      models/{model_type}_model.pkl
      models/label_encoder.pkl
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_type}_model.pkl"
    encoder_path = model_dir / "label_encoder.pkl"

    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"\n[保存] 模型已保存至：{model_path}")
    print(f"[保存] 标签编码器已保存至：{encoder_path}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main(config_path: str) -> None:
    # 1. 加载配置
    cfg = load_config(config_path)
    root = get_project_root()

    # 从 config 读取各项参数
    model_type: str = cfg["train"]["model_type"]
    max_features: int = int(cfg["train"]["max_features"])
    ngram_range: tuple = tuple(cfg["train"]["ngram_range"])  # YAML 列表 → tuple
    seed: int = cfg["project"].get("seed", 42)

    processed_dir = root / cfg["data"]["processed_dir"]
    train_file = processed_dir / cfg["data"]["train_file"]
    val_file = processed_dir / cfg["data"]["val_file"]
    model_dir = root / cfg["output"]["model_dir"]

    print("=" * 60)
    print(f"  NLP 文本分类训练")
    print(f"  model_type   : {model_type}")
    print(f"  max_features : {max_features}")
    print(f"  ngram_range  : {ngram_range}")
    print(f"  seed         : {seed}")
    print("=" * 60)

    # 2. 加载数据
    print("\n[数据] 加载训练集...")
    df_train = load_dataset(train_file)

    print("[数据] 加载验证集...")
    df_val = load_dataset(val_file)

    # 3. 标签编码
    # 在全量标签上 fit，保证训练集/验证集类别一致
    label_encoder = LabelEncoder()
    all_labels = pd.concat([df_train["label"], df_val["label"]], ignore_index=True)
    label_encoder.fit(all_labels)

    y_train = label_encoder.transform(df_train["label"])
    y_val = label_encoder.transform(df_val["label"])
    label_names = list(label_encoder.classes_)

    print(f"\n[标签] 共 {len(label_names)} 个类别：{label_names}")
    print(f"[数据] 训练集大小：{len(df_train)}，验证集大小：{len(df_val)}")

    # 4. 构建 TF-IDF + 分类器 Pipeline
    # 使用 tokens 列（已分词的空格分隔字符串）作为输入
    X_train = df_train["tokens"].tolist()
    X_val = df_val["tokens"].tolist()

    print(f"\n[模型] 构建 Pipeline：TF-IDF -> {model_type}")
    pipeline = build_pipeline(model_type, max_features, ngram_range)

    # 5. 训练 + 评估
    val_accuracy = train_and_evaluate(
        pipeline, X_train, y_train, X_val, y_val, label_names
    )

    # 6. 保存模型与标签编码器
    save_artifacts(pipeline, label_encoder, model_dir, model_type)

    print(f"\n[完成] 验证集最终 Accuracy = {val_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="中文文本分类模型训练（tfidf_nb / tfidf_svm / tfidf_lr）"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="配置文件路径，支持相对于项目根目录的路径（默认：configs/default.yaml）",
    )
    args = parser.parse_args()
    main(args.config)
