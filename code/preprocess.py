# -*- coding: utf-8 -*-
"""
数据预处理入口脚本。

将 data/online_shopping_10_cats.csv 处理为模型可用的
train.csv / val.csv / test.csv，保存到 data/processed/。

用法：
    python code/preprocess.py
    python code/preprocess.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 将 code/ 目录加入搜索路径，使 utils 包可被直接找到
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import get_project_root, load_config


# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

# 数据集中 cat 列的全部合法类别（10 类）
VALID_CATS = {"书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"}

# 源文件相对于项目根目录的路径
SOURCE_FILE = "data/online_shopping_10_cats.csv"


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def load_stopwords(stopwords_path: str | Path | None) -> set:
    """
    从文件加载停用词集合。

    若路径为 None 或文件不存在，返回空集合（跳过停用词过滤）。
    停用词文件每行一个词。
    """
    if stopwords_path is None:
        return set()

    path = Path(stopwords_path)
    if not path.exists():
        print(f"[提示] 停用词文件不存在，跳过停用词过滤：{path}")
        return set()

    with path.open(encoding="utf-8") as f:
        words = {line.strip() for line in f if line.strip()}
    print(f"[信息] 已加载停用词 {len(words)} 个：{path}")
    return words


def tokenize(text: str, stopwords: set) -> str:
    """
    对单条文本进行 jieba 分词，去除停用词和空白词元，
    返回以单空格分隔的词元字符串。
    """
    import jieba  # 延迟导入，若未安装仅影响分词步骤

    tokens = jieba.cut(text, cut_all=False)
    filtered = [t for t in tokens if t.strip() and t not in stopwords]
    return " ".join(filtered)


def clean_dataframe(df: pd.DataFrame, min_text_length: int) -> pd.DataFrame:
    """
    清洗原始 DataFrame：
      1. 删除 cat / review 列含空值的行
      2. 删除重复行（以 cat + review 为去重键）
      3. 将 cat 限定为 10 个合法类别
      4. 删除 review 文本过短（< min_text_length 字符）的行
    """
    original_size = len(df)

    # 1. 去除空值
    df = df.dropna(subset=["cat", "review"])
    after_na = len(df)
    print(f"  去除空值：{original_size - after_na} 行  (剩余 {after_na})")

    # 2. 去除重复（保留第一次出现）
    df = df.drop_duplicates(subset=["cat", "review"])
    after_dedup = len(df)
    print(f"  去除重复：{after_na - after_dedup} 行  (剩余 {after_dedup})")

    # 3. 限定合法类别
    df = df[df["cat"].isin(VALID_CATS)].copy()
    after_cat = len(df)
    print(f"  过滤非法类别：{after_dedup - after_cat} 行  (剩余 {after_cat})")

    # 4. 过滤过短文本
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() >= min_text_length].copy()
    after_len = len(df)
    print(f"  过滤短文本(<{min_text_length}字符)：{after_cat - after_len} 行  (剩余 {after_len})")

    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按比例将 df 随机划分为 train / val / test 三个子集。

    划分策略：
      - 先从全量数据中分出 test_ratio 比例作为测试集
      - 再从剩余数据中分出 val_ratio / (1 - test_ratio) 比例作为验证集
      - 余下为训练集

    使用 stratify=label 保证类别分布均匀。
    """
    label_col = df["label"]

    # 第一步：分出测试集
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=label_col,
    )

    # 第二步：从 train_val 中分出验证集
    # 验证集在 train_val 中的占比
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=train_val["label"],
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def print_statistics(
    total: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """打印数据集统计信息：总量、各子集大小、类别分布。"""
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"清洗后总样本数：{total}")
    print(f"  训练集：{len(train)}")
    print(f"  验证集：{len(val)}")
    print(f"  测试集：{len(test)}")

    print("\n各集合类别分布：")
    for name, subset in [("训练集", train), ("验证集", val), ("测试集", test)]:
        print(f"\n  {name}（共 {len(subset)} 条）：")
        dist = subset["label"].value_counts().sort_index()
        for cat, cnt in dist.items():
            pct = cnt / len(subset) * 100
            print(f"    {cat:<6} {cnt:>5} 条  ({pct:.1f}%)")

    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    """预处理主函数。"""
    # ── 1. 加载配置 ──────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    root = get_project_root()

    seed = cfg["project"]["seed"]
    min_text_length = cfg["preprocess"]["min_text_length"]
    val_ratio = cfg["preprocess"]["val_ratio"]
    test_ratio = cfg["preprocess"]["test_ratio"]
    use_jieba = cfg["preprocess"].get("use_jieba", True)
    # 停用词文件路径优先从 data 节读取，兼容写在 preprocess 节的情况
    stopwords_path = (
        cfg["data"].get("stopwords_file", None)
        or cfg["preprocess"].get("stopwords_file", None)
    )

    processed_dir = root / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_file = processed_dir / cfg["data"]["train_file"]
    val_file = processed_dir / cfg["data"]["val_file"]
    test_file = processed_dir / cfg["data"]["test_file"]

    print(f"[配置] 随机种子={seed}  最短文本={min_text_length}字符")
    print(f"[配置] val_ratio={val_ratio}  test_ratio={test_ratio}")
    print(f"[配置] 输出目录：{processed_dir}\n")

    # ── 2. 读取原始数据 ───────────────────────────────────────────────────────
    # 优先使用 config 中 data.raw_file 指定的路径
    raw_file_cfg = cfg["data"].get("raw_file", None)
    if raw_file_cfg:
        source_file = root / raw_file_cfg
    else:
        source_file = root / SOURCE_FILE
        if not source_file.exists():
            # 兼容放在 data/raw/ 下的情况
            raw_dir = cfg["data"].get("raw_dir", "data/raw")
            source_file = root / raw_dir / "online_shopping_10_cats.csv"

    if not source_file.exists():
        raise FileNotFoundError(
            f"找不到原始数据文件，请将 online_shopping_10_cats.csv 放至：\n"
            f"  {root / SOURCE_FILE}"
        )

    print(f"[信息] 读取原始数据：{source_file}")
    df_raw = pd.read_csv(source_file, dtype=str)
    print(f"[信息] 原始行数：{len(df_raw)}")

    # 检查必要列是否存在
    required_cols = {"cat", "review"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"CSV 文件缺少必要列：{missing}，实际列：{list(df_raw.columns)}")

    # ── 3. 数据清洗 ───────────────────────────────────────────────────────────
    print("\n[步骤1] 数据清洗")
    df = clean_dataframe(df_raw, min_text_length)

    # ── 4. 加载停用词 ─────────────────────────────────────────────────────────
    # 若 config 提供了停用词路径则解析为绝对路径
    if stopwords_path is not None:
        sp = Path(stopwords_path)
        if not sp.is_absolute():
            sp = root / sp
        stopwords_path = sp

    stopwords = load_stopwords(stopwords_path)

    # ── 5. 分词 ───────────────────────────────────────────────────────────────
    if use_jieba:
        try:
            import jieba  # noqa: F401  仅做可用性检测
            print("\n[步骤2] jieba 分词中，请稍候...")
            # 关闭 jieba 默认的进度日志，减少输出噪音
            import logging
            logging.getLogger("jieba").setLevel(logging.ERROR)

            df["tokens"] = df["review"].apply(lambda t: tokenize(t, stopwords))
            print(f"[信息] 分词完成，示例：\n  原文：{df['review'].iloc[0][:30]}…")
            print(f"  词元：{df['tokens'].iloc[0][:60]}…")
        except ImportError:
            print("[警告] jieba 未安装，跳过分词，tokens 列将与 text 列相同。")
            print("       可通过 pip install jieba 安装。")
            df["tokens"] = df["review"]
    else:
        # 不分词时，tokens 直接保存原始文本
        df["tokens"] = df["review"]
        print("\n[步骤2] 已跳过分词（use_jieba=false）")

    # ── 6. 构造输出 DataFrame（text / tokens / label）────────────────────────
    df_out = pd.DataFrame({
        "text": df["review"],       # 原始评论文本
        "tokens": df["tokens"],     # 分词结果（空格分隔）
        "label": df["cat"],         # 类别名作为分类标签
    })

    # ── 7. 数据集划分 ─────────────────────────────────────────────────────────
    print("\n[步骤3] 划分 train / val / test")
    train_df, val_df, test_df = split_data(df_out, val_ratio, test_ratio, seed)

    # ── 8. 保存文件 ───────────────────────────────────────────────────────────
    print("\n[步骤4] 保存文件")
    train_df.to_csv(train_file, index=False, encoding="utf-8")
    val_df.to_csv(val_file, index=False, encoding="utf-8")
    test_df.to_csv(test_file, index=False, encoding="utf-8")
    print(f"  训练集 -> {train_file}")
    print(f"  验证集 -> {val_file}")
    print(f"  测试集 -> {test_file}")

    # ── 9. 统计输出 ───────────────────────────────────────────────────────────
    print_statistics(len(df_out), train_df, val_df, test_df)

    print("\n[完成] 预处理结束。")


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="中文文本分类数据预处理：online_shopping_10_cats -> train/val/test"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="配置文件路径（相对于项目根目录或绝对路径），默认 configs/default.yaml",
    )
    args = parser.parse_args()
    main(args.config)
