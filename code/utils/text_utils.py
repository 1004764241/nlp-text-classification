# -*- coding: utf-8 -*-
"""
文本处理工具函数模块

提供NLP任务中常用的文本清洗、分词、停用词加载及标签映射构建等工具函数。
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import jieba

logger = logging.getLogger(__name__)

# 匹配需要保留的字符：中文、英文字母、数字、空格
_KEEP_PATTERN = re.compile(r"[^一-龥a-zA-Z0-9\s]")
# 匹配HTML标签
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# 匹配连续多余空白字符
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """去除HTML标签、特殊符号和多余空格，保留中文、英文字母和数字。

    处理流程：
    1. 去除HTML标签（如 <p>、<br/> 等）
    2. 去除不在保留字符集内的特殊符号
    3. 合并连续空白字符为单个空格
    4. 去除首尾空格

    Args:
        text: 待清洗的原始字符串。

    Returns:
        清洗后的字符串。若输入为空字符串则原样返回。

    Examples:
        >>> clean_text("<p>Hello, 世界！  123</p>")
        'Hello 世界 123'
        >>> clean_text("  多余   空格  ")
        '多余 空格'
    """
    if not text:
        return text

    # 步骤1：去除HTML标签
    text = _HTML_TAG_PATTERN.sub(" ", text)

    # 步骤2：去除特殊符号，保留中文、英文、数字、空白
    text = _KEEP_PATTERN.sub(" ", text)

    # 步骤3：合并连续空白为单个空格，并去除首尾空格
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def tokenize(text: str, stopwords: set = None) -> str:
    """对文本进行jieba分词，返回空格分隔的tokens字符串。

    使用jieba精确模式分词。分词结果中会过滤掉纯空白token；
    若提供停用词集合，则同时过滤停用词。

    Args:
        text: 待分词的字符串，建议先经过 ``clean_text`` 处理。
        stopwords: 可选的停用词集合。为 ``None`` 时不进行停用词过滤。

    Returns:
        以单个空格分隔的tokens字符串。若输入为空则返回空字符串。

    Examples:
        >>> tokenize("我爱自然语言处理")
        '我 爱 自然 语言 处理'
        >>> tokenize("我爱自然语言处理", stopwords={"我", "爱"})
        '自然 语言 处理'
    """
    if not text:
        return ""

    tokens = jieba.cut(text, cut_all=False)

    if stopwords is not None:
        filtered = [tok for tok in tokens if tok.strip() and tok not in stopwords]
    else:
        filtered = [tok for tok in tokens if tok.strip()]

    return " ".join(filtered)


def load_stopwords(filepath: str | Path) -> set:
    """从文件加载停用词，每行一个词。

    文件编码默认为UTF-8。自动跳过空行。若文件不存在或读取失败，
    记录警告日志并返回空集合，不抛出异常，确保调用方可安全降级。

    Args:
        filepath: 停用词文件路径，可为字符串或 ``pathlib.Path`` 对象。

    Returns:
        由停用词组成的集合。文件不存在时返回空集合。

    Examples:
        >>> sw = load_stopwords("stopwords.txt")
        >>> isinstance(sw, set)
        True
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning("停用词文件不存在，返回空集合：%s", filepath)
        return set()

    try:
        with filepath.open(encoding="utf-8") as f:
            stopwords = {line.rstrip("\n") for line in f if line.strip()}
        logger.info("成功加载停用词 %d 个，来源：%s", len(stopwords), filepath)
        return stopwords
    except OSError as exc:
        logger.warning("读取停用词文件失败，返回空集合：%s — %s", filepath, exc)
        return set()


def build_label_mapping(labels: list) -> tuple[dict, dict]:
    """根据标签列表构建双向映射字典。

    对输入列表去重并排序后，依次分配从0开始的整数ID，
    保证相同标签集合每次生成的映射一致（确定性）。

    Args:
        labels: 标签列表，元素类型通常为 ``str``，允许包含重复值。

    Returns:
        二元组 ``(label2id, id2label)``：

        - ``label2id``：``dict[str, int]``，label -> id 映射。
        - ``id2label``：``dict[int, str]``，id -> label 映射。

    Raises:
        ValueError: 若 ``labels`` 为空列表。

    Examples:
        >>> l2i, i2l = build_label_mapping(["正面", "负面", "中性", "正面"])
        >>> l2i
        {'中性': 0, '正面': 1, '负面': 2}
        >>> i2l
        {0: '中性', 1: '正面', 2: '负面'}
    """
    if not labels:
        raise ValueError("labels 不能为空列表")

    unique_labels = sorted(set(labels))
    label2id: dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}
    id2label: dict[int, str] = {idx: label for label, idx in label2id.items()}

    logger.info("构建标签映射完成，共 %d 个类别：%s", len(label2id), list(label2id.keys()))
    return label2id, id2label
