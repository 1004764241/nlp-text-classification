# 数据目录说明

## 目录用途

| 目录 | 说明 |
|------|------|
| `raw/` | 原始下载数据，不做修改 |
| `processed/` | 预处理后的训练/验证/测试集 |
| `external/` | 停用词表、词典等辅助文件 |

## 推荐数据格式

预处理后的 CSV 示例：

```csv
text,label
这是一条体育新闻,体育
产品很好用，推荐购买,正面
```

## 常用公开数据集（文本分类）

- [THUCNews 子集](https://github.com/gaussic/text-classification-cnn-rnn) — 中文新闻分类
- [ChnSentiCorp](https://github.com/pengming617/sentiment_analysis) — 中文情感/评论
- [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 英文垃圾短信（体积小，适合 baseline）

## 注意事项

- 原始数据体积较大，**不要**提交到 Git；仅在本 README 中记录来源与下载方式。
- 若需共享小样本，可放 `data/samples/` 并控制在 MB 级别以内。
