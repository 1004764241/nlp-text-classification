# 自然语言处理课程大作业

基于 CPU 的中文文本分类项目（可按选题自行调整）。

## 项目结构

```
.
├── code/                 # 源代码
│   ├── preprocess.py     # 数据预处理
│   ├── train.py          # 模型训练
│   ├── evaluate.py       # 模型评估
│   ├── predict.py        # 单条/批量预测
│   └── utils/            # 工具函数
├── configs/              # 配置文件（YAML）
├── data/
│   ├── raw/              # 原始数据集（不提交到 Git）
│   ├── processed/        # 预处理后的数据（不提交到 Git）
│   └── external/         # 第三方辅助数据（不提交到 Git）
├── models/               # 训练保存的模型（不提交到 Git）
├── results/              # 实验结果、图表、日志
├── docs/                 # 课程报告、答辩材料
└── notebooks/            # 探索性分析（可选）
```

## 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 使用流程

```bash
# 1. 将原始数据放入 data/raw/（参见 data/README.md）

# 2. 数据预处理
python code/preprocess.py --config configs/default.yaml

# 3. 训练模型
python code/train.py --config configs/default.yaml

# 4. 评估模型
python code/evaluate.py --config configs/default.yaml

# 5. 预测
python code/predict.py --text "待分类的文本"
```

## 数据集说明

请在 `data/README.md` 中记录所用数据集的来源、格式与划分方式。

## 注意事项

- `data/raw/`、`data/processed/`、`models/` 等大文件目录已在 `.gitignore` 中排除，不会上传到 GitHub。
- 课程报告模板等本地文件请放在 `docs/`，根目录下的 `.doc/.ppt` 文件默认不上传。
