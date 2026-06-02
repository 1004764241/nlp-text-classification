"""
生成《基于TF-IDF与支持向量机的中文商品评论多分类研究》课程报告PPT
运行：python make_ppt.py
输出：docs/课程报告.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm
import os, copy
from lxml import etree

# ──────────────────────────────────────────────
# 颜色常量（深蓝主题）
# ──────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x1F, 0x4E, 0x79)   # 深蓝 — 标题背景
MID_BLUE    = RGBColor(0x2E, 0x75, 0xB6)   # 中蓝 — 强调
LIGHT_BLUE  = RGBColor(0xBD, 0xD7, 0xEE)   # 浅蓝 — 内容背景条
ORANGE      = RGBColor(0xC5, 0x5A, 0x11)   # 橙色 — 重点标注
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT   = RGBColor(0x1A, 0x1A, 0x1A)
GRAY        = RGBColor(0x59, 0x59, 0x59)
LIGHT_GRAY  = RGBColor(0xF2, 0xF2, 0xF2)
GREEN       = RGBColor(0x37, 0x86, 0x53)   # 绿色 — 好结果
RED_SOFT    = RGBColor(0xC0, 0x39, 0x2B)   # 红色 — 问题

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

FIGURES = os.path.join(os.path.dirname(__file__), "figures")
RESULTS  = os.path.join(os.path.dirname(__file__), "results")


# ──────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────
def add_rect(slide, l, t, w, h, fill_color=None, line_color=None, line_width=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(h)
    )
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        if line_width:
            shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, text, l, t, w, h,
                 font_size=18, bold=False, color=DARK_TEXT,
                 align=PP_ALIGN.LEFT, italic=False, font_name="微软雅黑"):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.text_frame.word_wrap = True
    p = txb.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    return txb


def add_para(tf, text, font_size=16, bold=False, color=DARK_TEXT,
             align=PP_ALIGN.LEFT, space_before=6, indent=None, font_name="微软雅黑"):
    """向 text_frame 追加一个段落"""
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    if indent is not None:
        p.level = indent
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    return p


def header_bar(slide, title, subtitle=None):
    """顶部深蓝标题条"""
    add_rect(slide, 0, 0, 13.33, 1.1, fill_color=DARK_BLUE)
    add_text_box(slide, title, 0.4, 0.1, 10, 0.7,
                 font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text_box(slide, subtitle, 0.4, 0.72, 10, 0.35,
                     font_size=14, color=LIGHT_BLUE, align=PP_ALIGN.LEFT)
    # 底部橙色装饰条
    add_rect(slide, 0, 1.1, 13.33, 0.06, fill_color=ORANGE)


def slide_number(slide, num, total):
    add_text_box(slide, f"{num} / {total}", 12.2, 7.1, 1.0, 0.35,
                 font_size=11, color=GRAY, align=PP_ALIGN.RIGHT)


# ──────────────────────────────────────────────
# Slide 1 — 封面
# ──────────────────────────────────────────────
def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # 背景深蓝渐变块
    add_rect(slide, 0, 0, 13.33, 7.5, fill_color=DARK_BLUE)

    # 顶部浅蓝装饰矩形
    add_rect(slide, 0, 0, 13.33, 0.12, fill_color=MID_BLUE)

    # 右侧橙色竖条
    add_rect(slide, 12.9, 0, 0.43, 7.5, fill_color=ORANGE)

    # 主标题
    add_text_box(slide,
                 "基于 TF-IDF 与支持向量机的",
                 0.8, 1.6, 11.8, 0.85,
                 font_size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide,
                 "中文商品评论多分类研究",
                 0.8, 2.38, 11.8, 0.85,
                 font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # 橙色分割线
    add_rect(slide, 3.5, 3.4, 6.3, 0.05, fill_color=ORANGE)

    # 副信息
    info = [
        ("课　　程", "自然语言处理"),
        ("院　　校", "重庆邮电大学"),
        ("学　　期", "2025-2026 学年第 2 学期"),
    ]
    for i, (k, v) in enumerate(info):
        y = 3.65 + i * 0.55
        add_text_box(slide, f"{k}：{v}", 3.5, y, 8, 0.5,
                     font_size=18, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)


# ──────────────────────────────────────────────
# Slide 2 — 目录
# ──────────────────────────────────────────────
def slide_outline(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "目  录")
    slide_number(slide, sn, total)

    items = [
        ("01", "研究背景与意义"),
        ("02", "相关工作"),
        ("03", "系统架构"),
        ("04", "数据预处理"),
        ("05", "TF-IDF 特征提取"),
        ("06", "分类模型"),
        ("07", "实验设置"),
        ("08", "实验结果与分析"),
        ("09", "总结与展望"),
    ]
    cols = [items[:5], items[5:]]
    col_x = [0.5, 7.0]
    for ci, col in enumerate(cols):
        for ri, (num, txt) in enumerate(col):
            y = 1.5 + ri * 1.0
            x = col_x[ci]
            # 编号圆角矩形
            add_rect(slide, x, y, 0.7, 0.65, fill_color=MID_BLUE)
            add_text_box(slide, num, x, y + 0.05, 0.7, 0.55,
                         font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
            # 条目文字
            add_text_box(slide, txt, x + 0.85, y + 0.06, 5.5, 0.55,
                         font_size=19, color=DARK_TEXT)


# ──────────────────────────────────────────────
# Slide 3 — 研究背景
# ──────────────────────────────────────────────
def slide_background(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "研究背景与意义", "为什么做中文商品评论分类？")
    slide_number(slide, sn, total)

    # 左侧三个要点卡片
    cards = [
        (MID_BLUE,  "📦 数据规模",
         "淘宝、京东每天产生海量用户评论，人工审核效率低下，自动化文本分类需求迫切。"),
        (ORANGE,    "🔤 中文挑战",
         "中文无天然词边界，需专用分词工具；存在大量同义词、歧义词、网络新词；口语化噪声多。"),
        (GREEN,     "🎯 研究目标",
         "构建覆盖 10 类商品评论的自动分类系统，探索 TF-IDF + 经典分类器的性能基线。"),
    ]
    for i, (color, title, body) in enumerate(cards):
        y = 1.35 + i * 1.88
        add_rect(slide, 0.4, y, 8.0, 1.65, fill_color=LIGHT_GRAY,
                 line_color=color, line_width=2)
        # 左侧色条
        add_rect(slide, 0.4, y, 0.12, 1.65, fill_color=color)
        add_text_box(slide, title, 0.7, y + 0.08, 7.5, 0.45,
                     font_size=17, bold=True, color=color)
        add_text_box(slide, body, 0.7, y + 0.52, 7.5, 1.0,
                     font_size=15, color=DARK_TEXT)

    # 右侧数据亮点
    add_rect(slide, 8.9, 1.35, 4.0, 5.7, fill_color=DARK_BLUE)
    add_text_box(slide, "数据集概览", 9.0, 1.45, 3.8, 0.5,
                 font_size=16, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

    stats = [
        ("62,605", "条有效评论"),
        ("10", "商品类别"),
        ("8:1:1", "训练/验证/测试"),
        ("50,000", "TF-IDF 特征维度"),
        ("7.6 秒", "SVM 训练耗时（CPU）"),
    ]
    for i, (val, label) in enumerate(stats):
        y = 2.1 + i * 0.98
        add_text_box(slide, val, 9.0, y, 3.8, 0.5,
                     font_size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, label, 9.0, y + 0.45, 3.8, 0.35,
                     font_size=13, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)


# ──────────────────────────────────────────────
# Slide 4 — 相关工作
# ──────────────────────────────────────────────
def slide_related(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "相关工作", "文本分类方法的发展历程")
    slide_number(slide, sn, total)

    stages = [
        (MID_BLUE,  "① 规则方法",
         "依赖人工规则与词典\n可解释性强，维护成本高"),
        (GREEN,     "② 统计机器学习",
         "TF-IDF + SVM / NB / LR\n新闻分类、垃圾邮件过滤效果好\n◀ 本文关注范畴"),
        (ORANGE,    "③ 深度学习",
         "TextCNN / TextRNN\n捕捉局部/序列特征\n显著优于传统方法"),
        (RGBColor(0x7B,0x2F,0xBE), "④ 预训练模型",
         "BERT / BERT-wwm / MacBERT\n当前主流方案，中文任务大幅领先"),
    ]

    # 时间轴主线
    add_rect(slide, 0.5, 3.8, 12.3, 0.08, fill_color=GRAY)

    for i, (color, title, body) in enumerate(stages):
        x = 0.5 + i * 3.1
        # 连接圆点
        add_rect(slide, x + 1.2, 3.56, 0.42, 0.42, fill_color=color)
        # 卡片
        add_rect(slide, x, 1.3, 2.9, 2.15, fill_color=LIGHT_GRAY,
                 line_color=color, line_width=1.5)
        add_rect(slide, x, 1.3, 2.9, 0.45, fill_color=color)
        add_text_box(slide, title, x + 0.1, 1.33, 2.7, 0.42,
                     font_size=15, bold=True, color=WHITE)
        add_text_box(slide, body, x + 0.1, 1.82, 2.7, 1.6,
                     font_size=13, color=DARK_TEXT)
        # 年代标注
        years = ["1990s", "2000s", "2014+", "2018+"]
        add_text_box(slide, years[i], x + 0.9, 4.1, 1.2, 0.4,
                     font_size=13, color=GRAY, align=PP_ALIGN.CENTER)

    # 本文定位
    add_rect(slide, 0.5, 5.0, 12.3, 1.3, fill_color=LIGHT_BLUE)
    add_text_box(slide,
                 "本文定位：聚焦阶段②，深入实现 TF-IDF + LinearSVC/MultinomialNB/LogisticRegression，"
                 "为引入深度学习方法提供基线对比。",
                 0.7, 5.1, 11.9, 1.1,
                 font_size=15, color=DARK_BLUE)


# ──────────────────────────────────────────────
# Slide 5 — 系统架构
# ──────────────────────────────────────────────
def slide_arch(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "系统架构", "四模块流水线设计")
    slide_number(slide, sn, total)

    # 插入流水线示意图（如果存在）
    fig_path = os.path.join(FIGURES, "fig_pipeline.png")
    if os.path.exists(fig_path):
        slide.shapes.add_picture(fig_path,
                                 Inches(1.0), Inches(1.3),
                                 Inches(11.3), Inches(4.5))
    else:
        # 手工绘制流水线
        modules = [
            (MID_BLUE,  "原始 CSV 数据",    "online_shopping_10_cats"),
            (GREEN,     "数据预处理",        "jieba 分词\n停用词过滤\n数据清洗\n集合划分"),
            (ORANGE,    "TF-IDF 特征工程",  "Bigram + Unigram\nmax_features=50k\nL2 归一化"),
            (MID_BLUE,  "分类器训练",        "LinearSVC\nMultinomialNB\nLogisticRegression"),
            (GREEN,     "评估与可视化",      "Accuracy / F1\n混淆矩阵\n分类报告"),
        ]
        for i, (color, title, body) in enumerate(modules):
            x = 0.4 + i * 2.55
            add_rect(slide, x, 1.4, 2.25, 2.6, fill_color=LIGHT_GRAY,
                     line_color=color, line_width=1.5)
            add_rect(slide, x, 1.4, 2.25, 0.5, fill_color=color)
            add_text_box(slide, title, x + 0.05, 1.43, 2.15, 0.46,
                         font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
            add_text_box(slide, body, x + 0.05, 1.98, 2.15, 2.0,
                         font_size=12, color=DARK_TEXT, align=PP_ALIGN.CENTER)
            if i < len(modules) - 1:
                add_text_box(slide, "▶", x + 2.25, 2.45, 0.3, 0.5,
                             font_size=20, color=MID_BLUE, align=PP_ALIGN.CENTER)

    # 输出文件说明
    add_rect(slide, 0.4, 5.85, 12.5, 1.3, fill_color=LIGHT_BLUE)
    outputs = [
        "📊 classification_report.json",
        "🔥 confusion_matrix.png",
        "📈 metrics_bar.png",
        "💾 tfidf_svm_model.pkl",
    ]
    add_text_box(slide, "输出文件：" + "　　".join(outputs),
                 0.6, 5.95, 12.2, 1.0,
                 font_size=13, color=DARK_BLUE)


# ──────────────────────────────────────────────
# Slide 6 — 数据预处理
# ──────────────────────────────────────────────
def slide_preprocess(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "数据预处理", "清洗 · 分词 · 划分")
    slide_number(slide, sn, total)

    # 左：三步骤
    steps = [
        (MID_BLUE, "① 数据清洗",
         "• 删除 cat / review 空值记录（1 条）\n"
         "• 去除类别+内容重复记录（17 条）\n"
         "• 过滤长度 < 5 字符的短文本（151 条）\n"
         "→ 62,774 条 → 62,605 条有效数据"),
        (GREEN, "② jieba 中文分词",
         "• 精确模式（cut_all=False）\n"
         "• 基于前缀词典 + HMM，处理歧义词\n"
         "• 过滤停用词与纯标点符号\n"
         "例：「这款手机信号很差」→「这款 手机 信号 很 差」"),
        (ORANGE, "③ 数据集划分（分层抽样）",
         "• 训练集：50,083 条\n"
         "• 验证集：  6,261 条\n"
         "• 测试集：  6,261 条\n"
         "• 各类别比例保持一致（stratified split）"),
    ]
    for i, (color, title, body) in enumerate(steps):
        y = 1.35 + i * 1.88
        add_rect(slide, 0.4, y, 7.8, 1.7, fill_color=LIGHT_GRAY,
                 line_color=color, line_width=1.5)
        add_rect(slide, 0.4, y, 0.12, 1.7, fill_color=color)
        add_text_box(slide, title, 0.65, y + 0.05, 7.0, 0.44,
                     font_size=15, bold=True, color=color)
        add_text_box(slide, body, 0.65, y + 0.52, 7.2, 1.1,
                     font_size=13, color=DARK_TEXT)

    # 右：数据集各类样本数
    add_rect(slide, 8.6, 1.35, 4.4, 5.7, fill_color=DARK_BLUE)
    add_text_box(slide, "各类别测试集分布", 8.7, 1.4, 4.2, 0.45,
                 font_size=14, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)

    rows = [
        ("书籍",  "385",  "6.1%"),
        ("平板",  "997",  "15.9%"),
        ("手机",  "229",  "3.7%"),
        ("水果",  "1000", "16.0%"),
        ("洗发水","1000", "16.0%"),
        ("热水器","54 ⚠","0.9%"),
        ("蒙牛",  "202",  "3.2%"),
        ("衣服",  "1000", "16.0%"),
        ("计算机","398",  "6.4%"),
        ("酒店",  "996",  "15.9%"),
    ]
    for i, (cat, n, pct) in enumerate(rows):
        y = 2.0 + i * 0.47
        color = ORANGE if "⚠" in n else WHITE
        add_text_box(slide, cat,  8.7,  y, 1.5, 0.42, font_size=12, color=color)
        add_text_box(slide, n,    10.2, y, 1.2, 0.42, font_size=12, color=color, align=PP_ALIGN.RIGHT)
        add_text_box(slide, pct,  11.5, y, 1.3, 0.42, font_size=12, color=LIGHT_BLUE, align=PP_ALIGN.RIGHT)


# ──────────────────────────────────────────────
# Slide 7 — TF-IDF 特征提取
# ──────────────────────────────────────────────
def slide_tfidf(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "TF-IDF 特征提取", "词频-逆文档频率 × Bigram")
    slide_number(slide, sn, total)

    # 公式区
    add_rect(slide, 0.4, 1.25, 12.5, 1.9, fill_color=LIGHT_BLUE)
    add_text_box(slide, "核心公式",
                 0.6, 1.3, 3.0, 0.5, font_size=14, bold=True, color=DARK_BLUE)
    formulas = [
        "TF-IDF(t, d)  =  TF(t, d)  ×  IDF(t)",
        "TF(t, d)  =  1 + log( count(t, d) )          （次线性平滑）",
        "IDF(t)  =  log[ (1+N) / (1+df(t)) ] + 1",
    ]
    for i, f in enumerate(formulas):
        add_text_box(slide, f, 0.6, 1.75 + i * 0.44, 12.0, 0.42,
                     font_size=13, color=DARK_BLUE, italic=(i > 0))

    # 设计要点
    points = [
        (MID_BLUE, "Bigram 特征",
         "ngram_range=(1,2)，同时捕捉「非常 满意」「质量 差」等短语，增强判别力"),
        (ORANGE, "词表大小",
         "max_features=50,000，按权重保留最具区分力的 5 万个特征（含 Bigram）"),
        (GREEN, "频率过滤",
         "min_df=2 过滤极低频词；max_df=0.95 过滤出现在 95%+ 文档中的高频词"),
        (MID_BLUE, "L2 归一化",
         "消除文档长度差异影响，使短评与长评在向量空间中可比"),
    ]
    for i, (color, title, body) in enumerate(points):
        col = i % 2
        row = i // 2
        x = 0.4 + col * 6.5
        y = 3.35 + row * 1.85
        add_rect(slide, x, y, 6.2, 1.65, fill_color=LIGHT_GRAY,
                 line_color=color, line_width=1.5)
        add_rect(slide, x, y, 0.1, 1.65, fill_color=color)
        add_text_box(slide, title, x + 0.25, y + 0.1, 5.8, 0.42,
                     font_size=14, bold=True, color=color)
        add_text_box(slide, body,  x + 0.25, y + 0.55, 5.8, 1.0,
                     font_size=13, color=DARK_TEXT)


# ──────────────────────────────────────────────
# Slide 8 — 分类模型
# ──────────────────────────────────────────────
def slide_models(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "分类模型", "三种经典分类器对比")
    slide_number(slide, sn, total)

    models = [
        (MID_BLUE, "LinearSVC", "线性支持向量机",
         "最大化类间间隔，One-vs-Rest 多分类",
         "C=1.0, max_iter=2000",
         "89.23% ✓ 最优"),
        (GREEN, "MultinomialNB", "多项式朴素贝叶斯",
         "贝叶斯定理 + 特征条件独立假设",
         "alpha=1.0（拉普拉斯平滑）",
         "~84%"),
        (ORANGE, "Logistic\nRegression", "逻辑回归",
         "Softmax 建模后验概率，L2 正则化",
         "solver=lbfgs, max_iter=1000",
         "~87%"),
    ]
    for i, (color, name, cname, principle, params, acc) in enumerate(models):
        x = 0.4 + i * 4.3
        # 卡片
        add_rect(slide, x, 1.35, 4.05, 5.75, fill_color=LIGHT_GRAY,
                 line_color=color, line_width=2)
        add_rect(slide, x, 1.35, 4.05, 0.7, fill_color=color)
        add_text_box(slide, name,  x+0.1, 1.38, 3.85, 0.38,
                     font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, cname, x+0.1, 1.76, 3.85, 0.35,
                     font_size=12, color=WHITE, align=PP_ALIGN.CENTER)

        add_text_box(slide, "核心原理", x+0.15, 2.18, 3.75, 0.38,
                     font_size=12, bold=True, color=color)
        add_text_box(slide, principle, x+0.15, 2.55, 3.75, 0.65,
                     font_size=12, color=DARK_TEXT)

        add_text_box(slide, "主要参数", x+0.15, 3.28, 3.75, 0.38,
                     font_size=12, bold=True, color=color)
        add_text_box(slide, params, x+0.15, 3.64, 3.75, 0.55,
                     font_size=12, color=DARK_TEXT)

        add_text_box(slide, "优　　势", x+0.15, 4.28, 3.75, 0.38,
                     font_size=12, bold=True, color=color)
        advantages = [
            "高维稀疏特征下速度快\n训练耗时仅约 7.6s（CPU）",
            "训练预测速度极快\n参数量少，内存占用低",
            "输出概率估计\n可设置置信度阈值",
        ]
        add_text_box(slide, advantages[i], x+0.15, 4.64, 3.75, 0.7,
                     font_size=12, color=DARK_TEXT)

        # 准确率徽章
        acc_color = MID_BLUE if "最优" in acc else GRAY
        add_rect(slide, x+0.3, 5.55, 3.45, 0.9, fill_color=acc_color)
        add_text_box(slide, f"准确率  {acc}", x+0.3, 5.65, 3.45, 0.7,
                     font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ──────────────────────────────────────────────
# Slide 9 — 实验设置
# ──────────────────────────────────────────────
def slide_setup(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "实验设置", "参数配置与复现说明")
    slide_number(slide, sn, total)

    rows_left = [
        ("随机种子",          "42",              "保证实验可复现"),
        ("最短文本长度",      "5 字符",           "过滤无效短文本"),
        ("训练/验证/测试",    "8 : 1 : 1",        "分层抽样"),
        ("TF-IDF 词表大小",   "50,000",           "按权重取前 50k 特征"),
        ("n-gram 范围",       "(1, 2)",           "Unigram + Bigram"),
    ]
    rows_right = [
        ("sublinear_tf",     "True",            "使用 1+log(tf) 平滑"),
        ("min_df",           "2",               "过滤超低频词"),
        ("max_df",           "0.95",            "过滤超高频词"),
        ("SVM C 值",         "1.0",             "正则化权衡参数"),
        ("SVM 最大迭代",      "2000",            "确保收敛"),
    ]

    def draw_table(rows, x_start, y_start):
        headers = ["参数", "取值", "说明"]
        col_w = [2.5, 1.5, 2.8]
        col_x = [x_start + sum(col_w[:j]) for j in range(3)]

        # 表头
        for j, (h, w) in enumerate(zip(headers, col_w)):
            add_rect(slide, col_x[j], y_start, w, 0.44, fill_color=DARK_BLUE)
            add_text_box(slide, h, col_x[j]+0.05, y_start+0.05, w-0.1, 0.35,
                         font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        for ri, row in enumerate(rows):
            bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
            for j, (cell, w) in enumerate(zip(row, col_w)):
                add_rect(slide, col_x[j], y_start+0.44+ri*0.44, w, 0.44,
                         fill_color=bg, line_color=RGBColor(0xCC,0xCC,0xCC), line_width=0.5)
                add_text_box(slide, cell, col_x[j]+0.05, y_start+0.48+ri*0.44,
                             w-0.1, 0.38, font_size=12, color=DARK_TEXT,
                             align=PP_ALIGN.CENTER if j==1 else PP_ALIGN.LEFT)

    draw_table(rows_left,  0.35, 1.3)
    draw_table(rows_right, 6.75, 1.3)

    # 底部环境说明
    add_rect(slide, 0.35, 5.95, 12.6, 1.1, fill_color=DARK_BLUE)
    env_info = "🖥  运行环境：Python 3.9+ | scikit-learn 1.x | jieba 0.42 | 纯 CPU 训练（无需 GPU）  " \
               "⏱  SVM 训练耗时约 7.6 秒  |  预处理（jieba 分词）约 3-5 分钟"
    add_text_box(slide, env_info, 0.55, 6.05, 12.2, 0.9,
                 font_size=13, color=LIGHT_BLUE)


# ──────────────────────────────────────────────
# Slide 10 — 实验结果（整体 + 各类别表格）
# ──────────────────────────────────────────────
def slide_results(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "实验结果", "TF-IDF + LinearSVC 测试集表现")
    slide_number(slide, sn, total)

    # 整体指标三卡片
    metrics = [
        (MID_BLUE,  "Accuracy",    "89.23%"),
        (ORANGE,    "Macro-F1",    "88.11%"),
        (GREEN,     "Weighted-F1", "89.24%"),
    ]
    for i, (color, name, val) in enumerate(metrics):
        x = 0.4 + i * 4.3
        add_rect(slide, x, 1.3, 4.0, 1.5, fill_color=color)
        add_text_box(slide, val, x, 1.4, 4.0, 0.85,
                     font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, name, x, 2.1, 4.0, 0.55,
                     font_size=16, color=WHITE, align=PP_ALIGN.CENTER)

    # 各类别详细表格
    headers = ["类别", "Precision", "Recall", "F1-score", "样本数"]
    col_w   = [1.5, 1.7, 1.7, 1.8, 1.3]
    col_x   = [0.35 + sum(col_w[:j]) for j in range(5)]
    y0 = 3.05

    for j, (h, w) in enumerate(zip(headers, col_w)):
        add_rect(slide, col_x[j], y0, w, 0.42, fill_color=DARK_BLUE)
        add_text_box(slide, h, col_x[j]+0.04, y0+0.04, w-0.08, 0.34,
                     font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    data = [
        ("书籍",   "0.956", "0.956", "0.956", "385"),
        ("平板",   "0.806", "0.817", "0.812", "997"),
        ("手机",   "0.957", "0.873", "0.913", "229"),
        ("水果",   "0.916", "0.892", "0.904", "1000"),
        ("洗发水", "0.803", "0.876", "0.838", "1000"),
        ("热水器", "0.923", "0.444", "0.600", "54"),
        ("蒙牛",   "1.000", "1.000", "1.000", "202"),
        ("衣服",   "0.876", "0.878", "0.877", "1000"),
        ("计算机", "0.950", "0.910", "0.929", "398"),
        ("酒店",   "0.990", "0.974", "0.982", "996"),
    ]
    for ri, row in enumerate(data):
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        # 热水器行标红
        is_bad = row[0] == "热水器"
        is_best = row[0] == "蒙牛"
        for j, (cell, w) in enumerate(zip(row, col_w)):
            cell_bg = RGBColor(0xFF,0xEB,0xEB) if is_bad else (RGBColor(0xE8,0xF5,0xE9) if is_best else bg)
            add_rect(slide, col_x[j], y0+0.42+ri*0.41, w, 0.41,
                     fill_color=cell_bg, line_color=RGBColor(0xCC,0xCC,0xCC), line_width=0.5)
            text_color = RED_SOFT if is_bad else (GREEN if is_best else DARK_TEXT)
            add_text_box(slide, cell, col_x[j]+0.04, y0+0.45+ri*0.41,
                         w-0.08, 0.35, font_size=11, color=text_color,
                         align=PP_ALIGN.CENTER)

    # 图例说明
    add_rect(slide, 8.5, 3.05, 4.5, 4.0, fill_color=DARK_BLUE)
    add_text_box(slide, "关键发现", 8.6, 3.1, 4.3, 0.42,
                 font_size=14, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    notes = [
        ("✅", "蒙牛 F1=1.00", "词汇特征极具区分性"),
        ("✅", "酒店 F1=0.98", "领域词汇清晰"),
        ("✅", "书籍 F1=0.96", "特有词汇多"),
        ("⚠️", "热水器 Recall=0.44", "严重样本不均衡（54条）"),
        ("⚠️", "平板/洗发水", "数码/消费品词汇重叠"),
    ]
    for i, (icon, title, desc) in enumerate(notes):
        y = 3.65 + i * 0.66
        add_text_box(slide, icon,  8.55, y, 0.5, 0.5, font_size=14, color=WHITE)
        add_text_box(slide, title, 9.05, y, 3.8, 0.35, font_size=12, bold=True, color=WHITE)
        add_text_box(slide, desc,  9.05, y+0.33, 3.8, 0.3, font_size=11, color=LIGHT_BLUE)


# ──────────────────────────────────────────────
# Slide 11 — 混淆矩阵 & 指标图
# ──────────────────────────────────────────────
def slide_charts(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "可视化分析", "混淆矩阵 & 各类别指标柱状图")
    slide_number(slide, sn, total)

    conf_path    = os.path.join(FIGURES, "fig_confusion.png")
    metrics_path = os.path.join(FIGURES, "fig_metrics.png")

    if os.path.exists(conf_path):
        slide.shapes.add_picture(conf_path,
                                 Inches(0.3), Inches(1.3),
                                 Inches(6.2), Inches(5.7))
    if os.path.exists(metrics_path):
        slide.shapes.add_picture(metrics_path,
                                 Inches(6.8), Inches(1.3),
                                 Inches(6.2), Inches(5.7))

    add_text_box(slide, "混淆矩阵", 0.3, 6.95, 6.2, 0.4,
                 font_size=13, color=GRAY, align=PP_ALIGN.CENTER)
    add_text_box(slide, "各类别 Precision / Recall / F1", 6.8, 6.95, 6.2, 0.4,
                 font_size=13, color=GRAY, align=PP_ALIGN.CENTER)


# ──────────────────────────────────────────────
# Slide 12 — 结果分析
# ──────────────────────────────────────────────
def slide_analysis(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "结果分析", "亮点解读 & 局限性分析")
    slide_number(slide, sn, total)

    # 亮点
    add_rect(slide, 0.35, 1.3, 12.6, 0.42, fill_color=MID_BLUE)
    add_text_box(slide, "✅ 模型亮点", 0.5, 1.32, 5, 0.38,
                 font_size=14, bold=True, color=WHITE)

    goods = [
        "蒙牛 F1=1.00：品牌词「牛奶」「蒙牛」「乳」高度区分，TF-IDF 精准捕捉",
        "酒店 F1=0.98：领域词「入住」「前台」「房间」边界清晰",
        "书籍 F1=0.96：「作者」「章节」「阅读」等专属词汇丰富",
        "整体准确率 89.23%，纯 CPU 训练仅 7.6 秒，具备实用价值",
    ]
    for i, g in enumerate(goods):
        add_text_box(slide, f"• {g}", 0.5, 1.85 + i*0.5, 12.3, 0.48,
                     font_size=14, color=DARK_TEXT)

    # 局限
    add_rect(slide, 0.35, 4.1, 12.6, 0.42, fill_color=RED_SOFT)
    add_text_box(slide, "⚠️ 主要局限", 0.5, 4.12, 5, 0.38,
                 font_size=14, bold=True, color=WHITE)

    bads = [
        ("热水器 Recall=0.444",
         "训练样本仅 435 条（0.87%），严重类别不均衡导致分类器学习不足；\n"
         "可用上采样（SMOTE）或 class_weight='balanced' 缓解"),
        ("平板 / 洗发水 / 衣服混淆",
         "数码产品（平板/手机/计算机）共享「屏幕」「性能」等词；\n"
         "消费品评论风格相似，TF-IDF 无法捕捉语义关系"),
    ]
    for i, (title, body) in enumerate(bads):
        y = 4.65 + i * 1.3
        add_rect(slide, 0.35, y, 12.6, 1.15, fill_color=RGBColor(0xFF,0xEB,0xEB),
                 line_color=RED_SOFT, line_width=1)
        add_text_box(slide, title, 0.5, y+0.05, 4.0, 0.4,
                     font_size=13, bold=True, color=RED_SOFT)
        add_text_box(slide, body, 0.5, y+0.45, 12.0, 0.7,
                     font_size=13, color=DARK_TEXT)


# ──────────────────────────────────────────────
# Slide 13 — 总结与展望
# ──────────────────────────────────────────────
def slide_conclusion(prs, sn, total):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header_bar(slide, "总结与展望")
    slide_number(slide, sn, total)

    # 工作总结
    add_rect(slide, 0.35, 1.3, 5.9, 5.7, fill_color=DARK_BLUE)
    add_text_box(slide, "📋 工作总结", 0.55, 1.38, 5.5, 0.45,
                 font_size=16, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    summary = [
        "构建了完整的中文文本分类流水线",
        "（数据清洗 → 分词 → TF-IDF → 训练 → 评估）",
        "",
        "TF-IDF + LinearSVC 取得：",
        "  • 准确率 89.23%",
        "  • Macro-F1  88.11%",
        "  • 训练耗时  7.6 秒（纯 CPU）",
        "",
        "对比三种经典分类器：",
        "  LinearSVC > 逻辑回归 > 朴素贝叶斯",
        "",
        "验证了传统机器学习方法在中文",
        "评论分类任务上的有效性",
    ]
    for i, line in enumerate(summary):
        add_text_box(slide, line, 0.55, 1.95 + i*0.37, 5.5, 0.36,
                     font_size=13, color=WHITE if line else WHITE,
                     bold=("•" in line or "LinearSVC" in line))

    # 展望方向
    add_rect(slide, 6.6, 1.3, 6.7, 5.7, fill_color=LIGHT_GRAY)
    add_text_box(slide, "🔭 未来展望", 6.8, 1.38, 6.3, 0.45,
                 font_size=16, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

    future = [
        (MID_BLUE, "改进特征工程",
         "引入停用词词典进一步净化特征空间；\n字符级 n-gram 补充词汇覆盖"),
        (ORANGE, "引入词/句向量",
         "Word2Vec / fastText 引入语义信息；\nBERT 预训练模型替代 TF-IDF"),
        (GREEN, "解决类别不均衡",
         "SMOTE 上采样 / class_weight 代价敏感；\n针对热水器等少数类增加数据"),
        (RGBColor(0x7B,0x2F,0xBE), "深度学习对比",
         "TextCNN / TextRNN 与本文基线对比；\n量化深度模型的提升幅度"),
    ]
    for i, (color, title, body) in enumerate(future):
        y = 2.0 + i * 1.18
        add_rect(slide, 6.65, y, 6.5, 1.05, fill_color=WHITE,
                 line_color=color, line_width=1.5)
        add_rect(slide, 6.65, y, 0.1, 1.05, fill_color=color)
        add_text_box(slide, title, 6.85, y+0.04, 5.7, 0.38,
                     font_size=13, bold=True, color=color)
        add_text_box(slide, body,  6.85, y+0.44, 5.7, 0.58,
                     font_size=12, color=DARK_TEXT)


# ──────────────────────────────────────────────
# Slide 14 — 谢谢
# ──────────────────────────────────────────────
def slide_thanks(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, 13.33, 7.5, fill_color=DARK_BLUE)
    add_rect(slide, 0, 0, 13.33, 0.1, fill_color=ORANGE)
    add_rect(slide, 0, 7.4, 13.33, 0.1, fill_color=ORANGE)

    add_text_box(slide, "感  谢  聆  听", 0.5, 2.2, 12.3, 1.2,
                 font_size=52, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(slide, 4.5, 3.8, 4.3, 0.07, fill_color=ORANGE)

    add_text_box(slide, "敬请各位老师批评指正", 0.5, 4.1, 12.3, 0.7,
                 font_size=22, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
    add_text_box(slide,
                 "课程：自然语言处理　|　重庆邮电大学　|　2025-2026 学年第 2 学期",
                 0.5, 5.5, 12.3, 0.55,
                 font_size=15, color=GRAY, align=PP_ALIGN.CENTER)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    TOTAL = 14

    slide_cover(prs)                      # 1  封面
    slide_outline(prs,    2, TOTAL)       # 2  目录
    slide_background(prs, 3, TOTAL)       # 3  研究背景
    slide_related(prs,    4, TOTAL)       # 4  相关工作
    slide_arch(prs,       5, TOTAL)       # 5  系统架构
    slide_preprocess(prs, 6, TOTAL)       # 6  数据预处理
    slide_tfidf(prs,      7, TOTAL)       # 7  TF-IDF
    slide_models(prs,     8, TOTAL)       # 8  分类模型
    slide_setup(prs,      9, TOTAL)       # 9  实验设置
    slide_results(prs,   10, TOTAL)       # 10 实验结果
    slide_charts(prs,    11, TOTAL)       # 11 可视化图表
    slide_analysis(prs,  12, TOTAL)       # 12 结果分析
    slide_conclusion(prs,13, TOTAL)       # 13 总结展望
    slide_thanks(prs)                     # 14 谢谢

    out_path = os.path.join(os.path.dirname(__file__), "docs", "课程报告.pptx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prs.save(out_path)
    print(f"✅ PPT 已生成：{out_path}")
    print(f"   共 {TOTAL} 张幻灯片")


if __name__ == "__main__":
    main()
