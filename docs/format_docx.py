#!/usr/bin/env python3
"""
format_docx.py — 重邮学报风格论文格式化
目标：
  Heading 1  — 论文大标题：黑体二号，居中，段前12磅，段后6磅
  Heading 2  — 一级节标题（0 引言…）：黑体三号，左对齐，段前10磅，段后4磅
  Heading 3  — 二级节标题（2.1…）：黑体四号，左对齐，段前6磅，段后2磅
  Heading 4  — 三级节标题（2.2.1…）：黑体小四，左对齐，段前4磅，段后0磅
  Normal     — 正文：宋体（中）/ Times New Roman（英），小四(12pt)
               首行缩进2字符，1.5倍行距，段前0，段后0
  Source Code— 代码块：Courier New 9pt，无缩进，行距固定12pt
  TABLE:Normal—表格内：宋体小五(9pt)，居中，无缩进，行距单倍
"""

from copy import deepcopy
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_LINE_SPACING
import lxml.etree as etree

SRC  = "docs/正式论文.docx"
DST  = "docs/正式论文.docx"   # 原地覆盖

# ── 颜色常量（全部统一为黑色）──────────────────────────────────────────────
BLACK      = RGBColor(0x00, 0x00, 0x00)
DARK_BLUE  = BLACK
H2_COLOR   = BLACK
H3_COLOR   = BLACK
H4_COLOR   = BLACK

# ── 字体名 ──────────────────────────────────────────────────────────────────
SONG   = "宋体"
HEI    = "黑体"
TNR    = "Times New Roman"
COURIER= "Courier New"

# ── 工具函数 ────────────────────────────────────────────────────────────────

def set_run_font(run, zh_font, en_font, size_pt, bold=False, color=None):
    """设置 run 的中英文字体、字号、加粗、颜色。"""
    run.font.name = en_font
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    # 中文字体需要通过 XML 的 eastAsia 属性设置
    rpr = run._r.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rpr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), zh_font)
    rFonts.set(qn("w:ascii"),    en_font)
    rFonts.set(qn("w:hAnsi"),    en_font)


def set_para_spacing(para, line_rule, line_val,
                     space_before_pt=0, space_after_pt=0,
                     first_line_cm=None):
    """
    设置段落行距和段间距。
    line_rule: WD_LINE_SPACING.MULTIPLE / EXACTLY / AT_LEAST
    line_val : 行距值（MULTIPLE 时为倍数，EXACTLY 时为 Pt(n)）
    """
    pf = para.paragraph_format
    pf.line_spacing_rule = line_rule
    pf.line_spacing       = line_val
    pf.space_before       = Pt(space_before_pt)
    pf.space_after        = Pt(space_after_pt)
    if first_line_cm is not None:
        pf.first_line_indent = Cm(first_line_cm)


def style_heading1(para):
    """论文大标题：黑体二号，居中，深青色"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                     space_before_pt=12, space_after_pt=6,
                     first_line_cm=0)
    for run in para.runs:
        set_run_font(run, HEI, TNR, 22, bold=True, color=DARK_BLUE)


def style_heading2(para):
    """一级节标题（0 引言…）：黑体三号，左对齐，深青"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                     space_before_pt=10, space_after_pt=4,
                     first_line_cm=0)
    for run in para.runs:
        set_run_font(run, HEI, TNR, 16, bold=True, color=H2_COLOR)


def style_heading3(para):
    """二级节标题（2.1…）：黑体四号，青绿色"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                     space_before_pt=6, space_after_pt=2,
                     first_line_cm=0)
    for run in para.runs:
        set_run_font(run, HEI, TNR, 14, bold=True, color=H3_COLOR)


def style_heading4(para):
    """三级节标题（2.2.1…）：黑体小四，黑色"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                     space_before_pt=4, space_after_pt=0,
                     first_line_cm=0)
    for run in para.runs:
        set_run_font(run, HEI, TNR, 12, bold=True, color=H4_COLOR)


def style_normal(para):
    """正文：宋体/Times New Roman 小四，首行缩进2字符(约0.74cm)，1.5倍行距"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                     space_before_pt=0, space_after_pt=0,
                     first_line_cm=0.74)   # 小四字号≈0.37cm/字 × 2字
    for run in para.runs:
        set_run_font(run, SONG, TNR, 12, bold=False, color=BLACK)


def style_code(para):
    """代码块：Courier New 9pt，无缩进，固定行距12pt"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para_spacing(para, WD_LINE_SPACING.EXACTLY, Pt(14),
                     space_before_pt=0, space_after_pt=0,
                     first_line_cm=0)
    for run in para.runs:
        run.font.name  = COURIER
        run.font.size  = Pt(9)
        run.font.bold  = False
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)


def style_table_cell(para):
    """表格内正文：宋体五号(9pt)，居中，单倍行距"""
    para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.0,
                     space_before_pt=1, space_after_pt=1,
                     first_line_cm=0)
    for run in para.runs:
        set_run_font(run, SONG, TNR, 10, bold=False, color=BLACK)


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    doc = Document(SRC)

    # ── 1. 正文段落 ──────────────────────────────────────────────────────────
    for para in doc.paragraphs:
        sname = para.style.name
        txt   = para.text.strip()

        if sname == "Heading 1":
            style_heading1(para)

        elif sname == "Heading 2":
            style_heading2(para)

        elif sname == "Heading 3":
            # 判断是否为三级标题（形如 "2.2.1"）
            if txt and txt[0].isdigit() and txt.count("．") >= 2 or \
               txt and len(txt) > 3 and txt[1] == "." and txt[3] == ".":
                style_heading4(para)
            else:
                style_heading3(para)

        elif sname == "Normal":
            # 跳过空段落、图题（以"图"开头）、英文摘要副标题等短段
            if not txt:
                set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                                 space_before_pt=0, space_after_pt=0)
                continue
            # 图题/表题：居中，黑体五号，无缩进
            if txt.startswith(("图", "表", "Fig.", "Table")):
                para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                                 space_before_pt=2, space_after_pt=2,
                                 first_line_cm=0)
                for run in para.runs:
                    set_run_font(run, HEI, TNR, 10, bold=True, color=DARK_BLUE)
            # 摘要/Abstract 标签行（加粗首行）
            elif txt.startswith(("摘", "关键词", "中图", "Abstract", "Keywords")):
                para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                set_para_spacing(para, WD_LINE_SPACING.MULTIPLE, 1.5,
                                 space_before_pt=0, space_after_pt=0,
                                 first_line_cm=0.74)
                for run in para.runs:
                    set_run_font(run, SONG, TNR, 12, bold=False, color=BLACK)
            # 普通正文
            else:
                style_normal(para)

        elif sname == "Source Code":
            style_code(para)

    # ── 2. 表格内段落 + 表格整体居中 ────────────────────────────────────────
    for table in doc.tables:
        # 表格在页面上居中（设置 tblPr/jc 为 center）
        tblPr = table._tbl.find(qn("w:tblPr"))
        if tblPr is None:
            tblPr = OxmlElement("w:tblPr")
            table._tbl.insert(0, tblPr)
        jc = tblPr.find(qn("w:jc"))
        if jc is None:
            jc = OxmlElement("w:jc")
            tblPr.append(jc)
        jc.set(qn("w:val"), "center")

        # 单元格内容样式
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    style_table_cell(para)
        # 首行（表头）加粗
        if table.rows:
            for cell in table.rows[0].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.bold = True

    # ── 3. 页边距（上下2.54cm，左右3.0cm）───────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin   = Cm(3.00)
        section.right_margin  = Cm(3.00)

    doc.save(DST)
    print(f"✅ 格式化完成 → {DST}")


if __name__ == "__main__":
    main()
