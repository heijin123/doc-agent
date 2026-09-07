"""生成 docx/xlsx/pptx 演示样本（doc-agent 切片验证用）——运行一次即可。

产出去向：data/samples/generated/
- docx_spec.docx    Word 样本：标题层级 + 表格（验证结构切 + 表格二维保留）
- xlsx_products.xlsx  Excel 样本：商品表含空行（验证一行一记录 + 空行跳过）
- pptx_intro.pptx   PPT 样本：两页商务页（验证一页一块 + 标题进 section）

依赖：python-docx / openpyxl（pandas 直接读）/ python-pptx——已在 pyproject 声明。
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt
from pptx import Presentation
from pptx.util import Inches
from openpyxl import Workbook

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples" / "generated"


def make_docx_sample() -> None:
    document = Document()
    document.add_heading("产品规格说明", level=1)
    document.add_paragraph("本文件描述 DocAgent 演示产品的完整规格。")
    document.add_heading("核心参数", level=2)
    document.add_paragraph("所有参数均在出厂前经过严格测试。")

    table = document.add_table(rows=3, cols=3)
    table.rows[0].cells[0].text = "型号"
    table.rows[0].cells[1].text = "价格"
    table.rows[0].cells[2].text = "备注"
    table.rows[1].cells[0].text = "DA-100"
    table.rows[1].cells[1].text = "199 元"
    table.rows[1].cells[2].text = "入门款"
    table.rows[2].cells[0].text = "DA-500"
    table.rows[2].cells[1].text = "399 元"
    table.rows[2].cells[2].text = "旗舰款"
    document.add_paragraph("DA-500 支持本地知识库离线问答。")
    document.save(OUTPUT_DIR / "docx_spec.docx")


def make_xlsx_sample() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "商品清单"
    sheet.append(["商品名称", "价格", "库存"])
    sheet.append(["蓝牙耳机", 199, 50])
    sheet.append(["机械键盘", 399, 30])
    sheet.append([])  # 全空行：验证切块跳过
    sheet.append(["USB-C 扩展坞", 129, 80])
    workbook.save(OUTPUT_DIR / "xlsx_products.xlsx")


def make_pptx_sample() -> None:
    presentation = Presentation()
    first_slide_layout = presentation.slide_layouts[1]  # 标题 + 正文版式

    first_slide = presentation.slides.add_slide(first_slide_layout)
    first_slide.shapes.title.text = "DocAgent 介绍"
    first_slide.placeholders[1].text = "把企业散落的文档变成可检索、可问答、带出处的知识库"

    second_slide = presentation.slides.add_slide(first_slide_layout)
    second_slide.shapes.title.text = "核心能力"
    second_slide.placeholders[1].text = "多格式解析\n结构感知切片\n引用溯源问答"
    presentation.save(OUTPUT_DIR / "pptx_intro.pptx")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_docx_sample()
    make_xlsx_sample()
    make_pptx_sample()
    print("样本已生成到", OUTPUT_DIR)
