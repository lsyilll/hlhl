"""
main.py - 自助式数据分析（数据分析智能体）
Author: 骆昊
Version: 0.1
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import re
from utils import dataframe_agent
import io

# 配置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = r"E:\Tesseract\tesseract.exe"

def preprocess_image(image, lang="eng"):
    """图像预处理：灰度化、降噪、二值化、增强对比度"""
    # 灰度化
    gray = image.convert("L")

    # 降噪（中值滤波）
    denoised = gray.filter(ImageFilter.MedianFilter())

    # 增强对比度
    enhancer = ImageEnhance.Contrast(denoised)
    contrast_image = enhancer.enhance(1.5)  # 可调整增强倍数

    # 二值化（自适应阈值）
    threshold = 127
    if lang == "chi_sim":  # 中文需要更高对比度
        threshold = 150
    binary = contrast_image.point(lambda x: 0 if x < threshold else 255)

    return binary

def detect_table_format(text):
    """检测表格格式：CSV、TSV或空格分隔"""
    if "," in text and text.count(",") > text.count("\t"):
        return ","
    elif "\t" in text:
        return "\t"
    else:
        return None  # 可能是不规则表格或纯文本

def parse_table(text, separator=None):
    """解析表格数据，支持不规则格式，增加容错处理"""
    lines = text.strip().split("\n")

    # 检测表头和数据行
    header = []
    data = []
    for i, line in enumerate(lines):
        if i == 0:
            if separator:
                header = line.split(separator)
            else:
                header = re.split(r'\s{2,}', line.strip())
        else:
            if not line.strip():
                continue
            if separator:
                parts = line.split(separator)
            else:
                parts = re.split(r'\s{2,}', line.strip())
            # 处理可能的误识别，若列数与表头相差较大，尝试重新分割
            if len(parts) < len(header) // 2 or len(parts) > len(header) * 2:
                parts = re.split(r'[\s,;]+', line.strip())
            data.append(parts)

    # 处理不规则表格（补齐列数）
    max_cols = len(header)
    for i in range(len(data)):
        if len(data[i]) < max_cols:
            data[i].extend([""] * (max_cols - len(data[i])))
        elif len(data[i]) > max_cols:
            data[i] = data[i][:max_cols]

    return pd.DataFrame(data, columns=header)

def create_chart(input_data, chart_type):
    """生成统计图表"""
    df_data = pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    )
    df_data.set_index("x", inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        plt.plot(df_data.index, df_data["y"], marker="o", linestyle="--")
        plt.ylim(0, df_data["y"].max() * 1.1)
        plt.title("xxxxxxxxxxx")
        st.pyplot(plt.gcf())
    elif chart_type == "scatter":
        plt.scatter(df_data.index, df_data["y"])
        plt.title("Scatter Plot")
        st.pyplot(plt.gcf())

def save_text_as_txt(text):
    """将文本保存为txt文件并提供下载链接"""
    buffer = io.StringIO()
    buffer.write(text)
    buffer.seek(0)
    st.download_button(
        label="下载识别文本为txt",
        data=buffer,
        file_name="recognized_text.txt",
        mime="text/plain"
    )

st.title("千锋互联数据分析智能体")
option = st.radio("请选择数据文件类型:", ("Excel", "CSV", "截图"))
file_type = "xlsx" if option == "Excel" else "csv"

if option == "截图":
    screenshot = st.file_uploader("上传你的截图", type=["png", "jpg", "jpeg"])

    if screenshot:
        # 选择OCR语言
        lang = st.selectbox(
            "选择截图语言",
            ["eng", "chi_sim", "jpn", "kor", "spa", "fra", "deu"]
        )

        # 显示原始图像
        image = Image.open(screenshot)
        st.image(image, caption="原始截图", use_column_width=True)

        # 图像预处理
        processed_image = preprocess_image(image, lang)
        st.image(processed_image, caption="预处理后的图像", use_column_width=True)

        # OCR识别，尝试不同的OCR参数以提高准确性
        custom_config = r'--oem 3 --psm 6'  # 可选的OCR引擎模式和页面分割模式
        with st.spinner("正在识别图像内容..."):
            try:
                text = pytesseract.image_to_string(processed_image, lang=lang, config=custom_config)
                st.write("识别到的文本内容：")
                st.code(text, language="text")

                # 保存为txt并提供下载链接
                save_text_as_txt(text)

                # 表格解析
                separator = detect_table_format(text)
                try:
                    df = parse_table(text, separator)
                    st.write("解析后的表格数据：")
                    st.dataframe(df)

                    # 保存识别结果到会话状态
                    st.session_state["ocr_df"] = df

                except Exception as e:
                    st.warning(f"表格解析失败: {str(e)}")
                    st.session_state["ocr_df"] = None

            except Exception as e:
                st.error(f"OCR识别失败: {str(e)}")

        # 分析查询
        query = st.text_area(
            "请输入你关于以上识别文本的问题或数据可视化需求：",
            disabled=not text
        )
else:
    data = st.file_uploader(f"上传你的{option}数据文件", type=file_type)
    if data:
        if file_type == "xlsx":
            st.session_state["df"] = pd.read_excel(data, sheet_name='data')
        else:
            st.session_state["df"] = pd.read_csv(data)
        with st.expander("原始数据"):
            st.dataframe(st.session_state["df"])
        query = st.text_area(
            "请输入你关于以上数据集的问题或数据可视化需求：",
            disabled="df" not in st.session_state
        )

button = st.button("生成回答")
if button:
    if option == "截图" and "ocr_df" not in st.session_state:
        st.info("请先完成截图识别和解析")
        st.stop()

    if option != "截图" and "df" not in st.session_state:
        st.info("请先上传数据文件")
        st.stop()

    if query:
        with st.spinner("AI正在思考中，请稍等..."):
            try:
                if option == "截图":
                    result = dataframe_agent(st.session_state["ocr_df"], query)
                else:
                    result = dataframe_agent(st.session_state["df"], query)

                if "answer" in result:
                    st.write("分析结果：")
                    st.markdown(result["answer"])

                if "table" in result:
                    st.write("数据表格：")
                    st.table(pd.DataFrame(
                        result["table"]["data"],
                        columns=result["table"]["columns"]
                    ))

                if "bar" in result:
                    st.write("柱状图：")
                    create_chart(result["bar"], "bar")

                if "line" in result:
                    st.write("折线图：")
                    create_chart(result["line"], "line")

                if "scatter" in result:
                    st.write("散点图：")
                    create_chart(result["scatter"], "scatter")

            except Exception as e:
                st.error(f"分析失败: {str(e)}")
                st.exception(e)