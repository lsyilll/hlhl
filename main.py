import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from utils import dataframe_agent

def create_chart(input_data, chart_type):
    """生成统计图表（支持多种类型）"""
    df_data = pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    )
    df_data.set_index("x", inplace=True)
    
    if chart_type == "bar":
        st.bar_chart(df_data, use_container_width=True)
    elif chart_type == "line":
        plt.figure(figsize=(10, 6))
        plt.plot(df_data.index, df_data["y"], marker="o", linestyle="--", color="#1f77b4")
        plt.title("折线图", fontsize=14)
        plt.xlabel("类别", fontsize=12)
        plt.ylabel("数值", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(plt.gcf(), clear_figure=True)
    elif chart_type == "scatter":
        plt.figure(figsize=(10, 6))
        plt.scatter(df_data.index, df_data["y"], color="#ff7f0e", alpha=0.8, s=100)
        plt.title("散点图", fontsize=14)
        plt.xlabel("类别", fontsize=12)
        plt.ylabel("数值", fontsize=12)
        st.pyplot(plt.gcf(), clear_figure=True)
    elif chart_type == "area":
        plt.figure(figsize=(10, 6))
        plt.fill_between(df_data.index, df_data["y"], color="#2ca02c", alpha=0.3)
        plt.plot(df_data.index, df_data["y"], color="#2ca02c", marker="o")
        plt.title("面积图", fontsize=14)
        st.pyplot(plt.gcf(), clear_figure=True)
    elif chart_type == "pie":
        if len(df_data) == 0:
            st.error("饼图需要至少一个数据点")
            return
        plt.figure(figsize=(8, 8))
        plt.pie(df_data["y"], labels=df_data.index, autopct="%1.1f%%", startangle=90)
        plt.title("饼图", fontsize=14)
        plt.axis("equal")  # 保证饼图为圆形
        st.pyplot(plt.gcf(), clear_figure=True)
    else:
        st.warning("暂不支持该图表类型")

st.title("千锋互联数据分析智能体")
option = st.radio("请选择数据文件类型:", ("Excel", "CSV"))
file_type = "xlsx" if option == "Excel" else "csv"
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
    
    if button and "df" not in st.session_state:
        st.info("请先上传数据文件")
        st.stop()
    
    if query:
        with st.spinner("AI正在思考中，请稍等..."):
            result = dataframe_agent(st.session_state["df"], query)
            
            # 提取图表数据（兼容多种类型）
            chart_data = {}
            supported_types = ["bar", "line", "scatter", "area", "pie"]  # 定义支持的类型
            for typ in supported_types:
                if typ in result:
                    chart_data = result[typ]
                    ai_recommended_type = typ
                    break  # 优先使用AI推荐的类型
            
            # 新增多类型选择下拉框
            if chart_data:
                st.subheader("选择图表类型")
                selected_type = st.selectbox(
                    "请选择图表类型",
                    options=supported_types,
                    index=supported_types.index(ai_recommended_type) if ai_recommended_type in supported_types else 0
                )
                
                # 生成用户选择的图表
                create_chart(chart_data, selected_type)
            
            # 显示文本和表格结果
            if "answer" in result:
                st.write(result["answer"])
            if "table" in result:
                st.table(pd.DataFrame(result["table"]["data"], columns=result["table"]["columns"]))