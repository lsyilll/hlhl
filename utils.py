import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ---------------------------
# 自定义提示模板（支持多种图表类型）
# ---------------------------
PROMPT_TEMPLATE = """
你是一位专业的数据分析助手，用户将提供一个数据集，你的任务是根据用户的问题生成对应的分析结果。
请严格按照以下格式返回结果（确保JSON格式正确，不包含任何多余文本）：

- 纯文字回答：
  {{"answer": "简明答案（不超过50字）"}}

- 表格数据：
  {{"table": {{"columns": ["列名1", "列名2"], "data": [["值1", 100], ["值2", 200]]}}}}

- 柱状图数据：
  {{"bar": {{"columns": ["类别A", "类别B"], "data": [35, 42]}}}}

- 折线图数据：
  {{"line": {{"columns": ["月份1", "月份2"], "data": [89, 120]}}}}

- 散点图数据：
  {{"scatter": {{"columns": ["X轴标签", "Y轴标签"], "data": [23, 45, 67]}}}}

- 面积图数据：
  {{"area": {{"columns": ["季度1", "季度2"], "data": [150, 200]}}}}

- 饼图数据：
  {{"pie": {{"columns": ["产品A", "产品B"], "data": [60, 40]}}}}

注意：
1. 所有字符串使用英文双引号
2. 数值类型不得加引号
3. 确保数组闭合完整
4. 只返回JSON，不包含其他内容
5. 饼图的"data"应为数值列表，对应各部分的占比数值

用户问题：{{input}}
"""


def dataframe_agent(df, query):
    """创建数据分析智能体（支持多图表类型）"""
    # 加载环境变量（建议将API Key存入.env文件）
    load_dotenv()

    # 初始化模型（可替换为OpenAI/GPT-4或其他LLM）
    model = ChatOpenAI(
        model="gpt-4o-mini",  # 使用GPT-4o-mini获取更准确的图表推荐
        api_key="your-openai-api-key",  # 请替换为实际API Key
        base_url="https://twapi.openai-hk.com/v1",
        temperature=0.1,  # 低温度确保输出格式稳定
        max_tokens=1024
    )

    # 创建Pandas数据代理
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        agent="pandas-v2",  # 使用最新的Pandas代理版本
        handle_parsing_errors=True,
        max_iterations=5,  # 限制最大思考次数避免超时
        early_stopping_method="generate",
        verbose=False  # 关闭代理调试日志
    )

    # 生成带用户查询的完整提示
    prompt = PROMPT_TEMPLATE.replace("{{input}}", query)

    try:
        # 调用代理并解析结果
        response = agent.run(prompt)
        return json.loads(response)  # 解析JSON结果

    except Exception as e:
        raise ValueError(f"数据分析失败：{str(e)}")