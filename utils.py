import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

PROMPT_TEMPLATE = """你是一位数据分析助手，你的回应内容取决于用户的请求内容，请按照下面的步骤处理用户请求：

1. 思考阶段 (Thought) ：先分析用户请求类型（文字回答/表格/图表），并验证数据类型是否匹配。
2. 行动阶段 (Action) ：根据分析结果选择以下严格对应的格式。
   - 纯文字回答: 
     {"answer": "不超过50个字符的明确答案"}

   - 表格数据：  
     {"table":{"columns":["列名1", "列名2", ...], "data":[["第一行值1", "值2", ...], ["第二行值1", "值2", ...]]}}

   - 柱状图 
     {"bar":{"columns": ["A", "B", "C", ...], "data":[35, 42, 29, ...]}}

   - 折线图 
     {"line":{"columns": ["A", "B", "C", ...], "data": [35, 42, 29, ...]}}

   - 散点图 
     {"scatter":{"columns": ["A", "B", "C", ...], "data": [35, 42, 29, ...]}}

3. 格式校验要求
   - 字符串值必须使用英文双引号
   - 数值类型不得添加引号
   - 确保数组闭合无遗漏

   错误案例：{'columns':['Product', 'Sales'], data:[[A001, 200]]}  
   正确案例：{"columns":["product", "sales"], "data":[["A001", 200]]}

注意：响应数据的"output"中不要有换行符、制表符以及其他格式符号。

当前用户请求："""


def dataframe_agent(df, query):
    load_dotenv()
    model = ChatOpenAI(
        model="deepseek-chat",
        base_url='https://api.deepseek.com',
        api_key='sk-8dca673d82b74bf59bac651337b7fba8',
        temperature=0
    )
    if df is not None:
        agent = create_pandas_dataframe_agent(
            llm=model,
            df=df,
            agent_executor_kwargs={"handle_parsing_errors": True},
            max_iterations=10,
            early_stopping_method='generate',
            allow_dangerous_code=True,
            verbose=True
        )
    else:
        # 当处理截图文本时，这里可以根据实际情况调整逻辑
        # 目前简单地认为不需要使用 DataFrame 相关的 agent
        agent = None

    prompt = PROMPT_TEMPLATE + query
    if agent:
        response = agent.invoke({"input": prompt})
    else:
        # 这里可以实现不依赖 DataFrame 的调用逻辑
        # 暂时简单模拟
        from langchain.schema import AgentFinish
        output = model.invoke(prompt)
        response = AgentFinish({"name": "FinalAnswer", "parameters": {"output": output.content}}, "")
    return json.loads(response["output"])