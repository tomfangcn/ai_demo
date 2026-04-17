"""LangChain Demo with DeepSeek Model"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

# 加载环境变量
load_dotenv()

# 获取 DeepSeek API Key
api_key = os.getenv("DS_API_KEY_PY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 DS_API_KEY_PY")

# 创建 DeepSeek 模型实例
# DeepSeek 兼容 OpenAI API，使用 openai_base_url 配置

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=SecretStr(api_key),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
)

# 简单对话示例
def chat(message: str) -> str:
    """简单的对话函数"""
    response = llm.invoke([HumanMessage(content=message)])
    return str(response.content)

# Chain 示例 - 使用 LCEL
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 创建 prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手。"),
    ("human", "{input}")
])

# 创建 chain
chain = prompt | llm | StrOutputParser()

def chat_chain(question: str) -> str:
    """使用 Chain 的对话函数"""
    return chain.invoke({"input": question})

# 测试
if __name__ == "__main__":
    # 测试简单对话
    print("=== 测试简单对话 ===")
    response = chat("你好，请介绍一下你自己")
    print(f"回复: {response}\n")

    # 测试 Chain
    print("=== 测试 Chain ===")
    response = chat_chain("什么是 LangChain?")
    print(f"回复: {response}")