from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from pathlib import Path
import os
import logging.config
import yaml

# 获取当前脚本文件所在目录
script_dir = Path(__file__).parent
# 构建配置文件的绝对路径（假设配置文件与脚本在同一目录下）
config_path = script_dir.parent / 'logging.yaml'

# 配置日志格式和输出级别
# 读取 YAML 配置文件
with open(config_path, 'r', encoding='utf-8') as f:
    conf = yaml.safe_load(f)

# 应用配置
logging.config.dictConfig(conf)

# 获取自定义 logger
logger = logging.getLogger('apiLogger')

API_KEY = os.getenv("DS_API_KEY_PY")

if API_KEY is None:
    raise ValueError("环境变量 OPENAI_API_KEY 未设置")
try:
    client = OpenAI(api_key=API_KEY,base_url="https://api.deepseek.com")
except Exception as e:
    logger.critical(f"未预期的错误：{e}", exc_info=True)

try:
    resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role":"system","content":"你是一个乐于助人的助手。"},
        {"role":"user","content":"介绍一下python 如何打包。"}
    ],
    temperature=0.8,
    max_tokens=128
)
except RateLimitError as e:
    logger.error(f"请求频率超限：{e}", exc_info=True)
except APIConnectionError as e:
    logger.error(f"网络连接失败：{e}", exc_info=True)
except APIError as e:
    logger.error(f"API 返回错误：{e}", exc_info=True)
except Exception as e:
    logger.critical(f"未预期的错误：{e}", exc_info=True)
else:
    logger.info("API 调用成功")

# 输出助手回复
print(resp.choices[0].message.content)