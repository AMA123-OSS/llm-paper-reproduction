"""
API Keys Configuration

This module loads API keys from environment variables.
Create a .env file in the project root or set environment variables directly.

Example .env file:
    OPENAI_API_KEY=sk-your-openai-key
    ANTHROPIC_API_KEY=sk-your-anthropic-key
    PERSPECTIVE_API_KEY=your-perspective-key
    GPT_API_BASE=https://api.openai.com/v1
    GPT4_API_KEY=sk-your-gpt4-key
    GPT3.5_API_KEY=sk-your-gpt35-key
    QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
    QWEN_API_KEY=sk-your-qwen-key
    LLAMA_API_BASE=https://api.together.xyz/v1
    LLAMA_API_KEY=sk-your-llama-key
    
    # 自定义模型配置（支持任意模型名称）
    MYMODEL_API_KEY=sk-your-key
    MYMODEL_API_BASE=https://api.example.com/v1
"""

import os
from dotenv import load_dotenv

# 加载.env文件（如果存在）
load_dotenv("../.env")
load_dotenv(".env")

# OpenAI API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# GPT-4 and GPT-3.5 specific keys (optional, fallback to OPENAI_API_KEY)
GPT4_API_KEY = os.getenv("GPT4_API_KEY", OPENAI_API_KEY)
GPT35_API_KEY = os.getenv("GPT35_API_KEY", OPENAI_API_KEY)

# API Base URLs
GPT_API_BASE = os.getenv("GPT_API_BASE", "https://api.openai.com/v1")

# Anthropic API Key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Google Perspective API Key
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", "")

# Qwen API
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# Llama API
LLAMA_API_BASE = os.getenv("LLAMA_API_BASE", "https://api.together.xyz/v1")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")

# Default API configuration (used as fallback for custom models)
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", OPENAI_API_KEY)
DEFAULT_API_BASE = os.getenv("DEFAULT_API_BASE", GPT_API_BASE)

def get_model_config(model_name: str) -> dict:
    """
    获取任意模型的API配置
    
    配置优先级：
    1. 特定模型配置：{MODEL_NAME}_API_KEY 和 {MODEL_NAME}_API_BASE
    2. 预设模型配置（gpt, claude, qwen, llama等）
    3. 默认配置：DEFAULT_API_KEY 和 DEFAULT_API_BASE
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        dict: 包含api_key, api_base, model_type的配置字典
    """
    # 标准化模型名称（去除路径、转为大写）
    model_upper = model_name.upper().replace("-", "_").replace(".", "_")
    model_clean = model_name.split("/")[-1]  # 如果是路径，取最后一部分
    model_clean_upper = model_clean.upper().replace("-", "_").replace(".", "_")
    
    # 首先检查是否有特定模型配置（支持多种命名格式）
    for prefix in [model_upper, model_clean_upper, model_clean.upper()]:
        api_key = os.getenv(f"{prefix}_API_KEY", "")
        api_base = os.getenv(f"{prefix}_API_BASE", "")
        if api_key and api_base:
            return {
                "api_key": api_key,
                "api_base": api_base,
                "model_type": "custom",
                "model_name": model_name
            }
        elif api_key:
            # 有key但没有base，使用默认base
            return {
                "api_key": api_key,
                "api_base": DEFAULT_API_BASE,
                "model_type": "custom",
                "model_name": model_name
            }
    
    # 预设模型配置
    model_lower = model_name.lower()
    
    if model_lower.startswith("claude"):
        return {
            "api_key": ANTHROPIC_API_KEY,
            "api_base": "https://api.anthropic.com/v1",
            "model_type": "anthropic",
            "model_name": model_name
        }
    elif model_lower.startswith("gpt-4"):
        return {
            "api_key": GPT4_API_KEY if GPT4_API_KEY else OPENAI_API_KEY,
            "api_base": GPT_API_BASE,
            "model_type": "openai",
            "model_name": model_name
        }
    elif model_lower.startswith("gpt-3.5") or model_lower.startswith("gpt"):
        return {
            "api_key": GPT35_API_KEY if GPT35_API_KEY else OPENAI_API_KEY,
            "api_base": GPT_API_BASE,
            "model_type": "openai",
            "model_name": model_name
        }
    elif model_lower.startswith("qwen"):
        return {
            "api_key": QWEN_API_KEY,
            "api_base": QWEN_API_BASE,
            "model_type": "openai",  # Qwen使用OpenAI兼容格式
            "model_name": model_name
        }
    elif model_lower.startswith("llama"):
        return {
            "api_key": LLAMA_API_KEY,
            "api_base": LLAMA_API_BASE,
            "model_type": "openai",  # Together等平台使用OpenAI兼容格式
            "model_name": model_name
        }
    
    # 默认配置（使用环境变量中的默认设置）
    return {
        "api_key": DEFAULT_API_KEY,
        "api_base": DEFAULT_API_BASE,
        "model_type": "openai",
        "model_name": model_name
    }


def register_model(model_name: str, api_key: str, api_base: str = None):
    """
    运行时注册自定义模型配置
    
    Args:
        model_name: 模型名称
        api_key: API密钥
        api_base: API基础URL（可选，默认使用DEFAULT_API_BASE）
    """
    prefix = model_name.upper().replace("-", "_").replace(".", "_")
    os.environ[f"{prefix}_API_KEY"] = api_key
    if api_base:
        os.environ[f"{prefix}_API_BASE"] = api_base
    print(f"Registered model: {model_name}")


# Validate that required keys are set
def check_api_key(key_name, key_value):
    """Check if an API key is set, print warning if not"""
    if not key_value:
        print(f"Warning: {key_name} is not set. Please set it as an environment variable or in .env file")
        return False
    return True


# Optional: Validate keys on import (can be disabled)
VALIDATE_ON_IMPORT = os.getenv("VALIDATE_API_KEYS", "false").lower() == "true"

if VALIDATE_ON_IMPORT:
    check_api_key("OPENAI_API_KEY", OPENAI_API_KEY)
    check_api_key("PERSPECTIVE_API_KEY", PERSPECTIVE_API_KEY)
