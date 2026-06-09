# 使用任意模型名称指南

本文档说明如何在输出优化代码中使用任意自定义模型名称。

## 快速开始

### 方法1：使用环境变量（推荐）

假设你想使用模型名称 `my-custom-model`：

```bash
# 1. 设置环境变量
export MY_CUSTOM_MODEL_API_KEY="sk-your-api-key"
export MY_CUSTOM_MODEL_API_BASE="https://api.your-provider.com/v1"

# 2. 运行实验
python filtering.py \
    --agent_model "my-custom-model" \
    --method LCO \
    --experiment reward_hacking
```

### 方法2：使用 .env 文件

编辑 `.env` 文件：

```env
MY_CUSTOM_MODEL_API_KEY=sk-your-api-key
MY_CUSTOM_MODEL_API_BASE=https://api.your-provider.com/v1
```

然后直接运行：

```bash
python filtering.py \
    --agent_model "my-custom-model" \
    --method LCO \
    --experiment reward_hacking
```

### 方法3：运行时注册（Python代码中）

```python
from api_keys import register_model

# 注册自定义模型
register_model(
    "my-custom-model",
    api_key="sk-your-api-key",
    api_base="https://api.your-provider.com/v1"
)

# 然后在filtering.py中正常使用
```

---

## 命名规则

环境变量命名时，模型名称中的特殊字符会被替换：

| 模型名称 | 环境变量名称 |
|---------|-------------|
| `gpt-4` | `GPT_4_API_KEY` |
| `gpt-3.5-turbo` | `GPT_3_5_TURBO_API_KEY` |
| `my.model` | `MY_MODEL_API_KEY` |
| `deepseek-chat` | `DEEPSEEK_CHAT_API_KEY` |

---

## 常用模型配置示例

### DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"

python filtering.py --agent_model "deepseek-chat" --method LCO
```

### 智谱GLM-4

```bash
export GLM_4_API_KEY="sk-your-glm-key"
export GLM_4_API_BASE="https://open.bigmodel.cn/api/paas/v4"

python filtering.py --agent_model "glm-4" --method LCO
```

### 通义千问 (DashScope)

```bash
export QWEN_API_KEY="sk-your-qwen-key"
export QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

python filtering.py --agent_model "qwen-turbo" --method LCO
```

### 第三方API代理

```bash
export MY_PROXY_API_KEY="sk-your-key"
export MY_PROXY_API_BASE="https://api.proxy.com/v1"

python filtering.py --agent_model "my-proxy" --method LCO
```

---

## 完整示例

### 示例1：使用DeepSeek运行完整实验

```bash
# 设置API
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"

# 运行单智能体实验
python filtering.py \
    --experiment reward_hacking \
    --agent_model "deepseek-chat" \
    --judge_model "gpt-3.5-turbo" \
    --method LCO \
    --n_rounds 11 \
    --seed 0 \
    --agent_idx 0

# 运行多智能体实验
python filtering.py \
    --experiment reward_hacking \
    --agent_model "deepseek-chat" \
    --judge_model "gpt-3.5-turbo" \
    --method LCO \
    --n_rounds 11 \
    --seed 0 \
    --agent_idx -1
```

### 示例2：批量运行多个自定义模型

```bash
# 创建脚本 run_custom_models.sh
#!/bin/bash

MODELS=("deepseek-chat" "glm-4" "qwen-turbo")
METHODS=("base" "LCO" "self_defense")

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        python filtering.py \
            --experiment reward_hacking \
            --agent_model "$model" \
            --method "$method" \
            --n_rounds 11 \
            --seed 0
    done
done
```

---

## 配置优先级

当获取模型配置时，系统按以下优先级查找：

1. **特定模型配置**（最高优先级）
   - 查找 `{MODEL_NAME}_API_KEY` 和 `{MODEL_NAME}_API_BASE`

2. **预设模型配置**
   - `claude-*` → 使用 Anthropic API
   - `gpt-4*` → 使用 GPT4_API_KEY
   - `gpt-3.5*` → 使用 GPT35_API_KEY
   - `qwen*` → 使用 QWEN_API_KEY
   - `llama*` → 使用 LLAMA_API_KEY

3. **默认配置**（最低优先级）
   - 使用 `DEFAULT_API_KEY` 和 `DEFAULT_API_BASE`

---

## 故障排除

### 问题：API key not found

**症状：**
```
ValueError: No API key found for model: my-model
```

**解决方案：**

1. 检查环境变量是否设置正确：
```bash
echo $MY_MODEL_API_KEY
echo $MY_MODEL_API_BASE
```

2. 确认命名正确（注意大小写和特殊字符）：
```bash
# 模型名称: my-custom-model
# 环境变量: MY_CUSTOM_MODEL_API_KEY
export MY_CUSTOM_MODEL_API_KEY="sk-xxx"
```

3. 在Python中检查配置：
```python
from api_keys import get_model_config
config = get_model_config("my-model")
print(config)
```

### 问题：API base URL 错误

**症状：**
```
Error: Connection error to https://xxx.com/v1
```

**解决方案：**

1. 检查API base URL是否完整：
```bash
# 正确的格式
export MY_MODEL_API_BASE="https://api.example.com/v1"

# 错误的格式（缺少 /v1）
export MY_MODEL_API_BASE="https://api.example.com"
```

2. 测试API连接：
```bash
curl https://api.example.com/v1/models \
  -H "Authorization: Bearer $MY_MODEL_API_KEY"
```

---

## 高级用法

### 混合使用多个模型

```bash
# Agent使用自定义模型
export MY_MODEL_API_KEY="sk-xxx"
export MY_MODEL_API_BASE="https://api.custom.com/v1"

# Judge使用OpenAI
export OPENAI_API_KEY="sk-yyy"

python filtering.py \
    --agent_model "my-model" \
    --judge_model "gpt-3.5-turbo" \
    --method LCO
```

### 使用本地和云端模型混合

```bash
# 本地vLLM模型
python filtering.py \
    --agent_model "/path/to/local/model" \
    --judge_model "gpt-3.5-turbo" \
    --method LCO

# 自定义云端模型
export CUSTOM_API_KEY="sk-xxx"
export CUSTOM_API_BASE="https://api.custom.com/v1"

python filtering.py \
    --agent_model "custom" \
    --judge_model "/path/to/local/model" \
    --method LCO
```

---

## 相关文档

- [API_SETUP.md](API_SETUP.md) - API配置指南
- [VLLM_USAGE.md](VLLM_USAGE.md) - VLLM本地部署指南
- [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) - 脚本功能说明
