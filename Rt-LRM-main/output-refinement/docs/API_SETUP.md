# API 配置指南

本文档说明如何配置API密钥和环境变量。

## ⚠️ 安全警告

**永远不要将API密钥硬编码到代码中！** 本文档中的所有密钥已从代码中移除，改为从环境变量读取。

---

## 快速开始

### 方法1：使用 .env 文件（推荐）

1. 复制示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的API密钥：
```bash
OPENAI_API_KEY=sk-your-actual-key
PERSPECTIVE_API_KEY=your-actual-key
```

3. 运行代码：
```bash
python filtering.py --agent_model gpt-4 ...
```

### 方法2：使用环境变量

```bash
export OPENAI_API_KEY=sk-your-key
export PERSPECTIVE_API_KEY=your-key
export GPT_API_BASE=https://api.openai.com/v1

python filtering.py --agent_model gpt-4 ...
```

### 方法3：命令行传递（仅限base URL）

某些脚本支持通过命令行参数传递配置。

---

## 必需的API密钥

| API | 环境变量 | 用途 | 获取方式 |
|-----|----------|------|----------|
| OpenAI | `OPENAI_API_KEY` | GPT-4/GPT-3.5推理 | [OpenAI Platform](https://platform.openai.com) |
| Perspective | `PERSPECTIVE_API_KEY` | 毒性评分 | [Google Cloud](https://developers.perspectiveapi.com) |
| Anthropic | `ANTHROPIC_API_KEY` | Claude模型（可选） | [Anthropic](https://console.anthropic.com) |

---

## 可选的API配置

### 不同的API提供商

如果你使用第三方API代理（如xiaoai.plus, aigcbest等）：

```bash
export GPT_API_BASE=https://xiaoai.plus/v1
export OPENAI_API_KEY=sk-your-key-from-provider
```

### 模型特定的密钥

如果使用不同的密钥管理不同模型：

```bash
export GPT4_API_KEY=sk-your-gpt4-key
export GPT35_API_KEY=sk-your-gpt35-key
export QWEN_API_KEY=sk-your-qwen-key
export LLAMA_API_KEY=sk-your-llama-key
```

---

## 配置文件详解

### .env 文件示例

```env
# ========== OpenAI API ==========
OPENAI_API_KEY=sk-your-openai-key
GPT_API_BASE=https://api.openai.com/v1

# 模型特定密钥（可选，默认使用OPENAI_API_KEY）
GPT4_API_KEY=sk-your-gpt4-key
GPT35_API_KEY=sk-your-gpt35-key

# ========== Anthropic API ==========
ANTHROPIC_API_KEY=sk-your-anthropic-key

# ========== Google Perspective API ==========
PERSPECTIVE_API_KEY=AIzaSy-your-perspective-key

# ========== 其他模型API ==========
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-your-qwen-key

LLAMA_API_BASE=https://api.together.xyz/v1
LLAMA_API_KEY=sk-your-llama-key

# ========== 代理设置（可选） ==========
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890

# ========== VLLM设置（可选） ==========
VLLM_TP_SIZE=1
VLLM_GPU_UTIL=0.85
```

---

## 代码中的密钥管理

### 修改后的文件结构

```
output-refinement/
├── api_keys.py          # 从环境变量加载密钥
├── .env.example         # 环境变量模板
├── .env                 # 你的实际配置（不提交到git）
├── .gitignore           # 确保.env不被提交
└── filtering.py         # 使用api_keys模块
```

### api_keys.py 工作原理

```python
# 从环境变量加载
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_API_BASE = os.getenv("GPT_API_BASE", "https://api.openai.com/v1")

# 提供默认值
GPT4_API_KEY = os.getenv("GPT4_API_KEY", OPENAI_API_KEY)  # 如果没有设置，使用通用密钥
```

---

## 迁移指南

### 如果你之前使用硬编码密钥

**旧方式（不安全）：**
```python
# api_keys.py
OPENAI_API_KEY = "sk-your-hardcoded-key"
```

**新方式（安全）：**
```bash
# .env文件
OPENAI_API_KEY=sk-your-key
```

```python
# api_keys.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
```

---

## 故障排除

### 问题：API key not found

**症状：**
```
Warning: OPENAI_API_KEY is not set
```

**解决：**
```bash
# 检查环境变量是否设置
echo $OPENAI_API_KEY

# 如果没有输出，重新设置
export OPENAI_API_KEY=sk-your-key

# 或创建.env文件
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### 问题：Invalid API base URL

**症状：**
```
Error: Connection to https://xiaoai.plus/v1 failed
```

**解决：**
```bash
# 检查API base设置
echo $GPT_API_BASE

# 设置正确的base
export GPT_API_BASE=https://api.openai.com/v1
```

### 问题：Permission denied for .env

**症状：**
```
Error: Cannot read .env file
```

**解决：**
```bash
# 检查文件权限
ls -la .env

# 修改权限
chmod 600 .env
```

---

## 安全最佳实践

1. **永远不要提交密钥到Git**
   ```bash
   # .gitignore中已包含.env
   cat .gitignore | grep env
   ```

2. **使用不同的密钥用于开发和生产**
   ```bash
   # 开发环境
   cp .env .env.development
   
   # 生产环境
   cp .env .env.production
   ```

3. **定期轮换密钥**
   - 在API提供商控制台中重新生成密钥
   - 更新.env文件

4. **限制密钥权限**
   - 只给密钥必要的权限
   - 使用只读密钥进行测试

---

## 更新记录

- **2024-XX-XX**: 移除所有硬编码API密钥，改为环境变量配置
- 新增 `api_keys.py` 统一管理密钥
- 新增 `.env.example` 模板文件
- 更新 `.gitignore` 防止密钥泄露
