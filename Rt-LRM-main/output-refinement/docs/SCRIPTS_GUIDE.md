# 输出优化代码库脚本说明及vLLM支持情况

本文档详细说明`output-refinement`目录下各脚本的功能及其对本地部署模型（vLLM）的支持情况。

---

## 📊 脚本功能总览

| 脚本 | 功能类别 | 核心功能 | 使用LLM | vLLM支持 | 支持任意模型名 |
|------|----------|----------|---------|----------|----------------|
| `filtering.py` | 核心实验 | 主实验脚本，运行LCO及基线方法 | ✅ | ✅ | ✅ |
| `async_filtering.py` | 核心实验 | 异步版本实验脚本 | ✅ | ✅ | ✅ |
| `run_experiments.py` | 批量运行 | 批量运行多个seed实验 | ✅ (调用filtering.py) | ✅ | ✅ |
| `Genetic_optimization.py` | 核心实验 | 遗传算法优化实验 | ✅ | ❌ | ✅ |
| `pairwise_voting.py` | 评估 | 成对比较评估 | ✅ | ✅ | ✅ |
| `vote_experiments.py` | 批量评估 | 批量运行pairwise_voting | ✅ (调用pairwise_voting.py) | ❌ | ✅ |
| `toxicity.py` | 评估 | Google Perspective API毒性评分 | ❌ (调用API) | N/A | N/A |
| `compute_ICRH.py` | 分析 | 计算TGR（毒性增长率） | ❌ | N/A | N/A |
| `compute_helpfulness.py` | 分析 | 计算有用性评分 | ❌ | N/A | N/A |
| `plot_results.py` | 可视化 | 结果可视化 | ❌ | N/A | N/A |
| `draw.py` | 可视化 | 绘制KDE和箱线图 | ❌ | N/A | N/A |
| `test.py` | 可视化 | 毒性分数可视化工具 | ❌ | N/A | N/A |
| `test_vllm.py` | 测试 | VLLM后端测试脚本 | ✅ | ✅ | ✅ |
| `vllm_backend.py` | 后端 | VLLM核心模块 | ✅ | ✅ | ✅ |
| `api_keys.py` | 配置 | API密钥管理 | N/A | N/A | N/A |

---

## 🔬 核心实验脚本（使用LLM）

### 1. filtering.py ⭐ 主要脚本

**功能：**
- 运行LCO（Learned Constraint Optimization）实验
- 支持基线方法：base, self_defense, goal_priority
- 实现进化采样（交叉、变异、安全投票）
- 支持单/多智能体配置

**vLLM支持：** ✅ 完整支持

**使用方法：**
```bash
# 云端API
python filtering.py \
    --agent_model gpt-4 \
    --method LCO \
    --experiment reward_hacking

# 本地vLLM模型
python filtering.py \
    --agent_model "/path/to/local/model" \
    --method LCO \
    --experiment reward_hacking \
    --vllm_tp_size 4

# 任意模型名称
python filtering.py \
    --agent_model "vllm:///data/my-custom-model" \
    --method LCO
```

**支持的模型格式：**
- 云端API: `gpt-4`, `gpt-3.5-turbo`, `qwen`, `llama`
- 本地路径: `/path/to/model`, `./models/llama-3`
- vLLM协议: `vllm:///path/to/model`
- HuggingFace: `Qwen/Qwen2.5-7B-Instruct`

---

### 2. async_filtering.py

**功能：**
- 异步版本的主实验脚本
- 使用`aiohttp`进行并发API调用
- 适合高API速率限制场景

**vLLM支持：** ✅ 已支持（通过线程池包装）

**使用方法：**
```bash
# 云端API
python async_filtering.py \
    --agent_model gpt-4 \
    --experiment reward_hacking

# 本地vLLM
python async_filtering.py \
    --agent_model "/path/to/local/model" \
    --experiment reward_hacking \
    --vllm_tp_size 4
```

**实现方式：**
使用`asyncio.to_thread`（Python 3.9+）或`loop.run_in_executor`包装同步的vLLM调用，使其与异步框架兼容。

---

### 3. Genetic_optimization.py

**功能：**
- 遗传算法优化实验
- 实现交叉、变异、选择操作
- 使用异步请求加速进化过程

**vLLM支持：** ❌ 暂不支持

**原因：**
- 类似`async_filtering.py`，使用`aiohttp`直接发送请求
- 未集成vLLM后端

**修改建议：**
需要将异步HTTP调用改为支持vLLM的方式。

---

### 4. pairwise_voting.py

**功能：**
- 对实验结果进行成对比较评估
- 使用LLM作为评判者对输出进行排名
- 计算成对比较分数

**vLLM支持：** ✅ 已支持（通过filtering.py导入）

**使用方法：**
```bash
# 使用云端API评判
python pairwise_voting.py \
    --data_json path/to/results \
    --judge_model gpt-3.5-turbo

# 使用本地vLLM评判
python pairwise_voting.py \
    --data_json path/to/results \
    --judge_model "/path/to/local/model"
```

**实现方式：**
通过调用`filtering.judge_responses`和`async_filtering.async_judge_responses`，间接支持vLLM。

---

### 5. run_experiments.py

**功能：**
- 批量运行多个seed的实验
- 使用`ThreadPoolExecutor`并行执行
- 调用`filtering.py`进行实际实验

**vLLM支持：** ✅ 支持（通过filtering.py）

**使用方法：**
```bash
python run_experiments.py \
    --agent_model "/path/to/local/model" \
    --method LCO \
    --vllm_tp_size 4
```

**注意：** 由于`run_experiments.py`调用`filtering.py`，它继承了`filtering.py`的vLLM支持能力。

---

### 6. vote_experiments.py

**功能：**
- 批量对结果文件夹中的所有JSON文件运行评估
- 调用`pairwise_voting.py`进行评估

**vLLM支持：** ✅ 已支持（通过pairwise_voting.py）

---

## 📈 评估与分析脚本（不使用LLM）

### 7. toxicity.py

**功能：**
- 使用Google Perspective API计算毒性分数
- 测量毒性、淫秽、侮辱、身份攻击等指标

**API调用：** Google Perspective API（非LLM）

**vLLM支持：** N/A

---

### 8. compute_ICRH.py

**功能：**
- 计算TGR（Toxicity Growth Rate）
- 使用Kendall's tau进行趋势分析
- 统计显著性检验（t-test, Wilcoxon）

**API调用：** 无

**vLLM支持：** N/A

---

### 9. compute_helpfulness.py

**功能：**
- 计算有用性评分
- 统计非拒绝响应的数量
- 基于关键词匹配（不涉及LLM）

**API调用：** 无

**vLLM支持：** N/A

---

## 📊 可视化脚本

### 10. plot_results.py

**功能：**
- 可视化参与度分数和毒性趋势
- 使用`choix`库进行参数估计
- 支持多智能体数据展示

**API调用：** 无

**vLLM支持：** N/A

---

### 11. draw.py

**功能：**
- 绘制Kendall Tau分布的KDE曲线和箱线图
- 显著性检验标注

**API调用：** 无

**vLLM支持：** N/A

---

### 12. test.py

**功能：**
- 毒性分数可视化工具
- 支持单/多智能体数据
- 命令行批处理

**API调用：** 无

**vLLM支持：** N/A

---

## 🔧 辅助模块

### 13. vllm_backend.py

**功能：**
- VLLM核心后端模块
- 支持多卡并行推理（Tensor Parallelism）
- 提供与OpenAI API兼容的接口

**主要函数：**
- `query_vllm()`: 单条推理
- `batch_query_vllm()`: 批量推理
- `VLLMChatModel`: LangChain兼容类
- `is_vllm_model()`: 检测模型类型
- `format_prompt()`: 格式化不同模型的提示

**vLLM支持：** ✅ 原生支持

---

### 14. api_keys.py

**功能：**
- 管理API密钥
- 从环境变量或.env文件加载
- 支持多种API提供商的配置

**配置项：**
- `OPENAI_API_KEY`
- `GPT4_API_KEY` / `GPT35_API_KEY`
- `ANTHROPIC_API_KEY`
- `PERSPECTIVE_API_KEY`
- `QWEN_API_KEY`
- `LLAMA_API_KEY`
- 各种`API_BASE`URL

---

## 📝 场景模块

### 15. reward_hacking/utils.py

**功能：**
- 奖励黑客场景（推文生成）的提示词模板
- 包括：自思考、交叉、变异、评判提示
- 数据加载函数（corporations, topics）

**API调用：** 无（仅提示词模板）

**vLLM支持：** N/A（被filtering.py调用）

---

### 16. optimization/utils.py

**功能：**
- 通用优化场景的提示词模板
- 数据加载函数

**API调用：** 无

**vLLM支持：** N/A

---

## 🎯 vLLM支持路线图

### 完全支持vLLM的脚本
- ✅ `filtering.py` - 主实验脚本
- ✅ `async_filtering.py` - 异步实验脚本
- ✅ `run_experiments.py` - 批量运行（依赖filtering.py）
- ✅ `pairwise_voting.py` - 成对比较评估
- ✅ `vote_experiments.py` - 批量评估（依赖pairwise_voting.py）
- ✅ `vllm_backend.py` - 后端模块
- ✅ `test_vllm.py` - 测试脚本

### vLLM支持状态更新
| 脚本 | 支持状态 | 备注 |
|------|----------|------|
| `filtering.py` | ✅ 完全支持 | 原生支持vLLM |
| `async_filtering.py` | ✅ 已支持 | 使用线程池包装同步vLLM调用 |
| `pairwise_voting.py` | ✅ 已支持 | 通过filtering.py间接支持 |
| `vote_experiments.py` | ✅ 已支持 | 通过pairwise_voting.py间接支持 |
| `run_experiments.py` | ✅ 已支持 | 调用filtering.py |
| `Genetic_optimization.py` | ❌ 暂不支持 | 需要类似async_filtering的修改 |

**注意：** `Genetic_optimization.py`目前暂不支持vLLM，如有需要可参考`async_filtering.py`的修改方式进行更新。

---

## 🚀 推荐使用流程（vLLM）

### 单卡本地实验

```bash
# 1. 安装依赖
pip install vllm torch transformers

# 2. 设置环境变量
export OPENAI_API_KEY=your-judge-model-key  # 评判模型仍可使用云端API

# 3. 运行实验
python filtering.py \
    --experiment reward_hacking \
    --agent_model "/path/to/local/llama-3-8b" \
    --judge_model "gpt-3.5-turbo" \
    --method LCO \
    --n_rounds 11 \
    --seed 0
```

### 多卡本地实验

```bash
# 使用4张GPU
python filtering.py \
    --experiment reward_hacking \
    --agent_model "/path/to/local/qwen-72b" \
    --judge_model "gpt-3.5-turbo" \
    --method LCO \
    --vllm_tp_size 4 \
    --n_rounds 11 \
    --seed 0
```

### 批量实验

```bash
# 运行10个不同seed的实验
python run_experiments.py \
    --agent_model "/path/to/local/model" \
    --method LCO \
    --judge_model "gpt-3.5-turbo" \
    --n_judges 3 \
    --agent_idx -1 \
    --ab_test False
```

---

## ⚠️ 注意事项

### 1. 混合部署
- Agent模型可以使用本地vLLM部署
- Judge模型通常仍使用云端API（如GPT-3.5）
- 这种方式可以节省大量API成本

### 2. 显存需求
- 确保GPU显存足够，或使用`--vllm_tp_size`增加GPU数量
- 使用`--vllm_gpu_util`调整GPU内存利用率

### 3. 模型名称任意性
- vLLM支持任意模型路径或HuggingFace模型名
- 通过`is_vllm_model()`函数自动检测
- 支持自定义模型架构（需正确配置tokenizer和对话模板）

### 4. 已知限制
- `async_filtering.py`和`Genetic_optimization.py`暂不支持vLLM
- 如需异步高并发，建议使用`filtering.py`的多进程版本（`run_experiments.py`）

---

## 📚 参考文档

- [VLLM_USAGE.md](VLLM_USAGE.md) - VLLM详细使用说明
- [VLLM_QUICK_START.md](VLLM_QUICK_START.md) - 快速上手指南
- [API_SETUP.md](API_SETUP.md) - API配置指南
