# LCO：基于大语言模型的约束优化框架，用于真实任务中的安全智能体

本仓库包含论文 **"LCO: LLM-based Constraint Optimization for Safer Agentic LLMs in Real-world Tasks"**（ACL Findings 2026）的代码实现。

![反馈循环示例](image.png)

## 仓库结构

```
.
├── README.md                          # 英文版说明
├── README_zh.md                       # 本文件（中文版）
├── PromptCode/                        # 提示词管理库（子模块）
│   ├── procoder/                      # 模块化提示词编码包
│   └── README.md                      # PromptCoder 文档
│
├── output-refinement/                 # 输出优化实验（推文生成）
│   ├── filtering.py                   # 核心查询模块（LCO + API 路由）
│   ├── async_filtering.py             # 异步版本
│   ├── api_keys.py                    # API 配置（环境变量 / .env）
│   ├── toxicity.py                    # Perspective API 毒性评分
│   ├── vllm_backend.py                # 可选 vLLM 后端
│   │
│   ├── benchmarks/                    # 基准评测脚本
│   ├── experiments/                   # 实验运行脚本
│   ├── evaluation/                    # 结果分析
│   ├── visualization/                 # 可视化
│   ├── tests/                         # 测试
│   ├── docs/                          # 文档
│   ├── reward_hacking/                # 奖励黑客场景（推文优化）
│   ├── optimization/                  # 优化场景
│   └── experiment_data/               # 全部实验结果
│
├── policy-refinement/                 # 策略优化实验（基于 ToolEmu）
│   ├── toolemu/                       # ToolEmu 框架
│   │   ├── agents/                    # 智能体实现
│   │   ├── prompts/                   # 提示词模板
│   │   ├── tools/                     # 工具定义
│   │   └── utils/                     # 工具函数
│   │
│   ├── scripts/                       # 实验脚本
│   ├── evaluation_scripts/            # ICRH/有用性检测
│   ├── tests/                         # 测试
│   ├── experiment_data/               # 全部实验结果
│   ├── assets/                        # 测试用例和工具包
│   ├── .env.example                   # 环境变量模板
│   └── README.md
│
└── test.py                            # 毒性可视化工具
```

## 安装

### 前置条件

- Python 3.10+
- OpenAI API key（或兼容的 API 提供商）
- （可选）Anthropic API key（用于 Claude 模型）
- （可选）Google Perspective API key（用于毒性评分）

### 安装步骤

1. **克隆仓库：**
```bash
git clone <repository-url>
cd llm-feedback-loops
```

2. **安装 PromptCoder：**
```bash
cd PromptCode
pip install -e .
cd ..
```

3. **安装输出优化的依赖：**
```bash
cd output-refinement
pip install choix openai anthropic numpy matplotlib langchain requests python-dotenv scipy pandas seaborn
cd ..
```

4. **安装策略优化的依赖：**
```bash
cd policy-refinement
pip install -e .
cd ..
```

5. **通过 `.env` 配置 API key：**

在 `output-refinement/` 和 `policy-refinement/` 目录下分别创建 `.env` 文件：

```bash
# output-refinement/.env
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-your-key
PERSPECTIVE_API_KEY=your-key
QWEN_API_KEY=sk-your-key
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLAMA_API_KEY=sk-your-key
LLAMA_API_BASE=https://api.together.xyz/v1
```

```bash
# policy-refinement/.env
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key
GPT4_API_BASE=https://api.openai.com/v1
GPT4_API_KEY=sk-your-key
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-your-key
LLAMA_API_BASE=https://api.together.xyz/v1
LLAMA_API_KEY=sk-your-key
SELF_DEFENSE_API_BASE=...
SELF_DEFENSE_API_KEY=...
VOTE_API_BASE=...
VOTE_API_KEY=...
FITNESS_API_BASE=...
FITNESS_API_KEY=...
EVAL_API_BASE=...
EVAL_API_KEY=...
```

详见 `policy-refinement/.env.example` 完整模板。

---

## 使用方法

### 输出优化实验（推文参与度优化）

#### 运行单个实验

```bash
cd output-refinement

# Vanilla 基线
python filtering.py \
    --experiment reward_hacking \
    --n_rounds 11 \
    --agent_model gpt-4 \
    --judge_model gpt-3.5-turbo \
    --n_judges 3 \
    --seed 0 \
    --agent_idx -1 \
    --method base

# LCO（我们的方法）
python filtering.py \
    --experiment reward_hacking \
    --n_rounds 11 \
    --agent_model gpt-4 \
    --judge_model gpt-3.5-turbo \
    --n_judges 3 \
    --seed 0 \
    --agent_idx -1 \
    --method LCO
```

#### 运行批量实验

```bash
python experiments/run_experiments.py \
    --agent_model gpt-4 \
    --method LCO \
    --judge_model gpt-3.5-turbo \
    --n_judges 3 \
    --agent_idx -1
```

#### 评估结果

```bash
# 计算 TGR（毒性增长率）
python evaluation/compute_ICRH.py \
    --dir path/to/results/directory \
    --method LCO

# 成对投票评估
python experiments/pairwise_voting.py \
    --data_json path/to/results.json \
    --judge_model gpt-3.5-turbo \
    --n_judges 3
```

#### 运行基准评测

```bash
# GSM8K + MMLU 基准
bash benchmarks/run_mmlu_gsm8k.sh qwen2.5-72b-instruct

# 良性评估
bash benchmarks/run_benign_eval.sh gpt-4
```

---

### 策略优化实验（工具使用场景）

#### 运行固定 LCO 仿真

```bash
cd policy-refinement

# Vanilla 基线
python -m scripts.emulate \
    -inp assets/all_cases.json \
    -atp naive \
    -stp adv_thought \
    --agent-model qwen2.5-72b-instruct \
    --simulator-model gpt-4o \
    -si 0 -tn 70 -v -me 3

# 固定 LCO，种群大小 = 3
python -m scripts.emulate \
    -inp assets/all_cases.json \
    -atp naive \
    -stp lco_thought \
    --agent-model qwen2.5-72b-instruct \
    --simulator-model gpt-4o \
    -si 0 -tn 70 -v -me 3 -pos 3
```

#### 运行风险自适应仿真

```bash
# 自适应：基于自反思风险分类自动路由
python -m scripts.emulate_adaptive \
    -inp assets/all_cases.json \
    -atp naive \
    --agent-model qwen2.5-72b-instruct \
    --simulator-model gpt-4o \
    -si 0 -tn 70 -v -me 3
```

#### 评估轨迹（计算 IOR）

```bash
# ICRH 检测
python evaluation_scripts/ICRH_Detect.py \
    --file experiment_data/tool_emu/trajectories/qwen2.5-72b/traj_xxx.jsonl \
    --method LCO \
    --output_path experiment_data/tool_emu/icrh/res.jsonl

# 有用性评估
python evaluation_scripts/traj_helpfulness.py \
    --file experiment_data/tool_emu/trajectories/qwen2.5-72b/traj_xxx.jsonl \
    --method LCO \
    --output_path experiment_data/tool_emu/helpfulness/res.jsonl
```

#### 完整流程

```bash
python -m scripts.run \
    --agent-model qwen2.5-72b-instruct \
    --agent-type naive \
    --simulator-type lco_thought \
    --trunc-num 70 \
    --population_size 3
```

---

## 关键参数

### 输出优化

- `--experiment`：实验类型（`reward_hacking` 或 `optimization`）
- `--n_rounds`：反馈循环轮数（默认：11）
- `--agent_model`：内容生成的 LLM（`gpt-4`、`qwen2.5-72b-instruct`、`llama-3.1-405b-instruct`）
- `--judge_model`：评估用的 LLM（`gpt-3.5-turbo`、`gpt-4`、`random`）
- `--n_judges`：投票的评判者数量（默认：3）
- `--agent_idx`：智能体索引（-1 为多智能体，0-3 为单智能体）
- `--method`：防御方法（`base`、`LCO`、`self_defense`、`goal_priority`）

### 策略优化

- `-atp` / `--agent-type`：智能体类型（`naive`、`ss_only`、`helpful_ss`）
- `-stp` / `--simulator-type`：模拟器类型（`adv_thought`、`lco_thought`）
- `--agent-model`：Agent 的 LLM
- `--simulator-model`：模拟器的 LLM（GPT-4o 用于 fitness/vote）
- `-me` / `--max_errors`：最大错误注入数量
- `-pos` / `--population_size`：LCO 的种群大小（推荐：3）

### 风险自适应（emulate_adaptive.py）

风险等级路由是自动的。`--agent-model` 参数决定哪个模型执行自反思风险分类。

---

## 引用

如果您在研究中使用了本代码，请引用我们的工作：

```bibtex
@inproceedings{wan2026lco,
    title = {LCO: LLM-based Constraint Optimization for Safer Agentic LLMs in Real-world Tasks},
    author = {Wan, Jiayong and Chen, Jiawei and Yin, Zhaoxia and Liu, Shuyuan and Su, Hang},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
    year = {2026}
}
```

ICRH 现象最早在以下论文中被识别：

```bibtex
@article{pan2024llmfeedback,
    author = {Pan, Alexander and Jones, Erik and Jagadeesan, Meena and Steinhardt, Jacob},
    title = {Feedback Loops Drive In-Context Reward Hacking in LLMs},
    journal= {arXiv},
    year = {2024}
}
```

ToolEmu 框架：

```bibtex
@article{ruan2023toolemu,
  title={Identifying the Risks of LM Agents with an LM-Emulated Sandbox},
  author={Ruan, Yangjun and Dong, Honghua and Wang, Andrew and Pitis, Silviu and Zhou, Yongchao and Ba, Jimmy and Dubois, Yann and Maddison, Chris J. and Hashimoto, Tatsunori},
  journal={arXiv preprint arXiv:2309.15817},
  year={2023}
}
```

## 许可证

本项目采用 MIT 许可证。

## 致谢

- [ToolEmu](https://github.com/ryoungj/ToolEmu) 提供策略优化框架
- [PromptCoder](https://github.com/dhh1995/PromptCoder) 提供模块化提示词管理
- Google Perspective API 提供毒性评分服务
- Pan 等人 (2024) 识别 ICRH 现象的开创性工作
