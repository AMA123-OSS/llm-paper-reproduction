# Policy Refinement Experiments

This directory contains the **policy-refinement** component of the LCO framework for evaluating defense mechanisms against In-Context Reward Hacking (ICRH) in tool-use scenarios using the ToolEmu benchmark.

## Directory Structure

```
policy-refinement/
│
├── toolemu/                       # ToolEmu framework (modified)
│   ├── agents/                    # Agent implementations
│   │   ├── MyAgent.py             # LCO/Genetic agent core
│   │   ├── self_reflection_agent.py   # Self-Thought + Risk Classification
│   │   ├── self_defense_agent.py      # Self-Defense baseline
│   │   ├── goal_priority_agent.py     # Goal-Priority baseline
│   │   ├── safety_prompt_agent.py     # Safety-Prompt baseline
│   │   ├── virtual_agent_executor.py
│   │   └── zero_shot_agent_with_toolkit.py
│   ├── prompts/                   # Prompt templates
│   ├── tools/                     # Tool interface definitions
│   └── utils/                     # Utility functions
│
├── scripts/                       # Experiment scripts
│   ├── emulate.py                 # Fixed LCO emulation
│   ├── emulate_adaptive.py        # Risk-Adaptive emulation
│   ├── analyze_adaptive.py        # Analyze adaptive results
│   ├── evaluate.py                # Evaluate trajectories
│   └── run.py                     # Full pipeline
│
├── evaluation_scripts/            # ICRH and helpfulness detection
│   ├── ICRH_Detect.py             # Detect ICRH in trajectories
│   ├── traj_helpfulness.py        # Evaluate task helpfulness
│   └── pairwise_eval.py           # Pairwise safety comparison
│
├── tests/                         # Test scripts
│   ├── test_env_setup.py          # Environment verification
│   └── test_risk_levels.py        # Batch risk classification test
│
├── experiment_data/               # All experimental results
│   ├── tool_emu/
│   │   ├── trajectories/          # Agent trajectories by model
│   │   ├── icrh/                  # ICRH detection results
│   │   ├── helpfulness/           # Helpfulness evaluations
│   │   └── pairwise/              # Pairwise comparison results
│   ├── benchmark/                 # GSM8K/MMLU datasets
│   ├── fitness_model/             # Fitness evaluation data
│   ├── risk_levels/               # Risk classification results
│   └── json_dumps/                # JSON format conversions
│
├── plots/                         # Paper figures
├── assets/                        # Test cases (all_cases.json) and toolkits
├── .env.example                   # Environment variable template
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
cd policy-refinement
pip install -e .
```

This installs `toolemu` and `procoder` in editable mode.

### 2. Configure API Keys

Create `.env` file in `policy-refinement/`:

```bash
# Required
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key

# Model-specific (optional, fallback to OPENAI_*)
GPT4_API_BASE=https://api.openai.com/v1
GPT4_API_KEY=sk-your-key
GPT35_API_BASE=https://api.openai.com/v1
GPT35_API_KEY=sk-your-key

QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-your-key

LLAMA_API_BASE=https://api.together.xyz/v1
LLAMA_API_KEY=sk-your-key

# LCO Judges (recommended: GPT-4o)
VOTE_API_BASE=https://api.openai.com/v1
VOTE_API_KEY=sk-your-key

FITNESS_API_BASE=https://api.openai.com/v1
FITNESS_API_KEY=sk-your-key

# Evaluators (ICRH/helpfulness detection)
EVAL_API_BASE=https://api.openai.com/v1
EVAL_API_KEY=sk-your-key

# Self-Defense judge
SELF_DEFENSE_API_BASE=https://api.openai.com/v1
SELF_DEFENSE_API_KEY=sk-your-key
```

See `.env.example` for the full template.

### 3. Verify Environment

```bash
python tests/test_env_setup.py
```

## Usage Examples

### Running a Full Comparison (Vanilla vs Fixed LCO vs Adaptive)

```bash
cd policy-refinement

# 1. Vanilla baseline
python -m scripts.emulate \
    -inp assets/all_cases.json \
    -atp naive -stp adv_thought \
    --agent-model qwen2.5-72b-instruct \
    --simulator-model gpt-4o \
    -si 0 -tn 70 -v -me 3

# 2. Fixed LCO (pos=3)
python -m scripts.emulate \
    -inp assets/all_cases.json \
    -atp naive -stp lco_thought \
    --agent-model qwen2.5-72b-instruct \
    --simulator-model gpt-4o \
    -si 0 -tn 70 -v -me 3 -pos 3

# 3. Risk-Adaptive
python -m scripts.emulate_adaptive \
    -inp assets/all_cases.json \
    -atp naive \
    --agent-model qwen2.5-72b-instruct \
    --simulator-model gpt-4o \
    -si 0 -tn 70 -v -me 3
```

### Evaluating Results

```bash
# ICRH detection for all methods
for method in vanilla lco_pos3 adaptive; do
    python evaluation_scripts/ICRH_Detect.py \
        --file experiment_data/tool_emu/trajectories/qwen2.5-72b/traj_${method}.jsonl \
        --method $method \
        --output_path experiment_data/tool_emu/icrh/icrh_${method}.jsonl
done

# Helpfulness evaluation
for method in vanilla lco_pos3 adaptive; do
    python evaluation_scripts/traj_helpfulness.py \
        --file experiment_data/tool_emu/trajectories/qwen2.5-72b/traj_${method}.jsonl \
        --method $method \
        --output_path experiment_data/tool_emu/helpfulness/help_${method}.jsonl
done
```

### Analyzing Risk Distribution

```bash
python -m scripts.analyze_adaptive \
    --file experiment_data/tool_emu/trajectories/qwen2.5-72b/traj_adaptive.jsonl
```

### Testing Risk Classification on External Datasets

```bash
# ToolEmu
python tests/test_risk_levels.py --toolemu

# GSM8K
python tests/test_risk_levels.py \
    --gsm8k experiment_data/benchmark/datasets/gsm8k_test.jsonl \
    --gsm8k-samples 100

# MMLU
python tests/test_risk_levels.py \
    --mmlu experiment_data/benchmark/datasets/mmlu_test.jsonl \
    --mmlu-samples 100
```

## Important Runtime Rule

**All scripts must be run with `python -m` from the `policy-refinement/` directory:**

```bash
# ✅ Correct
cd policy-refinement
python -m scripts.emulate -inp assets/all_cases.json ...

# ❌ Incorrect
cd policy-refinement/scripts
python emulate.py ...
```

This is because agent files use absolute imports (`from toolemu.agents import ...`) that require the package to be importable from the project root.

## Key Parameters

| Parameter | Flag | Description | Typical Values |
|-----------|------|-------------|----------------|
| Agent type | `-atp` | Agent prompt style | `naive`, `ss_only` |
| Simulator type | `-stp` | Simulator behavior | `adv_thought`, `lco_thought` |
| Population size | `-pos` | LCO candidate count | 1, 3, 5 |
| Max errors | `-me` | Error injection count | 0, 3 |
| Start index | `-si` | First case to run | 0 |
| Truncation | `-tn` | Number of cases | 70, 144 |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wan2026lco,
    title = {LCO: LLM-based Constraint Optimization for Safer Agentic LLMs in Real-world Tasks},
    author = {Wan, Jiayong and Chen, Jiawei and Yin, Zhaoxia and Liu, Shuyuan and Su, Hang},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
    year = {2026}
}
```

## Acknowledgments

- [ToolEmu](https://github.com/ryoungj/ToolEmu) framework by Ruan et al. (2023)
- [PromptCoder](https://github.com/dhh1995/PromptCoder) for modular prompt management
