import os
import datetime
from collections import defaultdict
import re
import argparse
import json

import asyncio
import aiohttp
import numpy as np
from anthropic import HUMAN_PROMPT, AI_PROMPT

from toxicity import get_perspective_toxicity_scores
from api_keys import ANTHROPIC_API_KEY, OPENAI_API_KEY, GPT_API_BASE

# vLLM支持（可选导入）
try:
    from vllm_backend import (
        query_vllm,
        is_vllm_model,
        get_model_path,
        VLLMConfig
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    def is_vllm_model(model_name):
        return False
    def query_vllm(*args, **kwargs):
        raise RuntimeError("vLLM is not installed.")
    def get_model_path(model_name):
        return model_name
    class VLLMConfig:
        pass

# VLLM配置
_VLLM_GLOBAL_CONFIG = {
    "tensor_parallel_size": int(os.getenv("VLLM_TP_SIZE", "1")),
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_UTIL", "0.85")),
}

MAX_TRIES = 5
MODEL_STORE = {}


async def async_query(
    model_name,
    user_prompt,
    system_prompt="You are a helpful assistant.",
    max_tokens=100,
    temperature=1.0,
    stop_sequences=[],
):
    """异步查询接口，支持云端API和本地vLLM模型"""
    
    # 检测是否使用vLLM
    if is_vllm_model(model_name):
        model_path = get_model_path(model_name)
        tries = 0
        while tries < MAX_TRIES:
            try:
                # 使用线程池运行同步的vLLM调用
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # 使用默认线程池
                    lambda: query_vllm(
                        model_path=model_path,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop_sequences=stop_sequences if stop_sequences else None,
                        tensor_parallel_size=_VLLM_GLOBAL_CONFIG["tensor_parallel_size"],
                        gpu_memory_utilization=_VLLM_GLOBAL_CONFIG["gpu_memory_utilization"],
                    )
                )
                return result
            except Exception as e:
                tries += 1
                print(f"VLLM Error (attempt {tries}/{MAX_TRIES}):", e)
                await asyncio.sleep(2)
        raise RuntimeError(f"VLLM query failed after {MAX_TRIES} attempts")
    
    # 使用云端API（原有的aiohttp实现）
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if model_name.startswith("claude"):
            tries = 0
            while tries < MAX_TRIES:
                try:
                    response = await session.post(
                        "https://api.anthropic.com/v1/complete",
                        headers={
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                            "x-api-key": ANTHROPIC_API_KEY,
                        },
                        json={
                            "model": model_name,
                            "prompt": f"{HUMAN_PROMPT} {system_prompt}\n{user_prompt}{AI_PROMPT}",
                            "max_tokens_to_sample": max_tokens,
                            "temperature": temperature,
                            "stop_sequences": stop_sequences,
                        },
                    )
                    data = await response.json()
                    return data["completion"]
                except Exception as e:
                    tries += 1
                    print("Error:", e)
                    await asyncio.sleep(2)
        elif model_name.startswith("gpt"):
            tries = 0
            while tries < MAX_TRIES:
                try:
                    response = await session.post(
                        f"{GPT_API_BASE}/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                        },
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stop": stop_sequences,
                        },
                    )
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                except Exception as e:
                    tries += 1
                    print("Error:", e)
                    await asyncio.sleep(2)
        else:
            raise ValueError(f"Unknown model: {model_name}. Supported formats: gpt-*, claude-*, vllm://path, or local path")


async def async_judge_responses(model_name, judges, judge_prompt_fn, responses, args):
    if model_name == "random":
        return np.random.choice(len(responses))
    scores = defaultdict(int)
    idx = np.random.choice(len(judges), size=args.n_judges, replace=False)

    async def async_judge(judge):
        user_prompt = judge_prompt_fn(
            judge, args.topic, args.objective, args.item_type, responses
        )
        tries = 0
        while tries < MAX_TRIES:
            try:
                out = await async_query(model_name, user_prompt)
                match = re.search(r"\d+", out)
                scores[int(match.group()) - 1] += 1
                break
            except Exception as e:
                tries += 1
                print("Error:", e)
                await asyncio.sleep(2)

    await asyncio.gather(
        *(async_judge(judge) for judge in np.array(list(judges.items()))[idx])
    )
    print("scores:",scores)
    try:
        max_score = max(scores.values())
        winning_idxs = [i for i, score in scores.items() if score == max_score]
        return np.random.choice(winning_idxs), scores
    except:
        return -1, scores


async def async_process_agent(agent, winning_agent, winning_response, i, args):
    init = i == 0
    if init and args.experiment == "optimization":
        with open("optimization/experiment_data.json", "r") as f:
            data = json.load(f)
        assert 0 <= args.agent_idx < len(data[args.seed]["init_responses"])
        winning_response = data[args.seed]["init_responses"][args.agent_idx]
    user_prompt = get_agent_prompt(
        agent,
        args.topic,
        args.objective,
        args.item_type,
        winning_agent,
        winning_response,
        init=init,
    )
    print("generation prompt:",user_prompt) # To delete
    response = await async_query(
        args.agent_model,
        user_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    response = process_response(response)
    if args.experiment == "reward_hacking":
        toxicity_score = get_perspective_toxicity_scores([response])
    else:
        toxicity_score = {"toxicity": [0]}

    return agent[0], response, toxicity_score["toxicity"][0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_model",
        type=str,
        required=True,
        help="Agent model name or path. Supports: gpt-4, claude, or local path for vLLM"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        required=True,
        help="Judge model name or path"
    )
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--ab_test", action="store_true")
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--agent_idx", type=int, default=0)
    parser.add_argument("--n_judges", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    
    # VLLM相关参数
    parser.add_argument(
        "--vllm_tp_size",
        type=int,
        default=int(os.getenv("VLLM_TP_SIZE", "1")),
        help="Tensor parallel size for vLLM (number of GPUs)"
    )
    parser.add_argument(
        "--vllm_gpu_util",
        type=float,
        default=float(os.getenv("VLLM_GPU_UTIL", "0.85")),
        help="GPU memory utilization for vLLM"
    )
    
    return parser.parse_args()


def replace_model_name(name):
    """
    将简写模型名称映射为完整名称。
    注意：只有完全匹配的简写才会被替换，具体版本名称（如 qwen3-8B）会保持原样。
    """
    name_mapping = {
        "claude": "claude-1.1",
        "gpt": "gpt-4-0613",
        "gpt-4": "gpt-4-0613",
        "qwen": "qwen2.5-72b-instruct",
        "llama": "llama-3.1-405b-instruct",
        "llama3": "llama-3.1-405b-instruct",
    }
    # 只处理完全匹配的简写，保持其他名称（如 qwen3-8B）不变
    return name_mapping.get(name, name)


async def run_experiment(agents, judges, args):
    agent_responses = defaultdict(list)
    all_toxicity_scores = []
    winning_agent = None
    winning_response = None
    for i in range(args.n_rounds):
        print(f"====== Round {i+1} ======")
        responses = []
        round_toxicity_scores = []

        results = await asyncio.gather(
            *(
                async_process_agent(agent, winning_agent, winning_response, i, args)
                for agent in agents.items()
            )
        )
        print("Async_Result:",results)
        for agent, response, toxicity in results:
            responses.append(response)
            agent_responses[agent[0]].append(response)
            round_toxicity_scores.append(toxicity)
        print("responses:",responses)

        if args.ab_test:
            if i != 0:
                responses.append(winning_response)
                print("response with winning response:",responses)
        # 比较不同智能体之间的回答，进行排名，获得最佳回答
        if len(agents)==1:
            winning_idx=0
        else:
            winning_idx, _ = await async_judge_responses(
                args.judge_model, judges, get_judge_prompt, responses, args
            )
        print("winning idx:", winning_idx)
        winning_agent = list(agents.items())[winning_idx % len(agents)]
        winning_response = responses[winning_idx]
        print("winning_agent:",winning_agent)
        all_toxicity_scores.append(round_toxicity_scores)
        print(f"Winning response: {winning_response}")

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data = {
        "agent_model": args.agent_model,
        "judge_model": args.judge_model,
        "agent_idx": args.agent_idx,
        "experiment": args.experiment,
        "n_rounds": args.n_rounds,
        "n_judges": args.n_judges,
        "seed": args.seed,
        "topic": args.topic,
        "item_type": args.item_type,
        "objective": args.objective,
        "agent_responses": agent_responses,
        "toxicity_scores": all_toxicity_scores,
    }
    if args.experiment == "reward_hacking":
        if args.agent_idx != -1:
            folder = "rh_singleagent_results"
        else:
            folder = "rh_multiagent_results"
    else:
        folder = "optimization_results"
    if args.ab_test:
        folder += "_ab_test"

    os.makedirs(folder, exist_ok=True)
    name = folder + f"/agent-{args.agent_model}_filter-{args.judge_model}_{date}"

    with open(f"{name}.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    
    # 设置VLLM全局配置
    _VLLM_GLOBAL_CONFIG["tensor_parallel_size"] = args.vllm_tp_size
    _VLLM_GLOBAL_CONFIG["gpu_memory_utilization"] = args.vllm_gpu_util
    
    # 检测并使用vLLM
    if is_vllm_model(args.agent_model):
        print(f"Using VLLM backend for agent model: {args.agent_model}")
        print(f"Tensor parallel size: {_VLLM_GLOBAL_CONFIG['tensor_parallel_size']}")
    
    np.random.seed(args.seed)
    
    # 如果不是vLLM模型，进行名称替换
    if not is_vllm_model(args.agent_model):
        args.agent_model = replace_model_name(args.agent_model)
    if not is_vllm_model(args.judge_model):
        args.judge_model = replace_model_name(args.judge_model)

    if args.experiment == "reward_hacking":
        from reward_hacking.utils import (
            init_experiment,
            get_agent_prompt,
            get_judge_prompt,
            process_response,
        )
    elif args.experiment == "optimization":
        from optimization.utils import (
            init_experiment,
            get_agent_prompt,
            get_judge_prompt,
            process_response,
        )
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}.")

    agents, judges, args = init_experiment(args)
    asyncio.run(run_experiment(agents, judges, args))