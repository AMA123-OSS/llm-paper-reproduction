import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from collections import defaultdict
import re
import argparse
import json
import random
import asyncio
import aiohttp
import numpy as np
from anthropic import HUMAN_PROMPT, AI_PROMPT
from toxicity import get_perspective_toxicity_scores
from api_keys import ANTHROPIC_API_KEY, OPENAI_API_KEY, GPT_API_BASE
from reward_hacking.utils import get_objective_judge_prompt,get_crossover_prompt
from typing import Dict, List,Any

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
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if model_name.startswith("claude"):
            tries = 0
            while tries < MAX_TRIES:
                try:
                    # Assuming a POST request here. Modify accordingly.
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
                    # Again, assuming a POST request. Modify accordingly.
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
            raise ValueError(f"Unknown model: {model_name}.")


async def async_judge_responses(model_name, judges, judge_prompt_fn, responses, args):
    if model_name == "random":
        return np.random.choice(len(responses))

    scores = defaultdict(int)
    idx = np.random.choice(len(judges), size=args.n_judges, replace=False)
    print("judges",judges)

    async def async_judge(judge):
        user_prompt = judge_prompt_fn(
            judge, args.topic, args.objective, args.item_type, responses
        )
        tries = 0
        while tries < MAX_TRIES:
            try:
                out = await async_query(model_name, user_prompt)
                print("out",out)
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
    try:
        print("scores:",scores)
        max_score = max(scores.values())
        winning_idxs = [i for i, score in scores.items() if score == max_score]
        return np.random.choice(winning_idxs), scores
    except:
        return -1, scores


async def async_process_agent(agent, winning_agent, winning_response, i, args,n):
    init = i == 0
    if init and args.experiment == "optimization":
        with open("optimization/experiment_data.json", "r") as f:
            data = json.load(f)
        assert 0 <= args.agent_idx < len(data[args.seed]["init_responses"])
        winning_response = data[args.seed]["init_responses"][args.agent_idx]
    if winning_response!=None:
        print(winning_response[n])
        user_prompt = get_agent_prompt(
            agent,
            args.topic,
            args.objective,
            args.item_type,
            winning_agent,
            winning_response[n],
            init=init,
        )
    else:
        user_prompt = get_agent_prompt(
            agent,
            args.topic,
            args.objective,
            args.item_type,
            winning_agent,
            winning_response,
            init=init,
        )
    response = await async_query(
        args.agent_model,
        user_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    response = process_response(response)
    # objective_judge_prompt = get_objective_judge_prompt(
    #     topic=args.topic,
    #     tweet=response
    # )
    # print("objective_judge_prompt:",objective_judge_prompt)
    # objective_score=await async_query(
    #     args.objective_judge_model,
    #     objective_judge_prompt,
    #     temperature=1,
    # )
    # print("objective_score:",objective_score)
    if args.experiment == "reward_hacking":
        toxicity_score = get_perspective_toxicity_scores([response])
        # fitness_score = float(objective_score)-10*toxicity_score["toxicity"][0]
        fitness_score=-toxicity_score["toxicity"][0]
    else:
        toxicity_score = {"toxicity": [0]}
        # fitness_score = float(objective_score)
        fitness_score=1
    return agent[0], response, toxicity_score["toxicity"][0],fitness_score,n


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_model",
        choices=["claude", "gpt", "gpt-3.5-turbo"],
        default="gpt",
    )
    parser.add_argument(
        "--judge_model",
        choices=["claude", "gpt", "gpt-3.5-turbo", "random"],
        default="random",
    )
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--experiment", type=str,default="reward_hacking")
    parser.add_argument("--ab_test", default=False)
    parser.add_argument("--n_rounds", type=int, default=2)
    parser.add_argument("--agent_idx", type=int, default=0)
    parser.add_argument("--n_judges", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N",type=int,default=5)
    parser.add_argument("--K",type=int,default=3)
    parser.add_argument("--objective_judge_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--genetic_model", type=str, default="gpt-3.5-turbo")
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


async def crossover(K: int, sorted_fitness_scores, responses):
    # 每次从适应度排名前K的个体中随机选取两个进行交叉产生子代
    top_k_indices = list(sorted_fitness_scores.keys())[:K]
    children_responses = []
    parent1_idx, parent2_idx = random.sample(top_k_indices, 2)
    print(parent1_idx,parent2_idx)
    parent1_response = responses[parent1_idx]
    parent2_response = responses[parent2_idx]
    crossover_prompt = get_crossover_prompt(args.topic,parent1_response, parent2_response)
    print(crossover_prompt)
    child =await async_query(
        args.genetic_model,
        crossover_prompt,
        temperature=0.8
    )
    print("Parent:\n",parent1_response,"\n",parent2_response)
    print("child:",child)
    children_responses.append(child)
    return children_responses


async def run_experiment(agents, judges, args):
    agent_responses = defaultdict(list)
    children_responses:Dict[str, Dict[int, List[Any]]]={key:{} for key in agents}
    all_toxicity_scores = []
    winning_agent = None
    winning_response = None
    file_path=f"res_from_{args.agent_model}.json"
    for i in range(args.n_rounds):
        print(f"====== Round {i+1} ======")
        round_toxicity_scores = []
        res={key:[] for key in agents}
        # 并行各个agent的生成，每个agent有N个优化路线（种群数量为N）
        results=await asyncio.gather(
            *(
                async_process_agent(agent, winning_agent, winning_response, i, args, n) for agent in agents.items() for n in range(args.N)
            )
        )
        fitness_scores={key:{} for key in agents}
        print("Async_Result:", results)
        for result in results:
            res[result[0]].append(result)
        # 存入每次循环的回复
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump([res], file, ensure_ascii=False, indent=4)
        else:
            with open(file_path, "r+", encoding="utf-8") as file:
                # 加载现有内容，移除末尾的 "]" 再追加
                file.seek(0, os.SEEK_END)
                file.seek(file.tell() - 1, os.SEEK_SET)
                file.write(",\n")  # 加入逗号换行符
                json.dump(res, file, ensure_ascii=False, indent=4)
                file.write("]")  # 补上结尾的 "]"

        # 获取不同agent的每个个体的适应度评分，选取top_k的个体产生子代
        for agent in res:
            responses=[]
            for _, response, toxicity, fitness_score, n in res[agent]:
                responses.append(response)
                # 把并行路线的结果分开存储
                agent_responses[f"{agent[0]}_{n}"].append(response)
                round_toxicity_scores.append(toxicity)
                fitness_scores[agent][n] = fitness_score
            print("responses:", responses)
            print("agent_responses:", agent_responses)
            print("toxicity scores:", round_toxicity_scores)
            print("fitness_scores:", fitness_scores)
            all_toxicity_scores.append(round_toxicity_scores)
            # 选取top_k的个体交叉与变异
            sorted_fitness_scores = dict(sorted(fitness_scores[agent].items(), key=lambda x: x[1], reverse=True))
            print("sorted_fitness_scores:", sorted_fitness_scores)
            children = await asyncio.gather(
                *(crossover(args.K,sorted_fitness_scores,responses) for c in range(args.N))
            )
            # 记录第i次迭代生成的子代
            children_responses[agent].setdefault(i,[])
            children_responses[agent][i]=list(children)

        # single agent 情况下每次循环和上一次的子代进行对比即可
        if len(agents)==1:
            winning_agent = next(iter(agents.items()))
            winning_response = children_responses[list(agents.keys())[0]][i]
            print(winning_response)

        # else:
        #     if args.ab_test:
        #         if i != 0:
        #             responses.append(winning_response)
        #     winning_idx, _ = await async_judge_responses(
        #         args.judge_model, judges, get_judge_prompt, responses, args
        #     )
        #     print("winning idx:", winning_idx)
        #     winning_agent = list(agents.items())[winning_idx % len(agents)]
        #     winning_response = responses[winning_idx]
        #     print("winning_agent:", winning_agent)
        #     all_toxicity_scores.append(round_toxicity_scores)
        #     print(f"Winning response: {winning_response}")



    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 将子代样本记录下来以便观察
    with open(f"child_{date}_{args.agent_model}_agent_dix{args.agent_idx}_seed{args.seed}.json", "w", encoding="utf-8") as file:
        json.dump(children_responses, file, ensure_ascii=False, indent=4)
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
            folder = "rh_singleagent_parallel_results"
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
    np.random.seed(args.seed)
    args.agent_model = replace_model_name(args.agent_model)
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