import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse

def check_if_exists(seed, folder):
    filename = f"seed_{seed}-"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in os.listdir(folder):
        if file.startswith(filename) and file.endswith(".json"):
            return True
    return False

def run_experiment(seed, method, agent_model, judge_model, n_judges, agent_idx, cons_model, ab_test_flag):
    """运行单个实验"""
    # 确定结果保存路径
    if agent_idx == -1:
        folder = f"rh_multi_agent_results/{agent_model}/{method}/"
    else:
        folder = f"rh_single_agent_results/{agent_model}/{method}/"
    
    # 如果文件已存在则跳过（可选）
    # if check_if_exists(seed, folder):
    #     print(f"Seed {seed} already exists, skipping.")
    #     return
    
    # 构建命令
    cmd_parts = [
        "python filtering.py",
        "--experiment reward_hacking",
        "--n_rounds 10",
        f"--agent_model {agent_model}",
        f"--judge_model {judge_model}",
        f"--n_judges {n_judges}",
        f"--seed {seed}",
        f"--agent_idx {agent_idx}",
        f"--agent_name agent{agent_idx}" if agent_idx != -1 else "--agent_name all",
        f"--method {method}"
    ]
    
    if cons_model:
        cmd_parts.append(f"--cons_model {cons_model}")
    
    if ab_test_flag:
        cmd_parts.append("--ab_test")
    
    command = " ".join(cmd_parts)
    print(f"\n[Running] {command}")
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行过滤实验")
    parser.add_argument("--agent_model", type=str, required=True, help="Agent模型名称")
    parser.add_argument("--method", type=str, required=True, help="方法 (base, LCO, self_reflection, self_defense, goal_priority)")
    parser.add_argument("--judge_model", type=str, required=True, help="Judge模型名称")
    parser.add_argument("--n_judges", type=int, default=1, help="Judge数量")
    parser.add_argument("--agent_idx", type=int, default=-1, help="Agent索引 (-1表示多agent)")
    parser.add_argument("--cons_model", type=str, default=None, help="约束生成模型")
    parser.add_argument("--ab_test", action="store_true", help="是否进行AB测试")
    parser.add_argument("--max_workers", type=int, default=1, help="并行工作数")
    parser.add_argument("--seeds", type=int, default=20, help="运行种子数 (默认0-19)")
    
    args = parser.parse_args()
    
    print(f"=== 批量运行实验 ===")
    print(f"模型: {args.agent_model}")
    print(f"方法: {args.method}")
    print(f"Judge: {args.judge_model}")
    print(f"种子数: 0-{args.seeds-1}")
    print(f"并行数: {args.max_workers}")
    print("=" * 50)
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                run_experiment, 
                seed, 
                args.method,
                args.agent_model,
                args.judge_model,
                args.n_judges,
                args.agent_idx,
                args.cons_model,
                args.ab_test
            ) 
            for seed in range(0, args.seeds)
        ]
        for future in tqdm(futures, desc="Running experiments", unit="seed"):
            future.result()  # 等待任务完成
    
    print("\n=== 所有实验完成 ===")