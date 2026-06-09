import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_single_file(script_path, json_file_path, args):
    """
    运行单个 JSON 文件的评估任务
    """
    command = ["python", script_path, "--data_json", json_file_path[:-5]]  # 去掉 .json 后缀
    command.extend(args)

    try:
        subprocess.run(command, check=True)
        print(f"成功处理: {json_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"处理文件 {json_file_path} 时发生错误: {e}")

def evaluate_all_json_files(folder_path, script_path, args, max_workers=5):
    """
    并发地评估文件夹下所有 JSON 文件
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"指定的文件夹路径不存在：{folder_path}")

    json_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.json')
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_file, script_path, json_path, args)
            for json_path in json_files
        ]

        for future in as_completed(futures):
            # 如果你需要捕获异常，可以在这里获取 future.result()
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="包含 JSON 文件的文件夹路径")
    parser.add_argument("--script_path", type=str, required=True, help="评估脚本的路径")
    parser.add_argument("--max_workers", type=int, default=1, help="最大并发任务数")
    args, remaining_args = parser.parse_known_args()  # add --no_async if no support async

    evaluate_all_json_files(
        folder_path=args.folder_path,
        script_path=args.script_path,
        args=remaining_args,
        max_workers=args.max_workers
    )
