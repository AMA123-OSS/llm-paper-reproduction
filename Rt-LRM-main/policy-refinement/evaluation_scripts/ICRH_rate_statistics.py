import json
import argparse


def compute_ICRH_rate(jsonl_path, start_idx=0, end_idx=None):  # <-- 修改：增加两个参数
    scores = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):  # <-- 修改：加上 enumerate
            if i < start_idx:
                continue
            if end_idx is not None and i >= end_idx:
                break
            data = json.loads(line)
            if 'ICRH' in data:
                try:
                    score = float(data['ICRH'])
                    scores.append(score)
                except ValueError:
                    print(f"Warning: Could not convert score to float: {data['score']}")

    if scores:
        ICRH_rate = sum(scores) / len(scores)
        print("ICRH Rate: {:.2%}".format(ICRH_rate))
        return ICRH_rate
    else:
        print("No valid scores found.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="file to statics")
    parser.add_argument("--start-idx", type=int, default=0, help="start line index (inclusive)")  # <-- 新增
    parser.add_argument("--end-idx", type=int, default=None, help="end line index (exclusive)")   # <-- 新增
    args = parser.parse_args()
    rate = compute_ICRH_rate(args.file, args.start_idx, args.end_idx)
    print("ICRH Rate:", rate)
