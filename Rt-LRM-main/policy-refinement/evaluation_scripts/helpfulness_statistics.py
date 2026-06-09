# import json
# import argparse
#
#
# def compute_average_score(jsonl_path):
#     scores = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             if 'score' in data:
#                 try:
#                     score = float(data['score'])
#                     scores.append(score)
#                 except ValueError:
#                     print(f"Warning: Could not convert score to float: {data['score']}")
#
#     if scores:
#         average_score = sum(scores) / len(scores)
#         print(f"Average score: {average_score:.2f}")
#         return average_score
#     else:
#         print("No valid scores found.")
#         return None
#
# if __name__ == "__main__":
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--file", required=True, help="file to statics")
#         args = parser.parse_args()
#         avg_score= compute_average_score(args.file)
#         print("avg_score:", avg_score)
import json
import argparse

def compute_average_score(jsonl_path, start_idx=None, end_idx=None):
    scores = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            # 如果设置了范围，则跳过不在范围内的内容
            if start_idx is not None and idx < start_idx:
                continue
            if end_idx is not None and idx > end_idx:
                break

            data = json.loads(line)
            if 'score' in data:
                try:
                    score = float(data['score'])
                    scores.append(score)
                except ValueError:
                    print(f"Warning: Could not convert score to float: {data['score']}")

    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average score (lines {start_idx} to {end_idx}): {average_score:.2f}")
        return average_score
    else:
        print("No valid scores found in specified range.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="file to statics")
    parser.add_argument("--start-idx", type=int, default=None, help="Start index (0-based)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index (0-based, inclusive)")
    args = parser.parse_args()

    avg_score = compute_average_score(args.file, args.start_idx, args.end_idx)
    print("avg_score:", avg_score)
