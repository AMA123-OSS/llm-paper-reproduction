"""
Analyze adaptive emulation results.

Usage: python -m scripts.analyze_adaptive -inp dumps/traj_adaptive_xxx.jsonl
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-inp", required=True, help="Path to adaptive trajectory JSONL")
    args = parser.parse_args()

    stats = defaultdict(
        lambda: {"count": 0, "sim_types": defaultdict(int), "pop_sizes": defaultdict(int)}
    )
    total = 0
    errors = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1

            if "error" in obj or "error" in obj.get("output", {}):
                errors += 1

            rl = obj.get("risk_level", "unknown")
            st = obj.get("adaptive_simulator_type", "unknown")
            ps = obj.get("adaptive_population_size", "unknown")

            stats[rl]["count"] += 1
            stats[rl]["sim_types"][st] += 1
            stats[rl]["pop_sizes"][ps] += 1

    print("=" * 60)
    print(f"Total trajectories: {total}")
    print(f"Failed trajectories: {errors}")
    print("=" * 60)

    for rl in sorted(stats.keys()):
        info = stats[rl]
        pct = info["count"] / total * 100 if total else 0
        print(f"\nRisk Level {rl}: {info['count']} cases ({pct:.1f}%)")
        print(f"  Simulator types : {dict(info['sim_types'])}")
        print(f"  Population sizes: {dict(info['pop_sizes'])}")

    print("\n" + "=" * 60)
    # Rough latency estimation based on ablation numbers in the paper
    # Vanilla ~2.56s, Pos=1 ~4.50s, Pos=3 ~9.30s
    latency_map = {"1": 2.56, "2": 4.50, "3": 9.30}
    if total:
        est_latency = sum(
            stats[rl]["count"] * latency_map.get(str(rl), 5.0) for rl in stats
        ) / total
        print(f"Estimated average latency (paper ablation): {est_latency:.2f}s")
        fixed_lco_latency = 9.30
        print(f"Fixed LCO (Pos=3) latency (paper ablation) : {fixed_lco_latency:.2f}s")
        print(f"Latency reduction vs Fixed LCO             : {(fixed_lco_latency - est_latency) / fixed_lco_latency * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
