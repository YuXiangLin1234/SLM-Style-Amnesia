import argparse
import os
import re
from collections import defaultdict

from eval_utils import extract_turn_index, read_metric_file


METRICS = ("tts_ratio_word",)


def check_ratio_logic(value, style_name):
    s = style_name.lower()
    if "fast" in s:
        return value > 1
    if "slow" in s:
        return value < 1
    return None


def run(audio_dirs):
    for dir_path in audio_dirs:
        # global_stats[metric][turn_id] -> [passed, total]
        global_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        style = os.path.basename(dir_path)
        style_lower = style.lower()

        if "fast" not in style_lower and "slow" not in style_lower:
            print(f"\nSkipping directory (no fast/slow rule): {dir_path}")
            continue

        print(f"\n--- Processing Directory: {dir_path} (Style: {style}) ---")

        try:
            rows = read_metric_file(dir_path, type="heuristic")
        except FileNotFoundError:
            print(f"Warning: Metric file not found for {dir_path}, skipping.")
            continue

        rows = [
            r for r in rows
        ]
        if not rows:
            print("No valid rows found after filtering.")
            continue

        for r in rows:
            basename = os.path.basename(r["file"])
            turn_id = extract_turn_index(str(basename))

            for metric in METRICS:
                if metric not in r:
                    continue
                value = r[metric]
                if value is None:
                    print(f"Skipping {basename}: metric {metric} is None")
                    continue

                res = check_ratio_logic(value, style)
                if res is None:
                    continue

                stats = global_stats[metric][turn_id]
                stats[1] += 1
                if res:
                    stats[0] += 1

        for metric, turn_data in global_stats.items():
            if not turn_data:
                print("  No data found for this metric.")
                continue

            sorted_turn_ids = sorted(
                turn_data.keys(),
                key=lambda x: int(re.sub(r"\D", "", str(x or "0"))),
            )
            for turn_id in sorted_turn_ids:
                p, tot = turn_data[turn_id]
                acc = (p / tot * 100) if tot > 0 else float("nan")
                print(f"  {turn_id}: acc={acc:2.1f} ({p}/{tot})", end="  |  ")
            print()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dirs",
        nargs="+",
        help="Result directories whose basename contains 'fast' or 'slow' (e.g. results/<model>/<location>/en_fast).",
    )
    args = parser.parse_args()
    run(args.audio_dirs)
