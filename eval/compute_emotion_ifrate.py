import argparse
import os
from collections import defaultdict

from eval_utils import extract_turn_index, read_metric_file


# Style key -> emotion2vec label that should win the argmax.
MAPPING = {
    "en_anger": "Anger",
    "en_neutral": "Neutral",
    "en_happiness": "Happiness",
    "en_sadness": "Sadness",
}

# When True, restrict the argmax to the four target emotions defined above.
FOUR_WAY = True


def check_following(vals):
    """Return True iff the predicted emotion matches MAPPING for the style key."""
    origin_key, probs_dict = next(iter(vals.items()))
    expected_category = MAPPING.get(origin_key)

    if FOUR_WAY:
        target_categories = set(MAPPING.values())
        filtered_probs = {
            category: prob
            for category, prob in probs_dict.items()
            if category in target_categories
        }
        predicted_category = max(filtered_probs, key=filtered_probs.get)
    else:
        predicted_category = max(probs_dict, key=probs_dict.get)

    return expected_category == predicted_category


def run(audio_dirs):
    for dir in audio_dirs:
        print(f"[Dirs]: {dir}")

        # data[basename][style] -> metric row
        data = defaultdict(dict)
        rows = read_metric_file(dir, type="emotion")
        rows = [
            r for r in rows
        ]

        for r in rows:
            basename = os.path.basename(r["file"])
            style = os.path.basename(dir)
            data[basename][style] = r

        # Bucket files by their turn index.
        turn_to_files = defaultdict(list)
        for basename in data:
            turn_to_files[extract_turn_index(str(basename))].append(basename)

        turn_stats = {}
        for turn_id, file_list in turn_to_files.items():
            passed = 0
            total = 0
            for basename in file_list:
                vals = {}
                for style, row in data[basename].items():
                    try:
                        vals[style] = row["emotion"]
                    except KeyError:
                        continue
                if not vals:
                    continue
                total += 1
                if check_following(vals):
                    passed += 1
            turn_stats[turn_id] = (passed, total)

        for turn_id in sorted(turn_stats):
            p, tot = turn_stats[turn_id]
            acc = (p / tot * 100) if tot > 0 else float("nan")
            print(f"{turn_id}: acc={acc:2.1f} ({p}/{tot})", end="  |  ")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dirs",
        nargs="+",
        help="Result directories to evaluate (e.g. results/<model>/<location>/<style>).",
    )
    args = parser.parse_args()
    run(args.audio_dirs)
