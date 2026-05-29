"""First-turn IF rate (IF_1) and Degradation rate (D) from the paper.

Paper §3.3, formulas (1) and (2):

    IF_j(s) = (1 / N) * sum_i 1(o_{i,j} matches s) * 100%
    D(s)    = sum_{j=2..K} max(IF_1(s) - IF_j(s), 0) / (K - 1)

where N is the number of topics, K is the dialogue horizon, and 1(.) is the
style-specific pass indicator. The pass indicator depends on the style
dimension:

  emotion : argmax of the 4-way filtered emotion2vec-Large probabilities
            matches MAPPING[style_key]
  accent  : argmax of the 3-way filtered Voxlect probabilities matches
            MAPPING[style_key]
  speed   : tts_ratio_word > 1 for *_fast, < 1 for *_slow
  volume  : tts_ratio_loudness < 1 for *_loud, > 1 for *_quiet
            (LUFS is negative dB, so the inequality flips)

Each result directory passed in is auto-classified by its basename. The script
prints IF_1 and D per directory and writes a `if1_degradation.json` alongside
the metric file with the full breakdown.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from eval_utils import extract_turn_index, read_metric_file
from instruction_collection import ACCENT_KEY, EMOTION_KEY, SPEED_KEY, VOLUME_KEY


PAPER_K = 4  # Dialogue horizon assumed by the paper.

EMOTION_MAPPING = {
    "en_anger": "Anger",
    "en_happiness": "Happiness",
    "en_neutral": "Neutral",
    "en_sadness": "Sadness",
}

ACCENT_MAPPING = {
    "en_north_america": "North America",
    "en_indian": "South Asia",
}


def _argmax_match(probs_dict, target_categories, expected):
    """Return True if argmax over the filtered probs matches `expected`."""
    if not probs_dict or not expected:
        return None
    filtered = {c: p for c, p in probs_dict.items() if c in target_categories}
    if not filtered:
        return None
    return max(filtered, key=filtered.get) == expected


def _check_emotion(row, style_key):
    return _argmax_match(
        row.get("emotion", {}),
        set(EMOTION_MAPPING.values()),
        EMOTION_MAPPING.get(style_key),
    )


def _check_accent(row, style_key):
    return _argmax_match(
        row.get("accent", {}),
        set(ACCENT_MAPPING.values()),
        ACCENT_MAPPING.get(style_key),
    )


def _check_speed(row, style_key):
    value = row.get("tts_ratio_word")
    if value is None:
        return None
    if "fast" in style_key:
        return value > 1
    if "slow" in style_key:
        return value < 1
    return None


def _check_volume(row, style_key):
    # tts_ratio_loudness uses negative-dB LUFS values: louder => smaller ratio.
    value = row.get("tts_ratio_loudness")
    if value is None:
        return None
    if "loud" in style_key or "shout" in style_key:
        return value < 1
    if "quiet" in style_key or "whisper" in style_key:
        return value > 1
    return None


DIMENSION_DISPATCH = {
    "emotion": {"metric_type": "emotion", "checker": _check_emotion, "keys": EMOTION_KEY},
    "accent":  {"metric_type": "accent",  "checker": _check_accent,  "keys": ACCENT_KEY},
    "speed":   {"metric_type": "heuristic", "checker": _check_speed,  "keys": SPEED_KEY},
    "volume":  {"metric_type": "heuristic", "checker": _check_volume, "keys": VOLUME_KEY},
}


def _classify_dir(audio_dir):
    style = os.path.basename(audio_dir)
    for dim, info in DIMENSION_DISPATCH.items():
        if style in info["keys"]:
            return dim, info
    return None, None


def _filter_rows(rows, paraphrasing_filter, max_turn):
    """Apply the standard paper filters: skip turns past max_turn, optionally pin to one paraphrasing."""
    par_pattern = (
        re.compile(rf"_par{paraphrasing_filter}_")
        if paraphrasing_filter is not None
        else None
    )
    out = []
    for r in rows:
        fname = r.get("file", "")
        try:
            turn = extract_turn_index(fname)
        except Exception:
            continue
        if turn < 1 or turn > max_turn:
            continue
        # Word-boundary match so --paraphrasing 1 does not catch par10/par11/...
        if par_pattern is not None and not par_pattern.search(fname):
            continue
        out.append(r)
    return out


def _compute_if_per_turn(rows, checker, style_key, max_turn):
    """Return {turn: (passed, total)} for turns 1..max_turn."""
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        turn = extract_turn_index(r["file"])
        result = checker(r, style_key)
        if result is None:
            continue
        counts[turn][1] += 1
        if result:
            counts[turn][0] += 1
    return {t: tuple(counts[t]) for t in range(1, max_turn + 1)}


def _if_rate(passed, total):
    return (passed / total * 100.0) if total > 0 else float("nan")


def _degradation_rate(if_per_turn, max_turn):
    """D = sum_{j=2..K} max(IF_1 - IF_j, 0) / (K - 1).

    The paper formula sums over a fixed (K-1) terms. If any turn 2..K has no
    samples we cannot compute that term, so D is NaN — silently dropping the
    term would bias D downward while keeping the divisor fixed.
    """
    if max_turn < 2:
        return float("nan")
    passed_1, total_1 = if_per_turn.get(1, (0, 0))
    if total_1 == 0:
        return float("nan")
    if_1 = _if_rate(passed_1, total_1)
    total = 0.0
    for j in range(2, max_turn + 1):
        p, t = if_per_turn.get(j, (0, 0))
        if t == 0:
            return float("nan")
        total += max(if_1 - _if_rate(p, t), 0.0)
    return total / (max_turn - 1)


def process_directory(audio_dir, paraphrasing_filter, max_turn):
    dim, info = _classify_dir(audio_dir)
    if dim is None:
        print(f"[Skip] {audio_dir}: cannot classify style dimension from basename")
        return None

    try:
        rows = read_metric_file(audio_dir, type=info["metric_type"])
    except FileNotFoundError as e:
        print(f"[Skip] {audio_dir}: {e}")
        return None

    style_key = os.path.basename(audio_dir)
    rows = _filter_rows(rows, paraphrasing_filter, max_turn)
    if_per_turn = _compute_if_per_turn(rows, info["checker"], style_key, max_turn)

    passed_1, total_1 = if_per_turn.get(1, (0, 0))
    if_1 = _if_rate(passed_1, total_1)
    d_rate = _degradation_rate(if_per_turn, max_turn)

    summary = {
        "dir": audio_dir,
        "dimension": dim,
        "style": style_key,
        "K": max_turn,
        "paraphrasing_filter": paraphrasing_filter,
        "IF1": if_1,
        "D": d_rate,
        "per_turn": {
            str(j): {
                "passed": if_per_turn[j][0],
                "total": if_per_turn[j][1],
                "IF": _if_rate(*if_per_turn[j]),
            }
            for j in range(1, max_turn + 1)
        },
    }

    out_path = os.path.join(audio_dir, "if1_degradation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[{dim}/{style_key}]  IF1={if_1:5.1f}%   D={d_rate:5.1f}   "
        + "  ".join(
            f"turn{j}={_if_rate(*if_per_turn[j]):5.1f}%({if_per_turn[j][0]}/{if_per_turn[j][1]})"
            for j in range(1, max_turn + 1)
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dirs",
        nargs="+",
        help="Result directories (e.g. results/<model>/<location>/<style>).",
    )
    parser.add_argument(
        "--paraphrasing",
        type=int,
        default=None,
        help="If set, only include records whose filename contains 'par<N>' (paper default: 1).",
    )
    parser.add_argument(
        "--max_turn",
        type=int,
        default=PAPER_K,
        help=f"Dialogue horizon K (default: {PAPER_K}).",
    )
    args = parser.parse_args()

    for d in args.audio_dirs:
        process_directory(d, paraphrasing_filter=args.paraphrasing, max_turn=args.max_turn)


if __name__ == "__main__":
    main()
