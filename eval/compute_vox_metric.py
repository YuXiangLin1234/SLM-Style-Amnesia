import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


def parse_target_from_filename(fname):
    stem = Path(fname).stem
    parts = stem.split('_')
    if len(parts) <= 2:
        return None

    key = parts[2]
    return key 

def argmax_key(d: dict):
    return max(d.items(), key=lambda kv: kv[1])[0]

def analyze(json_path: Path, thresholds=(0.5, 0.8)):
    data = json.loads(json_path.read_text(encoding='utf-8'))
    type = os.path.basename(json_path).split(".")[0].split("_")[1]
    items = data["results"]
    labels = data["labels"]

    total = 0
    skipped = 0

    top1_correct = 0
    thr_hits = Counter()         
    cls_total = Counter()     
    cls_top1 = Counter()        
    cls_thr_hits = {t: Counter() for t in thresholds}

    for it in items:
        fname = it.get('file') or Path(it.get('path','')).name
        if not fname:
            continue

        target = parse_target_from_filename(fname)
        if target is None:
            skipped += 1
            continue

        file_level = it.get('file_level', {}) or {}
        probs = file_level.get('probs') or {}
        if not probs:
            skipped += 1
            continue

        if target not in probs:
            skipped += 1
            continue

        total += 1
        cls_total[target] += 1

        if type in ("emotion", "accent"):
            pred = argmax_key(probs)
            if pred == target:
                top1_correct += 1
                cls_top1[target] += 1

        elif type in ("quality"):
            for t in thresholds:
                if probs[target] >= t:
                    thr_hits[t] += 1
                    cls_thr_hits[t][target] += 1
        else:
            raise ValueError("Not implemented.")

    base = {
        "num_items_total": len(items),
        "num_evaluated": total,
        "num_skipped": skipped,
        "per_class": {}
    }

    if type in ("emotion", "accent"):
        # Multi-class: report top-1 only
        report = {
            **base,
            "top1_accuracy_overall": (top1_correct / total) if total else None,
        }

        for label in labels:
            denom = cls_total[label]
            if denom == 0:
                continue
            report["per_class"][label] = {
                "count": denom,
                "top1_acc": (cls_top1[label] / denom) if denom else None,
            }

    elif type == "quality":
        # Multi-label: report threshold hit rate only
        report = {
            **base,
            "threshold_hit_rate_overall": {
                str(t): (thr_hits[t] / total) if total else None
                for t in thresholds
            },
        }

        for label in labels:
            denom = cls_total[label]
            if denom == 0:
                continue
            report["per_class"][label] = {
                "count": denom,
                "thr_hit_rate": {
                    str(t): (cls_thr_hits[t][label] / denom) if denom else None
                    for t in thresholds
                },
            }

    else:
        raise ValueError("Not implemented.")

    return report

def main():
    parser = argparse.ArgumentParser(
        description="Analyze voice-quality JSON: parse filename labels, compare with file_level.probs, and compute accuracies."
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        required=True,
        help="Path to the voice_quality.json file.",
    )
    parser.add_argument(
        "--thresholds", "-t",
        type=float,
        nargs="*",
        default=[0.5, 0.8],
        help="List of probability thresholds to evaluate (default: 0.5 0.8)."
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print the JSON report with indentation."
    )
    args = parser.parse_args()

    rep = analyze(args.json_path, thresholds=tuple(args.thresholds))
    if args.pretty:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rep, ensure_ascii=False))

if __name__ == "__main__":
    main()
