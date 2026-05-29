"""Accent judge: runs Voxlect English Dialect Whisper Large v3 over every
assistant audio in a result directory and writes per-file 16-class accent
probabilities to `accent_metrics.json`.

The 3-way reduction (North America / South Asia / English) used by the paper
happens at the consumer side (`compute_accent_ifrate.py`).

Voxlect is loaded from the user's local clone of
https://github.com/tiantiaf0627/voxlect — set `VOXLECT_REPO` to that path.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from eval_utils import (
    AUDIO_EXTS,
    get_speech_timestamps_seconds,
    list_audio_files,
    load_audio_mono,
)


VOXLECT_REPO = os.environ.get("VOXLECT_REPO", "")
VOXLECT_MODEL = "tiantiaf/voxlect-english-dialect-whisper-large-v3"

# Label order is fixed by the model; do not reorder.
ACCENT_LABELS = [
    "East Asia", "English", "Germanic",
    "Irish", "North America", "Northern Irish",
    "Oceania", "Other", "Romance",
    "Scottish", "Semitic", "Slavic",
    "South African", "Southeast Asia",
    "South Asia", "Welsh",
]

TARGET_SR = 16000
WIN_SEC = 15
MIN_LAST_SEC = 3
MAX_DURATION_SEC = 180


def _load_voxlect():
    if VOXLECT_REPO:
        sys.path.insert(0, VOXLECT_REPO)
    try:
        from src.model.dialect.whisper_dialect import WhisperWrapper
    except ImportError as e:
        raise ImportError(
            "Could not import Voxlect's WhisperWrapper. Clone "
            "https://github.com/tiantiaf0627/voxlect and set VOXLECT_REPO."
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperWrapper.from_pretrained(VOXLECT_MODEL).to(device)
    model.eval()
    return model, device


def _sliding_chunks(wav, win_sec, min_last_sec, sr):
    win_len = win_sec * sr
    min_last_len = min_last_sec * sr
    total_len = wav.numel()
    chunks = []

    idx = 0
    while idx + win_len < total_len:
        chunks.append({
            "start": idx / sr,
            "end": (idx + win_len) / sr,
            "tensor": wav[idx: idx + win_len].unsqueeze(0),
        })
        idx += win_len

    if idx < total_len:
        tail = wav[idx:]
        if tail.numel() >= min_last_len:
            chunks.append({
                "start": idx / sr,
                "end": total_len / sr,
                "tensor": tail.unsqueeze(0),
            })
    return chunks


@torch.no_grad()
def _infer_chunk(model, device, chunk_tensor):
    logits, _ = model(chunk_tensor.to(device), return_feature=True)
    probs = F.softmax(logits[0], dim=0)
    return {label: float(probs[i].item()) for i, label in enumerate(ACCENT_LABELS)}


@torch.no_grad()
def _infer_file(model, device, audio_path):
    wav = load_audio_mono(audio_path, sampling_rate=TARGET_SR)

    # Truncate by VAD end-of-speech timestamp, capped at MAX_DURATION_SEC.
    max_samples = MAX_DURATION_SEC * TARGET_SR
    ts, _ = get_speech_timestamps_seconds(audio_path)
    if ts:
        max_samples = min(int(ts[-1]["end"] * TARGET_SR), max_samples)
    if wav.numel() > max_samples:
        wav = wav[:max_samples]

    # Pad to at least MIN_LAST_SEC so the encoder always has input.
    min_samples = MIN_LAST_SEC * TARGET_SR
    if wav.numel() < min_samples:
        wav = F.pad(wav, (0, min_samples - wav.numel()), "constant", 0)

    chunks = _sliding_chunks(wav, WIN_SEC, MIN_LAST_SEC, TARGET_SR)

    per_chunk = []
    prob_accum = torch.zeros(len(ACCENT_LABELS), dtype=torch.float32, device=device)
    for ch in chunks:
        chunk_probs = _infer_chunk(model, device, ch["tensor"])
        per_chunk.append({"start_sec": ch["start"], "end_sec": ch["end"], "probs": chunk_probs})
        prob_accum += torch.tensor(
            [chunk_probs[label] for label in ACCENT_LABELS], device=device
        )

    if chunks:
        avg = (prob_accum / len(chunks)).cpu()
        file_probs = {label: float(avg[i].item()) for i, label in enumerate(ACCENT_LABELS)}
    else:
        file_probs = {}

    return {
        "file": os.path.basename(audio_path),
        "path": audio_path,
        "duration_sec": wav.numel() / TARGET_SR,
        "sample_rate": TARGET_SR,
        "num_chunks": len(chunks),
        "chunks": per_chunk,
        "accent": file_probs,
    }


def _load_existing_records(json_path):
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception as e:
        print(f"[Warning] Failed to load existing JSON {json_path}: {e}")
        return []


def analyze_directory(model, device, audio_dir, overwrite=False):
    json_path = os.path.join(audio_dir, "accent_metrics.json")
    existing = [] if overwrite else _load_existing_records(json_path)
    existing_files = {r["file"] for r in existing}

    new_records = []
    for path in list_audio_files(audio_dir):
        if os.path.basename(path) in existing_files:
            continue
        if os.path.splitext(path)[1].lower() not in AUDIO_EXTS:
            continue
        try:
            rec = _infer_file(model, device, path)
            rec["dir"] = os.path.basename(audio_dir)
            new_records.append(rec)
        except Exception as e:
            print(f"[Error] {path}: {e}")

    out = existing + new_records
    os.makedirs(audio_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {json_path} ({len(out)} records)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dirs",
        nargs="+",
        help="Result directories (e.g. results/<model>/<location>/en_indian).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model, device = _load_voxlect()
    for d in tqdm(args.audio_dirs):
        analyze_directory(model, device, d, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
