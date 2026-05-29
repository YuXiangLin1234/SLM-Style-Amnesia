"""Emotion judge: runs emotion2vec-Large over every assistant audio in a result
directory and writes per-file 9-class probabilities to `emotion_metrics.json`.

The 4-way reduction (Anger / Happiness / Neutral / Sadness) used by the paper
happens at the consumer side (`compute_emotion_ifrate.py`).
"""
import argparse
import json
import os
import tempfile

import soundfile as sf
from funasr import AutoModel
from tqdm import tqdm


EMOTION_MODEL_ID = "emotion2vec/emotion2vec_plus_large"

# Order matches the model's score vector — do not reorder.
EMOTION_LABELS = [
    "Anger", "Disgust", "Fear", "Happiness", "Neutral",
    "Other", "Sadness", "Surprise", "<unk>",
]

MAX_DURATION_SEC = 180

_emotion_model = None


def _get_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        _emotion_model = AutoModel(model=EMOTION_MODEL_ID, hub="hf")
    return _emotion_model


def compute_emotion(audio_path):
    """Return a 9-class probability dict over EMOTION_LABELS for one wav."""
    info = sf.info(audio_path)
    actual_path = audio_path
    temp_path = None

    if info.duration > MAX_DURATION_SEC:
        max_frames = int(MAX_DURATION_SEC * info.samplerate)
        audio_data, sr = sf.read(audio_path, frames=max_frames)
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, audio_data, sr)
        actual_path = temp_path

    model = _get_emotion_model()
    result = model.generate(actual_path, output_dir="./.emotion2vec_tmp", granularity="utterance")
    scores = result[0]["scores"]

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return {label: float(scores[i]) for i, label in enumerate(EMOTION_LABELS)}


def _list_target_wavs(audio_dir):
    files = []
    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav"):
            continue
        if "_cascaded_" in fname or fname.startswith("TTS_"):
            continue
        files.append(fname)
    return sorted(files)


def _load_existing_records(json_path):
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception as e:
        print(f"[Warning] Failed to load existing JSON {json_path}: {e}")
        return []


def analyze_directory(audio_dir, overwrite=False):
    json_path = os.path.join(audio_dir, "emotion_metrics.json")
    existing = [] if overwrite else _load_existing_records(json_path)
    # Records use basename for the `file` field, matching the consumer.
    existing_files = {r["file"] for r in existing}

    new_records = []
    for fname in _list_target_wavs(audio_dir):
        if fname in existing_files:
            continue
        path = os.path.join(audio_dir, fname)
        rec = {"file": fname, "dir": os.path.basename(audio_dir)}
        try:
            rec["emotion"] = compute_emotion(path)
        except Exception as e:
            print(f"[Error] {path}: {e}")
            continue
        y, sr = sf.read(path)
        rec["duration"] = len(y) / sr if sr else 0.0
        new_records.append(rec)

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
        help="Result directories (e.g. results/<model>/<location>/en_anger).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for d in tqdm(args.audio_dirs):
        analyze_directory(d, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
