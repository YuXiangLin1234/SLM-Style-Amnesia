"""Heuristic metric extractor for the Speed / Volume style dimensions.

For each `.wav` in the result directory we compute:
  - speed: VAD-derived `speech_time`, words via Parakeet TDT v2 ASR, `words_per_min`
           and `syll_cmu_per_min`. When a paired `TTS_<basename>.wav` exists,
           also compute `tts_ratio_word` = wpm_styled / wpm_reference (paper §3.3.1).
  - volume: integrated LUFS loudness + RMS amplitude via PyLoudnorm. When a
            paired reference exists, also compute `tts_ratio_loudness`.

The neutral TTS reference o'_{i,j} is produced by `tts_baseline_*.py`.

Outputs:
  results/<model>/<location>/<style>/heuristic_metrics.json
"""
import argparse
import json
import os
import sys
import tempfile

import soundfile as sf
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from eval_utils import (
    compute_volume,
    count_syllables_cmu,
    count_syllables_textstat,
    get_speech_timestamps_seconds,
)
from instruction_collection import SPEED_KEY, VOLUME_KEY


PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
MAX_ASR_DURATION_SEC = 180

_asr_model = None


def _get_asr_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL)
        if torch.cuda.is_available():
            _asr_model = _asr_model.to("cuda")
    return _asr_model


def _transcribe(audio_path):
    """Transcribe with Parakeet TDT v2, truncating audio longer than 180s."""
    info = sf.info(audio_path)
    actual_path = audio_path
    temp_path = None

    if info.duration > MAX_ASR_DURATION_SEC:
        max_frames = int(MAX_ASR_DURATION_SEC * info.samplerate)
        audio_data, sr = sf.read(audio_path, frames=max_frames)
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, audio_data, sr)
        actual_path = temp_path

    model = _get_asr_model()
    output = model.transcribe([actual_path], timestamps=True)
    text = output[0].text if hasattr(output[0], "text") else output[0]

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    return (text or "").strip()


def _compute_speed_record(audio_path):
    """Speed-related fields: speech_time, transcript, word/syllable counts and rates."""
    rec = {}

    _, speech_time = get_speech_timestamps_seconds(audio_path)
    rec["speech_time"] = speech_time

    text = _transcribe(audio_path)
    rec["text"] = text

    words = text.split()
    rec["words"] = len(words)
    rec["syllables_cmu"] = sum(count_syllables_cmu(w) for w in words)
    rec["syllables_textstat"] = sum(count_syllables_textstat(w) for w in words)

    speech_time_min = speech_time / 60.0 if speech_time and speech_time > 0 else 0.0
    if speech_time_min > 0:
        rec["words_per_min"] = rec["words"] / speech_time_min
        rec["syll_cmu_per_min"] = rec["syllables_cmu"] / speech_time_min
        rec["syll_text_per_min"] = rec["syllables_textstat"] / speech_time_min

    return rec


def _compute_volume_record(audio_path):
    return compute_volume(audio_path)


def _tts_reference_path(audio_path):
    return os.path.join(os.path.dirname(audio_path), "TTS_" + os.path.basename(audio_path))


def _analyze_file(path, metrics, with_tts_ratio):
    """Compute all requested metrics for one wav file."""
    rec = {"file": path}

    y, sr = sf.read(path)
    rec["duration"] = len(y) / sr if sr else 0.0

    if "speed" in metrics:
        rec.update(_compute_speed_record(path))

    if "volume" in metrics:
        rec.update(_compute_volume_record(path))

    if not with_tts_ratio:
        return rec

    tts_path = _tts_reference_path(path)
    if not os.path.exists(tts_path):
        print(f"[WARNING] Missing TTS reference: {tts_path}")
        return rec

    tts_rec = _analyze_file(tts_path, metrics, with_tts_ratio=False)
    if "speed" in metrics and rec.get("words_per_min") and tts_rec.get("words_per_min"):
        rec["tts_ratio_word"] = rec["words_per_min"] / tts_rec["words_per_min"]
        if rec.get("syll_cmu_per_min") and tts_rec.get("syll_cmu_per_min"):
            rec["tts_ratio_syll"] = rec["syll_cmu_per_min"] / tts_rec["syll_cmu_per_min"]

    if "volume" in metrics and rec.get("loudness_integrated") and tts_rec.get("loudness_integrated"):
        rec["tts_ratio_loudness"] = rec["loudness_integrated"] / tts_rec["loudness_integrated"]
        if rec.get("rms_amplitude") and tts_rec.get("rms_amplitude"):
            rec["tts_ratio_rms"] = rec["rms_amplitude"] / tts_rec["rms_amplitude"]

    return rec


def _list_target_wavs(audio_dir):
    """Skip cascaded user audio and TTS_ reference clips."""
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


def _infer_metrics_from_dir(audio_dir):
    """Pick the metric list based on the directory name (style key)."""
    style = os.path.basename(audio_dir)
    if style in SPEED_KEY:
        return ["speed"]
    if style in VOLUME_KEY:
        return ["volume"]
    return []


def analyze_directory(audio_dir, metrics=None, overwrite=False, with_tts_ratio=True):
    metrics = metrics or _infer_metrics_from_dir(audio_dir)
    if not metrics:
        print(f"[Skip] No metric inferred for {audio_dir}")
        return

    json_path = os.path.join(audio_dir, "heuristic_metrics.json")
    existing = [] if overwrite else _load_existing_records(json_path)
    existing_paths = {r["file"] for r in existing}

    new_records = []
    for fname in _list_target_wavs(audio_dir):
        path = os.path.join(audio_dir, fname)
        if path in existing_paths:
            continue
        rec = _analyze_file(path, metrics, with_tts_ratio=with_tts_ratio)
        rec["dir"] = os.path.basename(audio_dir)
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
        help="Result directories (e.g. results/<model>/<location>/en_fast).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["speed", "volume"],
        default=None,
        help="Override the metric list (default: inferred from directory name).",
    )
    parser.add_argument(
        "--no_tts_ratio",
        action="store_true",
        help="Skip the tts_ratio_* fields (use when no TTS_ reference exists).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for d in args.audio_dirs:
        analyze_directory(
            d,
            metrics=args.metrics,
            overwrite=args.overwrite,
            with_tts_ratio=not args.no_tts_ratio,
        )


if __name__ == "__main__":
    main()
