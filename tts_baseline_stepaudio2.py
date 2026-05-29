"""Generate the neutral TTS-style reference audio o'_{i,j} for Step-Audio-2-mini.

For every assistant transcript `.txt` in the result directory, query the local
Step-Audio-2-mini vLLM server with a neutral TTS prompt and save the result as
`TTS_<basename>.wav`.

The reference is consumed by `eval/save_heuristic_metric.py` to compute
`tts_ratio_word` (speed) and `tts_ratio_loudness` (volume) per paper §3.3.1.
"""
import argparse
import os
import re

from inference_stepaudio2 import MODEL_KEYS, stepaudio2_vllm_call
from inference_utils import save_audio
from instruction_collection import SPEED_KEY, VOLUME_KEY


TTS_PROMPTS = {
    "speed":  "You are a text-to-speech model. Please read the given text at a normal speed without adding or omitting anything.",
    "volume": "You are a text-to-speech model. Please read the given text at a normal volume without adding or omitting anything.",
}


def _infer_dimension(audio_dir):
    style = os.path.basename(audio_dir)
    if style in SPEED_KEY:
        return "speed"
    if style in VOLUME_KEY:
        return "volume"
    return None


def synthesize_neutral(text, dimension):
    history = [
        {
            "role": "human",
            "content": [{"type": "text", "text": TTS_PROMPTS[dimension] + "\n\n" + text}],
        },
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    wav_bytes, _, _ = stepaudio2_vllm_call(history)
    return wav_bytes


def _list_source_txts(audio_dir, paraphrasing_filter):
    files = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".txt"):
            continue
        if "_cascaded_" in fname or fname.startswith("TTS_"):
            continue
        if paraphrasing_filter is not None:
            if not re.search(rf"_par_?{paraphrasing_filter}_", fname):
                continue
        files.append(os.path.join(audio_dir, fname))
    return files


def process_directory(audio_dir, dimension=None, paraphrasing_filter=None, overwrite=False):
    dimension = dimension or _infer_dimension(audio_dir)
    if dimension is None:
        print(f"[Skip] {audio_dir}: no speed/volume style — neutral TTS not applicable")
        return

    for txt_path in _list_source_txts(audio_dir, paraphrasing_filter):
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(audio_dir, f"TTS_{stem}.wav")
        if os.path.exists(out_path) and not overwrite:
            print(f"[Skip] {out_path}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            print(f"[Skip empty] {txt_path}")
            continue

        try:
            wav_bytes = synthesize_neutral(text, dimension)
            save_audio(wav_bytes, out_path)
            print(f"[OK] {txt_path} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {txt_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dirs", nargs="+")
    parser.add_argument(
        "--model",
        default="step-audio2-mini",
        choices=MODEL_KEYS,
        help="Unused dispatch placeholder (the vLLM backend is fixed by env vars).",
    )
    parser.add_argument("--dimension", choices=sorted(TTS_PROMPTS.keys()), default=None)
    parser.add_argument("--paraphrasing", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for d in args.audio_dirs:
        process_directory(
            d,
            dimension=args.dimension,
            paraphrasing_filter=args.paraphrasing,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
