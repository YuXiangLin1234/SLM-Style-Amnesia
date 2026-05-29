"""Generate the neutral TTS-style reference audio o'_{i,j} for Gemini Live.

For every assistant transcript `.txt` in the result directory, query Gemini
Live with a neutral TTS prompt and save the result as `TTS_<basename>.wav`.

The reference is consumed by `eval/save_heuristic_metric.py` to compute
`tts_ratio_word` (speed) and `tts_ratio_loudness` (volume) per paper §3.3.1.
"""
import argparse
import asyncio
import io
import os
import re
import wave

from google.genai import types

from inference_gemini import MODEL_MAP, get_gemini_client
from inference_utils import save_audio
from instruction_collection import SPEED_KEY, VOLUME_KEY


TTS_PROMPTS = {
    "speed":  "You are a text-to-speech model. Please read the given text at a normal speed without adding or omitting anything.",
    "volume": "You are a text-to-speech model. Please read the given text at a normal volume without adding or omitting anything.",
}

# Gemini Live emits 24kHz, 16-bit, mono PCM.
_GEMINI_SR = 24000


def _infer_dimension(audio_dir):
    style = os.path.basename(audio_dir)
    if style in SPEED_KEY:
        return "speed"
    if style in VOLUME_KEY:
        return "volume"
    return None


async def synthesize_neutral(model_id, text, dimension):
    """Single-turn Gemini Live call, returns WAV bytes."""
    config = {
        "response_modalities": ["AUDIO"],
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": False, "silence_duration_ms": 3000}
        },
        "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
        "output_audio_transcription": {},
    }

    full_text = TTS_PROMPTS[dimension] + "\n\n" + text

    audio_chunks = []
    client = get_gemini_client()
    async with client.aio.live.connect(model=model_id, config=config) as session:
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": full_text}]},
            turn_complete=True,
        )
        async for resp in session.receive():
            if getattr(resp, "data", None):
                audio_chunks.append(resp.data)
            sc = getattr(resp, "server_content", None)
            if sc and getattr(sc, "turn_complete", False):
                break

    raw_pcm = b"".join(audio_chunks)
    if not raw_pcm:
        raise RuntimeError("No audio returned by Gemini Live")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_GEMINI_SR)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


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


async def process_directory(audio_dir, model_key, dimension=None, paraphrasing_filter=None, overwrite=False):
    dimension = dimension or _infer_dimension(audio_dir)
    if dimension is None:
        print(f"[Skip] {audio_dir}: no speed/volume style — neutral TTS not applicable")
        return

    model_id = MODEL_MAP[model_key]
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
            wav_bytes = await synthesize_neutral(model_id, text, dimension)
            save_audio(wav_bytes, out_path)
            print(f"[OK] {txt_path} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {txt_path}: {e}")
            if "resource_exhausted" in str(e).lower():
                print("[FATAL] quota exhausted, stopping")
                raise


async def _amain(args):
    for d in args.audio_dirs:
        await process_directory(
            d,
            model_key=args.model,
            dimension=args.dimension,
            paraphrasing_filter=args.paraphrasing,
            overwrite=args.overwrite,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dirs", nargs="+")
    parser.add_argument("--model", default="gemini-live", choices=sorted(MODEL_MAP.keys()))
    parser.add_argument("--dimension", choices=sorted(TTS_PROMPTS.keys()), default=None)
    parser.add_argument("--paraphrasing", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
