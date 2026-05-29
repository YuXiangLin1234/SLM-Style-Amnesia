import base64
import copy
import io
import os
import random

import librosa
import numpy as np
import resampy
import soundfile as sf
import torch
from scipy.io.wavfile import write

INPUT_SAMPLING_RATE = 16000


def fix_random_seed(seed=0, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False


def list_en_wavs(root_dir):
    """Return all `*_en.wav` files under `root_dir` (recursive)."""
    wav_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            stem, ext = os.path.splitext(filename)
            if ext == ".wav" and stem.endswith("_en"):
                wav_list.append(os.path.join(dirpath, filename))
    return wav_list


def load_as_pcm16(audio_path, sampling_rate=INPUT_SAMPLING_RATE):
    """Load an audio file and return raw PCM-16 bytes at `sampling_rate`."""
    wav, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
    buf = io.BytesIO()
    sf.write(buf, wav, sampling_rate, format="RAW", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def concat_audios_pcm16(
    first_audio_path,
    second_audio_path,
    sampling_rate=INPUT_SAMPLING_RATE,
    silence_sec=1.0,
):
    """Concatenate two audios with a silence gap, return raw PCM-16 bytes."""
    first, _ = librosa.load(str(first_audio_path), sr=sampling_rate, mono=True)
    second, _ = librosa.load(str(second_audio_path), sr=sampling_rate, mono=True)
    silence = np.zeros(int(sampling_rate * silence_sec), dtype=np.float32)
    combined = np.concatenate([first, silence, second])

    buf = io.BytesIO()
    sf.write(buf, combined, sampling_rate, format="RAW", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def wav_file_to_base64(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    buffer = io.BytesIO()
    write(buffer, sr, audio.astype(np.float32))
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def save_wav_bytes(path, wav_bytes):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(wav_bytes)


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio, orig_sr
    resampled = resampy.resample(audio, orig_sr, target_sr, filter="kaiser_best")
    return resampled, target_sr


def save_audio(audio, save_path, samplerate=24000):
    """Write audio to `save_path`, then re-encode at INPUT_SAMPLING_RATE for consistency."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(audio, (bytes, bytearray)):
        with open(save_path, "wb") as f:
            f.write(audio)
    elif isinstance(audio, np.ndarray):
        sf.write(save_path, audio, samplerate)
    elif isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().view(-1).numpy()
        sf.write(save_path, audio_np, samplerate)
    else:
        raise TypeError(f"Unsupported input type: {type(audio)}")

    # Normalize all outputs to INPUT_SAMPLING_RATE so downstream eval sees one rate.
    wav, sr = sf.read(save_path, dtype="float32")
    wav, sr = resample_audio(wav, sr, INPUT_SAMPLING_RATE)
    sf.write(save_path, wav, INPUT_SAMPLING_RATE)

    return save_path


def naming(model_key, role):
    return f"{model_key}_{role}"


def strip_base64_from_history(history):
    """Return a copy of `history` with audio/token blobs replaced by '<omitted>'."""
    cleaned = []
    for msg in copy.deepcopy(history):
        content = msg.get("content")

        if isinstance(content, list) and all(isinstance(c, dict) for c in content):
            msg["content"] = [
                {"type": c["type"], c["type"]: "<omitted>"}
                if c.get("type") in ("input_audio", "token")
                else c
                for c in content
            ]
        # list-of-str and str cases need no transformation: any audio is referenced
        # by file path, not embedded base64.

        cleaned.append(msg)
    return cleaned
