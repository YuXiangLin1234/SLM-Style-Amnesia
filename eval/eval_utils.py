import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import textstat
import torch
import torchaudio
import torchcrepe
from nltk.corpus import cmudict
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tabulate import tabulate

sys.path.append(os.path.dirname(__file__) + "/..")

_vad_model = load_silero_vad()

nltk.download("cmudict", quiet=True)
cmu_dict = cmudict.dict()

AUDIO_EXTS = {".wav", ".flac", ".mp3"}

# CREPE pitch detection parameters (speech-range).
_CREPE_FMIN = 50.0
_CREPE_FMAX = 550.0
_CREPE_PERIODICITY_THRESHOLD = 0.21
# 10ms hop -> 3 frames is roughly a 30ms smoothing window.
_CREPE_SMOOTHING_WIN = 3


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def extract_topic_index(filename):
    match = re.search(r"topic(\d+)", filename)
    if match:
        return int(match.group(1))
    raise ValueError("Cannot extract the index of the topic.")


def extract_tts_speed_index(filename):
    match = re.search(r"par_([0-9]+(?:\.[0-9]+)?)", filename)
    if match:
        return float(match.group(1))
    raise ValueError("Cannot extract the index of the tts speed.")


def extract_instruction_index(filename):
    match = re.search(r"par(\d+)", filename)
    if match:
        return int(match.group(1))
    raise ValueError("Cannot extract the index of the instruction.")


def extract_turn_index(fname):
    return int(fname.split("_turn")[-1].replace(".wav", ""))


def key_mapping(key):
    """Turn an instruction key like 'en_north_america' into 'North America'."""
    if "_" in key:
        _, tail = key.split("_", 1)
        return " ".join(word.capitalize() for word in tail.split("_"))
    return key


# ---------------------------------------------------------------------------
# Text / I/O helpers
# ---------------------------------------------------------------------------

def count_syllables_cmu(word):
    w = word.lower()
    if w in cmu_dict:
        return min(len([ph for ph in pron if ph[-1].isdigit()]) for pron in cmu_dict[w])
    return 0


def count_syllables_textstat(word):
    return textstat.syllable_count(word)


def read_txt_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_json_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def read_metric_file(dir, type=None):
    file_name = f"{type}_metrics.json" if type else "metrics.json"
    return read_json_file(os.path.join(dir, file_name))


def list_audio_files(dir):
    """List audio files in `dir`, skipping cascaded user-side and TTS-only outputs."""
    files = []
    for f in Path(dir).iterdir():
        if "_cascaded_" in str(f) or "TTS_" in str(f):
            continue
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            files.append(str(f))
    return sorted(files)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def to_mono(waveform):
    if waveform.dim() == 2 and waveform.size(0) > 1:
        return waveform.mean(dim=0)
    if waveform.dim() == 2 and waveform.size(0) == 1:
        return waveform[0]
    if waveform.dim() == 1:
        return waveform
    raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")


def load_audio_mono(path, sampling_rate=16000):
    wf, sr = torchaudio.load(path)
    wf = to_mono(wf)
    if sr != sampling_rate:
        wf = torchaudio.functional.resample(wf, sr, sampling_rate)
    return wf.float()


def get_speech_timestamps_seconds(wav_path, sr=16000):
    wav = read_audio(wav_path, sampling_rate=sr)
    speech_ts = get_speech_timestamps(wav, _vad_model, sampling_rate=sr, return_seconds=True)
    speech_time = sum(max(0, seg["end"] - seg["start"]) for seg in speech_ts)
    return speech_ts, speech_time


def _to_torch_1xT(x):
    """Convert numpy / tensor / sequence inputs to a contiguous [1, T] float32 tensor."""
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            x = x.mean(axis=-1)
        x = torch.from_numpy(x.astype(np.float32))
    elif hasattr(x, "detach"):
        x = x.detach().to(torch.float32)
        if x.ndim == 2:
            # Channel dim is whichever is shorter; average across it.
            channel_dim = 0 if x.shape[0] < x.shape[1] else -1
            x = x.mean(dim=channel_dim)
    else:
        x = torch.tensor(x, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x.contiguous()


def if_checker(number, mode):
    """Return True/False depending on `mode`. Returns False on None or unknown mode."""
    if number is None:
        return False
    if mode == "binary":
        return int(number) == 1
    if mode == "ratio_larger":
        return number > 1
    if mode == "ratio_smaller":
        return number < 1
    return False


# ---------------------------------------------------------------------------
# Pitch / volume
# ---------------------------------------------------------------------------

def detect_pitch_segment(samples, sr, hop_size=512, buffer_size=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = _to_torch_1xT(samples).to(device)

    # 10ms hop length.
    hop_length = max(1, int(sr * 0.01))

    pitch, periodicity = torchcrepe.predict(
        audio, sr, hop_length,
        fmin=_CREPE_FMIN, fmax=_CREPE_FMAX,
        model="full",
        batch_size=1024,
        device=device,
        return_periodicity=True,
    )

    # Official recipe: median-smooth periodicity, threshold-mask F0, then mean-smooth.
    periodicity = torchcrepe.filter.median(periodicity, _CREPE_SMOOTHING_WIN)
    pitch = torchcrepe.threshold.At(_CREPE_PERIODICITY_THRESHOLD)(pitch, periodicity)
    pitch = torchcrepe.filter.mean(pitch, _CREPE_SMOOTHING_WIN)

    pitch = pitch.squeeze(0).detach().cpu().numpy()
    n_frames = len(pitch)
    times = (np.arange(n_frames) * hop_length + 0.5 * hop_length) / float(sr)
    return np.array(times, dtype=np.float32), pitch.astype(np.float32)


def compute_pitch(wav_path, sr=16000, hop_size=512, buffer_size=2048):
    wav = read_audio(wav_path, sampling_rate=sr)
    speech_ts = get_speech_timestamps(wav, _vad_model, sampling_rate=sr, return_seconds=False)

    all_times = []
    all_pitches = []
    for seg in speech_ts:
        s, e = int(seg["start"]), int(seg["end"])
        # Skip segments shorter than 30ms (too short for reliable pitch).
        if (e - s) < int(sr * 0.03):
            continue
        times, pitches = detect_pitch_segment(
            wav[s:e], sr, hop_size=hop_size, buffer_size=buffer_size
        )
        all_times.append(times + (s / sr))
        all_pitches.append(pitches)

    if not all_pitches:
        return {
            "times": [],
            "pitches": [],
            "stats": {
                "mean": 0, "median": 0, "min": 0, "max": 0,
                "num_frames": 0, "num_valid_frames": 0,
            },
        }

    pitches_concat = np.concatenate(all_pitches)
    valid_pitches = pitches_concat[pitches_concat > 0]

    stats = {
        "mean": float(np.mean(valid_pitches)) if valid_pitches.size else 0.0,
        "median": float(np.median(valid_pitches)) if valid_pitches.size else 0.0,
        "min": float(np.min(valid_pitches)) if valid_pitches.size else 0.0,
        "max": float(np.max(valid_pitches)) if valid_pitches.size else 0.0,
        "num_frames": int(pitches_concat.size),
        "num_valid_frames": int(valid_pitches.size),
    }

    # Per-frame arrays are intentionally dropped from the output to keep metric files small.
    return {"times": None, "pitches": None, "stats": stats}


def compute_volume(wav_path):
    data, sr = sf.read(wav_path)
    meter = pyln.Meter(sr)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = data.astype(np.float32)
    rms_amplitude = float(np.sqrt(np.mean(data ** 2)))

    try:
        loudness = meter.integrated_loudness(data)
    except Exception as e:
        print("[ERROR] loudness meter:", e)
        loudness = None

    return {"loudness_integrated": loudness, "rms_amplitude": rms_amplitude}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def group_by_instruction(records):
    groups = {}
    for r in records:
        groups.setdefault(r.get("key"), []).append(r)
    return groups


def group_by_turn(records):
    groups = {}
    for r in records:
        groups.setdefault(extract_turn_index(r.get("file")), []).append(r)
    return groups


def group_by_parent_dir(dirs):
    groups = {}
    for d in dirs:
        groups.setdefault(os.path.dirname(d), []).append(d)
    return list(groups.values())


def display_stats(rows, max_turns=4):
    all_instr_ids = sorted({
        extract_instruction_index(r["file"])
        for r in rows
        if extract_instruction_index(r["file"]) is not None
    })

    table_rows = []
    for turn_id in range(max_turns):
        subset = [r for r in rows if extract_turn_index(r["file"]) == turn_id]

        counter = defaultdict(int)
        for r in subset:
            instr_id = extract_instruction_index(r["file"])
            if instr_id is not None:
                counter[instr_id] += 1

        row_dict = {"Turn": turn_id, "Total": len(subset)}
        for instr_id in all_instr_ids:
            row_dict[f"Instruction {instr_id}"] = counter[instr_id]
        table_rows.append(row_dict)

    headers = ["Turn", "Total"] + [f"Instruction {i}" for i in all_instr_ids]
    data_matrix = [[row[h] for h in headers] for row in table_rows]
    print(tabulate(data_matrix, headers=headers, tablefmt="fancy_grid"))


def aggregate_groups(groups, fields):
    keys = sorted(groups.keys())
    stats = {}
    for f in fields:
        arrs = [np.asarray([rec.get(f) for rec in groups[k]], dtype=float) for k in keys]
        mean = np.array([np.nanmean(a) for a in arrs])
        vmin = np.array([np.nanmin(a) for a in arrs])
        vmax = np.array([np.nanmax(a) for a in arrs])
        std = np.array([np.nanstd(a) for a in arrs])
        stats[f] = (mean, vmin, vmax, std)
    return keys, stats


def get_output_path(output_dir, dir_path, suffix, category=None):
    if category == "rank":
        parts = dir_path.split("/")[-3:-1]
    elif category == "comparison":
        parts = dir_path.split("/")[-2:]
    else:
        parts = dir_path.split("/")[-3:]
    output_name = "_".join(parts)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{output_name}{suffix}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color-blind friendly palette used for `plot_mean_rank`.
_PLOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#56b4e9", "#cc79a7",
    "#f0e442", "#009e73", "#332288", "#88ccee", "#ddcc77",
    "#44aa99", "#aa4499", "#117733", "#661100", "#6699cc",
    "#882255", "#bbbbbb", "#d55e00", "#0072b2", "#999933",
]

_PLOT_FONT_RC = {
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
}


def plot_mean_variance(
    keys, stats, fields,
    titles, ylabels, figure_title,
    out_path, ylims=None, legend_position="best",
):
    plt.rcParams.update(_PLOT_FONT_RC)

    x = np.arange(len(keys))
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 4 * len(fields) + 2), sharex=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, f, title, yl in zip(axes, fields, titles, ylabels):
        mean, vmin, vmax, std = stats[f]
        ax.scatter(x, mean, s=28, alpha=0.85, label="mean")
        ax.plot(x, mean, linewidth=1, alpha=0.7)
        ax.fill_between(x, mean - std, mean + std, color="orange", alpha=0.2, label="mean±std")
        ax.set_title(title)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_position)

        if ylims and f in ylims:
            ax.set_ylim(*ylims[f])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(keys, rotation=60, ha="right")
    axes[-1].set_xlabel("Turns")

    fig.suptitle(figure_title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mean_rank(
    all_keys, all_stats, all_labels,
    fields, titles, ylabels,
    figure_title, out_path, ylims=None,
    plot_variance=True, legend_position="best",
):
    plt.rcParams.update({**_PLOT_FONT_RC, "legend.fontsize": 16})
    fig, axes = plt.subplots(len(fields), 1, figsize=(16, 4 * len(fields) + 2), sharex=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, f, title, yl in zip(axes, fields, titles, ylabels):
        for i, (keys, stats, label) in enumerate(zip(all_keys, all_stats, all_labels)):
            x = np.arange(len(all_keys[i]))
            color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
            mean, vmin, vmax, std = stats[f]
            ax.scatter(x, mean, s=28, alpha=0.85, color=color)
            ax.plot(x, mean, linewidth=2, alpha=1, color=color, label=key_mapping(label))
            if plot_variance:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
            ax.set_title(title)
            ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            if ylims and f in ylims:
                ax.set_ylim(*ylims[f])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(keys, rotation=0, ha="right")
    axes[-1].set_xlabel("Turn")
    ax.legend(loc=legend_position)

    fig.suptitle(figure_title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_and_save(
    records, fields, titles,
    ylabels, dir_path, output_dir, suffix, ylims,
    category=None, plot_variance=True, legend_position="lower right",
):
    if isinstance(records, list):
        groups = group_by_turn(records)
        keys, stats = aggregate_groups(groups, fields)
        out_path = get_output_path(output_dir, dir_path, suffix)
        plot_mean_variance(
            keys, stats, fields, titles, ylabels, "", out_path,
            ylims=ylims, legend_position=legend_position,
        )
        return out_path

    if isinstance(records, dict):
        all_keys, all_stats, all_labels = [], [], []
        for dir_, record in records.items():
            groups = group_by_turn(record)
            keys, stats = aggregate_groups(groups, fields)
            all_keys.append(keys)
            all_stats.append(stats)
            all_labels.append(
                dir_.split("/")[-3] if category == "comparison" else os.path.basename(dir_)
            )
        out_path = get_output_path(output_dir, dir_path, suffix, category)
        plot_mean_rank(
            all_keys, all_stats, all_labels, fields, titles, ylabels, "",
            out_path, ylims=ylims, plot_variance=plot_variance, legend_position=legend_position,
        )
        return out_path

    raise TypeError(f"Unsupported records type: {type(records)}")
