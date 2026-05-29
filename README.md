# Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models

🚧 **Under Construction** 🚧

[![Paper](https://img.shields.io/badge/Paper-ACL_Findings_2026-blue.svg)](https://arxiv.org/abs/2512.23578) [![Demo](https://img.shields.io/badge/Demo-Website-orange.svg)](https://yuxianglin1234.github.io/SLM-Style-Amnesia/) [![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for the ACL Findings 2026 paper *"Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models"*.

The benchmark covers 10 speaking styles across 4 dimensions (emotion, accent, speed, volume) over 100 SODA conversation openers, giving 1,000 dialogues per spoken language model. K = 4 turns by default.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY and GOOGLE_API_KEY
```

`OPENAI_API_KEY` is needed even when running Gemini or Step-Audio-2, because the cascaded user simulator uses GPT-5 mini + GPT-4o mini TTS.

External judges and models:

- Step-Audio-2 (local vLLM): https://github.com/stepfun-ai/Step-Audio2
  Set `STEP_AUDIO2_REPO`, `STEP_AUDIO2_VLLM_URL`, `STEP_AUDIO2_MODEL`.
- emotion2vec-Large (`emotion2vec/emotion2vec_plus_large`, FunASR): https://github.com/ddlBoJack/emotion2vec
- Voxlect English Dialect (`tiantiaf/voxlect-english-dialect-whisper-large-v3`): https://github.com/tiantiaf0627/voxlect
  Set `VOXLECT_REPO` to the clone path.
- Parakeet TDT v2 (`nvidia/parakeet-tdt-0.6b-v2`) loads automatically via NeMo.

## Inference

```bash
python inference_gpt.py        --model gpt-4o          --instruction_location system
python inference_gemini.py     --model gemini-live     --instruction_location user
python inference_stepaudio2.py --model step-audio2-mini
```

Each script has a `-recall` model variant (`gpt-4o-recall`, `gemini-live-recall`, `step-audio2-mini-recall`) that re-asks the model to restate the style instruction before every turn after the first.

Output: `results/<model>/<location>/<style>/topic<i>_par<j>_<model>_A_turn<k>.wav` plus matching `.txt` transcripts and a `transcript.json`. All audio is resampled to 16 kHz. Runs are resumable — a conversation whose final-turn wav already exists is skipped.

CLI flags shared across scripts: `--instruction_location {system,user}`, `--max_turns`, `--topic_end_index`, `--instruction_index`.

## Evaluation

Three stages: generate the neutral TTS reference (speed/volume only), run the per-file judges, aggregate into IF₁ and D.

### Neutral TTS baseline

For each `.txt` in a result directory, query the same model with *"You are a text-to-speech model. Please read the given text at a normal {speed|volume} without adding or omitting anything."* The reference is saved as `TTS_<basename>.wav` next to the styled audio.

```bash
python tts_baseline_gpt.py        --model gpt-4o      results/gpt-4o/system/en_fast results/gpt-4o/system/en_slow
python tts_baseline_gemini.py     --model gemini-live results/gemini-live/user/en_loud results/gemini-live/user/en_quiet
python tts_baseline_stepaudio2.py                     results/step-audio2-mini/user/en_fast
```

### Per-file judges

```bash
python eval/save_emotion_metric.py   results/gpt-4o/system/en_anger   # → emotion_metrics.json
python eval/save_accent_metric.py    results/gpt-4o/system/en_indian  # → accent_metrics.json
python eval/save_heuristic_metric.py results/gpt-4o/system/en_fast    # → heuristic_metrics.json (WPM + LUFS + tts_ratio_*)
```

### IF₁ and D

```bash
python eval/compute_if1_degradation.py results/gpt-4o/system/en_anger results/gpt-4o/system/en_fast --paraphrasing 1
```

The dimension is auto-detected from the directory's basename. Output is printed and written to `if1_degradation.json` in each result directory.

### Per-turn breakdown

```bash
python eval/compute_emotion_ifrate.py results/gpt-4o/system/en_anger
python eval/compute_accent_ifrate.py  results/gpt-4o/system/en_north_america
python eval/compute_speed_rank.py     results/gpt-4o/system/en_fast results/gpt-4o/system/en_slow
python eval/compute_volume_rank.py    results/gpt-4o/system/en_loud results/gpt-4o/system/en_quiet
```

## Adding a new style

Add an entry to `INSTRUCTIONS` in `instruction_collection.py` (style key plus three paraphrasings) and register the key in the matching set (`EMOTION_KEY` / `ACCENT_KEY` / `SPEED_KEY` / `VOLUME_KEY`). For Gemini in user mode, also drop three matching audios into `data/instruction_audios/`. For emotion / accent, extend `MAPPING` in `eval/compute_if1_degradation.py` so the new style maps to a judge label.

## Citation

```bibtex
@inproceedings{lin2026styleamnesia,
  title     = {Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models},
  author    = {Lin, Yu-Xiang and Chiang, Cheng-Han and Lee, Hung-yi},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.23578}
}
```
