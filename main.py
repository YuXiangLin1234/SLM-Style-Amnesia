"""Deprecated entrypoint.

Use model-specific scripts instead:
- inference_gpt.py
- inference_gemini.py
- inference_stepaudio2.py
"""


if __name__ == "__main__":
    raise SystemExit(
        "`main.py` is deprecated. Run a model-specific script "
        "(e.g. `python inference_gpt.py`)."
    )
