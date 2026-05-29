import os
import sys

from eval_common import (
    apply_instruction_index,
    default_system_message,
    load_instructions,
    load_topics,
    parse_base_args,
    resolve_output_dir,
    run_batch,
)
from inference_utils import fix_random_seed, save_audio


# Point STEP_AUDIO2_REPO at a local clone of https://github.com/stepfun-ai/Step-Audio2,
# which provides stepaudio2vllm.py, token2wav.py, and the assets/ directory.
STEP_AUDIO2_REPO = os.environ.get("STEP_AUDIO2_REPO", "")
VLLM_URL = os.environ.get("STEP_AUDIO2_VLLM_URL", "http://localhost:8000/v1/chat/completions")
STEP_MODEL_NAME = os.environ.get("STEP_AUDIO2_MODEL", "Step-Audio-2-mini")

try:
    if STEP_AUDIO2_REPO:
        sys.path.append(STEP_AUDIO2_REPO)
    from stepaudio2vllm import StepAudio2 as StepAudio2VLLM
    from token2wav import Token2wav
except Exception:
    print("[Warning] Do not load Step-Audio2")

STEP_PROMPT_VOICE = (
    os.environ.get("STEP_AUDIO2_PROMPT_VOICE")
    or (os.path.join(STEP_AUDIO2_REPO, "assets", "default_female.wav") if STEP_AUDIO2_REPO else "")
)

# Step-Audio2 audio token vocabulary upper bound. Tokens >= 6561 are control tokens
# that token2wav cannot synthesize, so we strip them before decoding.
_AUDIO_TOKEN_MAX = 6561

# Sentinel string the model expects as the assistant prefix to start audio generation.
_TTS_START_TOKEN = "<tts_start>"

RECALL_MODELS = {"step-audio2-mini-recall"}

RECALL_PROMPT = (
    "What specific instructions did the user give in the first turn "
    "regarding your speaking style for this conversation?"
)

MODEL_KEYS = [
    "step-audio2-mini",
    "step-audio2-mini-recall",
]

_step_backends = {}


def _get_step_vllm_backend(model_name):
    if model_name not in _step_backends:
        model = StepAudio2VLLM(VLLM_URL, model_name.lower())
        token2wav_path = os.path.join(STEP_AUDIO2_REPO, model_name, "token2wav")
        t2w = Token2wav(token2wav_path)
        _step_backends[model_name] = (model, t2w)
    return _step_backends[model_name]


def stepaudio2_vllm_call(history, output_audio=True):
    step_model, token2wav = _get_step_vllm_backend(STEP_MODEL_NAME)
    response, text, audio_tokens = step_model(history, max_tokens=8192, temperature=1)

    wav_bytes = b""
    if audio_tokens is not None and output_audio:
        audio_tokens = [x for x in audio_tokens if x < _AUDIO_TOKEN_MAX]
        wav_bytes = token2wav(audio_tokens, prompt_wav=STEP_PROMPT_VOICE)
    return wav_bytes, text.strip(), response


def init_history_fn(instruction, instruction_location, eval_model):
    message = default_system_message(instruction, instruction_location)
    return None, [{"role": "system", "content": message}]


def _do_recall(self_history):
    """Append a recall Q/A pair to elicit a restatement of the original style instruction."""
    self_history.append({
        "role": "human",
        "content": [{"type": "text", "text": RECALL_PROMPT}],
    })
    # Step-Audio2 requires a placeholder assistant turn to prompt the response.
    self_history.append({"role": "assistant", "content": None})
    _, text, _ = stepaudio2_vllm_call(self_history, output_audio=False)
    self_history.pop()
    self_history.append({"role": "assistant", "content": text})


def call_model_fn(
    self_history,
    turn_index,
    instruction_location,
    instruction,
    new_audio_path=None,
    new_text=None,
    save_audio_path=None,
    system_message=None,
    state=None,
    eval_model=None,
):
    if turn_index != 1 and eval_model in RECALL_MODELS:
        _do_recall(self_history)

    if new_audio_path is not None:
        audio_part = {"type": "audio", "audio": str(new_audio_path)}
        if instruction_location == "user" and system_message is not None:
            content = [{"type": "text", "text": system_message}, audio_part]
        else:
            content = [audio_part]
        self_history.append({"role": "human", "content": content})
    elif new_text is not None:
        self_history.append({
            "role": "human",
            "content": [{"type": "text", "text": new_text}],
        })

    self_history.append({"role": "assistant", "content": _TTS_START_TOKEN, "eot": False})
    wav_bytes, text, response = stepaudio2_vllm_call(self_history)
    self_history.pop()
    self_history.append({"role": "assistant", "tts_content": response.get("tts_content", {})})

    save_path = save_audio(wav_bytes, save_audio_path)
    return save_path, text, self_history, state


def main():
    parser = parse_base_args()
    parser.add_argument("--model", type=str, default="step-audio2-mini", choices=MODEL_KEYS)
    args = parser.parse_args()

    fix_random_seed(args.seed)

    eval_model = args.model
    instructions = apply_instruction_index(args, load_instructions(args))

    args.output_dir = resolve_output_dir(args, eval_model)
    chat_opener_audios, all_topics = load_topics(args)

    run_batch(
        chat_opener_audios=chat_opener_audios,
        all_topics=all_topics,
        system_messages=instructions,
        output_dir=args.output_dir,
        eval_model=eval_model,
        call_model_fn=lambda **kwargs: call_model_fn(eval_model=eval_model, **kwargs),
        init_history_fn=init_history_fn,
        max_turns=args.max_turns,
        instruction_location=args.instruction_location,
        instruction_index=args.instruction_index,
    )


if __name__ == "__main__":
    main()
