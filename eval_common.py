"""Shared helpers for the inference scripts.

Each `inference_<model>.py` script wires a model-specific `call_model_fn` and
`init_history_fn` into `run_batch`, which iterates topics x instructions x
paraphrasings and runs `run_multiturn_discussion`.
"""
import argparse
import base64
import json
import os
import time

import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

from inference_utils import (
    INPUT_SAMPLING_RATE,
    fix_random_seed,
    list_en_wavs,
    naming,
    resample_audio,
    strip_base64_from_history,
)


DEFAULT_SYSTEM_PROMPT_EN = (
    "You are a chatbot. Please start a conversation by opening a new topic. "
    "Chat casually and feel free to role-play in different scenarios. "
    "If the conversation stalls, you can extend the topic. "
    "Speak within 20 English words each round. "
    "This is spoken dialogue, so do not use words or expressions "
    "that cannot be naturally spoken aloud."
)

# Models used by `call_cascaded_model` for the user-side (cascaded LM + TTS).
CASCADED_LM_MODEL = "gpt-5-mini-2025-08-07"
CASCADED_TTS_MODEL = "gpt-4o-mini-tts"

# Filename role tags. "A" = the model under test, "B" = the cascaded user.
# Kept as single letters so existing runs on disk stay compatible.
ROLE_TAG_ASSISTANT = "A"
ROLE_TAG_USER = "B"

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def tts_call(model, text, instruction, save_path, voice="alloy"):
    client = get_openai_client()
    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
    }
    if instruction is not None:
        kwargs["instructions"] = instruction

    with client.audio.speech.with_streaming_response.create(**kwargs) as resp:
        resp.stream_to_file(save_path)

    wav, sr = sf.read(save_path, dtype="float32")
    wav, sr = resample_audio(wav, sr, INPUT_SAMPLING_RATE)
    sf.write(save_path, wav, sr)


def cascaded_call(
    language_model_id,
    tts_model_id,
    history,
    tts_instruction,
    save_path,
    max_retries=5,
    retry_delay=3,
):
    """Run LM completion then TTS. Returns the LM text, writes audio to `save_path`."""
    response = None
    client = get_openai_client()

    for _attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=language_model_id,
                messages=history,
                temperature=1,
                max_completion_tokens=8192,
            )
            content = resp.choices[0].message.content
            if content:
                response = content.strip()
                break
        except Exception as e:
            print(e)
        time.sleep(retry_delay)

    tts_call(tts_model_id, response, tts_instruction, save_path)
    return response


def openai_gpt_call(model_id, history, max_retries=1, retry_delay=10, output_audio=True):
    """Call a GPT-4o-audio model. Returns (wav_bytes, transcript_text)."""
    output_modalities = ["text", "audio"] if output_audio else ["text"]
    client = get_openai_client()

    last_audio_obj = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                modalities=output_modalities,
                messages=history,
                seed=attempt + 6000,
                temperature=1,
                audio={"voice": "alloy", "format": "wav"},
                max_completion_tokens=16384,
            )
            msg = resp.choices[0].message
            if output_audio:
                audio_obj = getattr(msg, "audio", None)
                last_audio_obj = audio_obj
                has_data = bool(audio_obj and getattr(audio_obj, "data", None))
                has_text = bool(audio_obj and getattr(audio_obj, "transcript", None))
                if has_data and has_text:
                    return base64.b64decode(audio_obj.data), audio_obj.transcript
                if audio_obj is None:
                    raise ValueError("No audio object in the response. Retrying...")
            elif msg.content:
                return None, msg.content.strip()
        except Exception as e:
            print(f"[Retry] {attempt + 1} / seed {attempt} {e}")

        if attempt < max_retries:
            time.sleep(retry_delay)

    wav_bytes = (
        base64.b64decode(last_audio_obj.data)
        if (last_audio_obj and getattr(last_audio_obj, "data", None))
        else b""
    )
    text = getattr(last_audio_obj, "transcript", "") if last_audio_obj else ""
    return wav_bytes, text


def default_system_message(instruction, instruction_location):
    if instruction_location == "system":
        return f"You are a helpful assistant.\n{instruction}"
    return "You are a helpful assistant."


def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_dir", default="data/topic")
    parser.add_argument("--instruction_audio_dir", default="data/instruction_audios")
    parser.add_argument("--topic_txt_path", default="data/topic/topics.txt")
    parser.add_argument(
        "--instruction_location",
        type=str,
        default="user",
        choices=["system", "user"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topic_start_index", type=int, default=None)
    parser.add_argument("--topic_end_index", type=int, default=None)
    parser.add_argument("--instruction_index", type=int, default=None)
    return parser


def resolve_output_dir(args, eval_model):
    if args.output_dir is not None:
        return args.output_dir
    return os.path.join("results", eval_model, args.instruction_location)


def load_topics(args):
    chat_opener_audios = list_en_wavs(args.topic_dir)
    if args.topic_end_index is not None:
        chat_opener_audios = chat_opener_audios[: args.topic_end_index]
    with open(args.topic_txt_path, "r") as f:
        all_topics = f.readlines()
    return chat_opener_audios, all_topics


def load_instructions(args):
    from instruction_collection import INSTRUCTIONS
    return INSTRUCTIONS


def apply_instruction_index(args, instructions):
    if args.instruction_index is not None:
        instructions = {
            key: [value[args.instruction_index]] for key, value in instructions.items()
        }
    return instructions


def call_cascaded_model(self_history, new_text, save_audio_path, system_message=None):
    if new_text is not None:
        text = new_text if system_message is None else f"{new_text} {system_message}"
        self_history.append({"role": "user", "content": [{"type": "text", "text": text}]})

    text = cascaded_call(
        CASCADED_LM_MODEL,
        CASCADED_TTS_MODEL,
        self_history,
        None,
        save_audio_path,
        max_retries=5,
        retry_delay=3,
    )
    self_history.append({"role": "assistant", "content": text})
    return save_audio_path, text, self_history


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _dump_history(path, history):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            strip_base64_from_history(history),
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def _dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _turn_path(output_dir, topic_id, paraphrasing_id, tag, turn, ext):
    return os.path.join(
        output_dir, f"topic{topic_id}_par{paraphrasing_id}_{tag}_turn{turn}.{ext}"
    )


def run_multiturn_discussion(
    chat_opener_audio,
    chat_opener_text,
    paraphrasing_id,
    output_dir,
    instruction,
    chatbot_system_prompt,
    eval_model,
    call_model_fn,
    init_history_fn,
    max_turns=4,
    instruction_location="system",
):
    os.makedirs(output_dir, exist_ok=True)
    transcript = []
    topic_id = os.path.basename(chat_opener_audio).split("_")[0]

    assistant_key = eval_model
    user_key = "cascaded"
    assistant_tag = naming(assistant_key, ROLE_TAG_ASSISTANT)
    user_tag = naming(user_key, ROLE_TAG_USER)

    model_state, assistant_history = init_history_fn(
        instruction=instruction,
        instruction_location=instruction_location,
        eval_model=eval_model,
    )

    user_history = [
        {"role": "user", "content": chatbot_system_prompt},
        {"role": "assistant", "content": chat_opener_text},
    ]

    transcript.append({"speaker": "Trigger", "text": chatbot_system_prompt})
    transcript.append({"speaker": f"{user_key} ({ROLE_TAG_USER})", "text": chat_opener_text})

    # Skip work that is already complete.
    if os.path.exists(_turn_path(output_dir, topic_id, paraphrasing_id, assistant_tag, max_turns, "wav")):
        return

    # Turn 1: assistant responds to the chat opener.
    _, assistant_text, assistant_history, model_state = call_model_fn(
        self_history=assistant_history,
        turn_index=1,
        instruction_location=instruction_location,
        instruction=instruction,
        new_audio_path=chat_opener_audio,
        new_text=chat_opener_text,
        save_audio_path=_turn_path(output_dir, topic_id, paraphrasing_id, assistant_tag, 1, "wav"),
        system_message=instruction,
        state=model_state,
    )
    _write_text(
        _turn_path(output_dir, topic_id, paraphrasing_id, assistant_tag, 1, "txt"),
        assistant_text,
    )
    transcript.append({"speaker": f"{assistant_key} ({ROLE_TAG_ASSISTANT})", "text": assistant_text})

    # Turns 2..max_turns: alternate user (cascaded) -> assistant.
    for turn in range(2, max_turns + 1):
        user_audio_path, user_text, user_history = call_cascaded_model(
            self_history=user_history,
            new_text=assistant_text,
            save_audio_path=_turn_path(output_dir, topic_id, paraphrasing_id, user_tag, turn, "wav"),
        )
        _write_text(
            _turn_path(output_dir, topic_id, paraphrasing_id, user_tag, turn, "txt"),
            user_text,
        )
        transcript.append({"speaker": f"{user_key} ({ROLE_TAG_USER})", "text": user_text})

        _, assistant_text, assistant_history, model_state = call_model_fn(
            self_history=assistant_history,
            turn_index=turn,
            instruction_location=instruction_location,
            instruction=instruction,
            new_audio_path=user_audio_path,
            new_text=user_text,
            save_audio_path=_turn_path(output_dir, topic_id, paraphrasing_id, assistant_tag, turn, "wav"),
            system_message=None,
            state=model_state,
        )
        _write_text(
            _turn_path(output_dir, topic_id, paraphrasing_id, assistant_tag, turn, "txt"),
            assistant_text,
        )
        transcript.append({"speaker": f"{assistant_key} ({ROLE_TAG_ASSISTANT})", "text": assistant_text})

        _dump_history(
            os.path.join(output_dir, f"topic{topic_id}_par{paraphrasing_id}_A_history.json"),
            assistant_history,
        )
        _dump_history(
            os.path.join(output_dir, f"topic{topic_id}_par{paraphrasing_id}_B_history.json"),
            user_history,
        )

    _dump_json(
        os.path.join(output_dir, f"topic{topic_id}_par{paraphrasing_id}_transcript.json"),
        transcript,
    )


def run_batch(
    chat_opener_audios,
    all_topics,
    system_messages,
    output_dir,
    eval_model,
    call_model_fn,
    init_history_fn,
    max_turns=4,
    instruction_location="system",
    instruction_index=None,
):
    os.makedirs(output_dir, exist_ok=True)

    for sample_idx, chat_opener_audio in enumerate(chat_opener_audios):
        for key, sys_msg_list in system_messages.items():
            out_dir = os.path.join(output_dir, key)
            for idx, sys_msg in enumerate(sys_msg_list):
                if instruction_index is not None:
                    assert len(sys_msg_list) == 1
                    idx = instruction_index

                run_multiturn_discussion(
                    chat_opener_audio=chat_opener_audio,
                    chat_opener_text=all_topics[sample_idx],
                    paraphrasing_id=idx,
                    output_dir=out_dir,
                    instruction=sys_msg,
                    chatbot_system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
                    eval_model=eval_model,
                    call_model_fn=call_model_fn,
                    init_history_fn=init_history_fn,
                    max_turns=max_turns,
                    instruction_location=instruction_location,
                )


def run_base_cli(eval_model, call_model_fn, init_history_fn, extra_args=None):
    parser = parse_base_args()
    if extra_args:
        extra_args(parser)
    args = parser.parse_args()

    fix_random_seed(args.seed)

    instructions = load_instructions(args)
    instructions = apply_instruction_index(args, instructions)

    args.output_dir = resolve_output_dir(args, eval_model)
    chat_opener_audios, all_topics = load_topics(args)

    run_batch(
        chat_opener_audios=chat_opener_audios,
        all_topics=all_topics,
        system_messages=instructions,
        output_dir=args.output_dir,
        eval_model=eval_model,
        call_model_fn=call_model_fn,
        init_history_fn=init_history_fn,
        max_turns=args.max_turns,
        instruction_location=args.instruction_location,
        instruction_index=args.instruction_index,
    )
    return args
