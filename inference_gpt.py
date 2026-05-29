from eval_common import (
    apply_instruction_index,
    default_system_message,
    load_instructions,
    load_topics,
    openai_gpt_call,
    parse_base_args,
    resolve_output_dir,
    run_batch,
)
from inference_utils import fix_random_seed, save_audio, wav_file_to_base64


MODEL_MAP = {
    "gpt-4o-mini": "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-mini-recall": "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o": "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-recall": "gpt-4o-audio-preview-2025-06-03",
}

RECALL_MODELS = {"gpt-4o-mini-recall", "gpt-4o-recall"}

RECALL_PROMPT = (
    "What specific instructions did the user give in the first turn "
    "regarding your speaking style for this conversation?"
)


def init_history_fn(instruction, instruction_location, eval_model):
    message = default_system_message(instruction, instruction_location)
    return None, [{"role": "system", "content": message}]


def _build_user_audio_content(audio_path, system_message=None, instruction_location="user"):
    audio_part = {
        "type": "input_audio",
        "input_audio": {
            "data": wav_file_to_base64(str(audio_path)),
            "format": "wav",
        },
    }
    if instruction_location == "user" and system_message is not None:
        return [{"type": "text", "text": system_message}, audio_part]
    return [audio_part]


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
    model_id = MODEL_MAP[eval_model]

    # Recall: ask the model to restate the original style instruction.
    if turn_index != 1 and eval_model in RECALL_MODELS:
        self_history.append(
            {"role": "user", "content": [{"type": "text", "text": RECALL_PROMPT}]}
        )
        _, recall_text = openai_gpt_call(model_id, self_history, output_audio=False)
        self_history.append({"role": "assistant", "content": recall_text})

    if new_audio_path is not None:
        self_history.append({
            "role": "user",
            "content": _build_user_audio_content(new_audio_path, system_message, instruction_location),
        })
    elif new_text is not None:
        self_history.append({"role": "user", "content": new_text})

    wav_bytes, text = openai_gpt_call(model_id, self_history)
    self_history.append({"role": "assistant", "content": text})
    save_path = save_audio(wav_bytes, save_audio_path)

    return save_path, text, self_history, state


def main():
    parser = parse_base_args()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=sorted(MODEL_MAP.keys()),
    )
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
