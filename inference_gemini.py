import asyncio
import io
import json
import os
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from eval_common import (
    DEFAULT_SYSTEM_PROMPT_EN,
    apply_instruction_index,
    call_cascaded_model,
    load_instructions,
    load_topics,
    parse_base_args,
    resolve_output_dir,
)
from inference_utils import (
    concat_audios_pcm16,
    fix_random_seed,
    load_as_pcm16,
    naming,
    save_audio,
    strip_base64_from_history,
)


MODEL_MAP = {
    "gemini-live": "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-live-recall": "gemini-2.5-flash-native-audio-preview-09-2025",
}

RECALL_MODELS = {"gemini-live-recall"}

RECALL_PROMPT = (
    "What specific instructions did the user give in the first turn "
    "regarding your speaking style for this conversation?"
)

# Gemini Live emits 24kHz PCM audio frames.
_GEMINI_OUTPUT_SR = 24000
# Minimum silence duration (20ms at 24kHz, 16-bit mono) used when the model returns
# no audio, so downstream code never has to handle a zero-byte WAV.
_MIN_SILENCE_BYTES = int(_GEMINI_OUTPUT_SR * 0.02 * 2)


_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _pcm_to_wav_bytes(pcm_bytes, sample_rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def gemini_live_call(session, audio_path, instruction_audio_path=None):
    """Send one user audio turn to Gemini Live, return (wav_bytes, transcript_text)."""
    if instruction_audio_path is not None:
        pcm16_16k = concat_audios_pcm16(audio_path, instruction_audio_path)
    else:
        pcm16_16k = load_as_pcm16(audio_path)

    await session.send_realtime_input(
        audio=types.Blob(data=pcm16_16k, mime_type="audio/pcm;rate=16000")
    )
    await session.send_realtime_input(audio_stream_end=True)

    audio_chunks = []
    text_chunks = []
    async for resp in session.receive():
        if getattr(resp, "data", None):
            audio_chunks.append(resp.data)

        sc = getattr(resp, "server_content", None)
        if sc:
            tr_out = getattr(sc, "output_transcription", None)
            if tr_out and getattr(tr_out, "text", None):
                text_chunks.append(tr_out.text)
            if getattr(sc, "turn_complete", False):
                break

    raw_pcm = b"".join(audio_chunks) if audio_chunks else b"\x00" * _MIN_SILENCE_BYTES
    return _pcm_to_wav_bytes(raw_pcm, _GEMINI_OUTPUT_SR), "".join(text_chunks).strip()


def _turn_path(output_dir, topic_id, par_id, tag, turn, ext):
    return os.path.join(output_dir, f"topic{topic_id}_par{par_id}_{tag}_turn{turn}.{ext}")


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


def _save_assistant_turn(output_dir, topic_id, par_id, tag, turn, wav_bytes, text, transcript, speaker_label):
    save_path = save_audio(wav_bytes, _turn_path(output_dir, topic_id, par_id, tag, turn, "wav"))
    _write_text(_turn_path(output_dir, topic_id, par_id, tag, turn, "txt"), text)
    transcript.append({"speaker": speaker_label, "text": text})
    return save_path


async def run_multiturn_discussion_async(
    chat_opener_audio,
    chat_opener_text,
    paraphrasing_id,
    output_dir,
    instruction,
    chatbot_system_prompt,
    eval_model,
    max_turns=4,
    instruction_location="user",
):
    os.makedirs(output_dir, exist_ok=True)
    transcript = []
    topic_id = os.path.basename(chat_opener_audio).split("_")[0]

    assistant_key = eval_model
    user_key = "cascaded"
    assistant_tag = naming(assistant_key, "assistant")
    user_tag = naming(user_key, "user")

    # Skip already-finished runs.
    if os.path.exists(_turn_path(output_dir, topic_id, paraphrasing_id, assistant_tag, max_turns, "wav")):
        return

    model_id = MODEL_MAP[assistant_key]
    config = {
        "response_modalities": ["AUDIO"],
        "realtime_input_config": {
            "automatic_activity_detection": {"disabled": False, "silence_duration_ms": 3000}
        },
        "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
        "output_audio_transcription": {},
    }

    assistant_history = []
    user_history = [
        {"role": "user", "content": chatbot_system_prompt},
        {"role": "assistant", "content": chat_opener_text},
    ]

    transcript.append({"speaker": "Trigger", "text": chatbot_system_prompt})
    transcript.append({"speaker": f"{user_key} (user)", "text": chat_opener_text})

    # When instruction_location == "user", `instruction` is the path to an instruction
    # audio clip that gets appended after the chat opener. Otherwise it is text.
    instruction_audio_for_turn1 = instruction if instruction_location == "user" else None
    assistant_speaker_label = f"{assistant_key} (assistant)"

    gemini_client = get_gemini_client()
    async with gemini_client.aio.live.connect(model=model_id, config=config) as session:
        # Turn 1: chat opener -> assistant.
        wav_bytes, assistant_text = await gemini_live_call(
            session,
            audio_path=chat_opener_audio,
            instruction_audio_path=instruction_audio_for_turn1,
        )
        _save_assistant_turn(
            output_dir, topic_id, paraphrasing_id, assistant_tag, 1,
            wav_bytes, assistant_text, transcript, assistant_speaker_label,
        )

        opener_content = [{"type": "audio", "audio": str(chat_opener_audio)}]
        if instruction_location == "user":
            opener_content.append({"type": "text", "text": instruction})
        assistant_history.append({"role": "user", "content": opener_content})
        assistant_history.append({
            "role": "assistant",
            "content": assistant_text or "<audio_only>",
        })
        user_history.append({"role": "user", "content": assistant_text})

        # Turns 2..max_turns: alternate user (cascaded) -> assistant.
        for turn in range(2, max_turns + 1):
            if assistant_key in RECALL_MODELS:
                assistant_history.append({"role": "user", "content": RECALL_PROMPT})

            user_audio_path, user_text, user_history = call_cascaded_model(
                self_history=user_history,
                new_text=assistant_text,
                save_audio_path=_turn_path(
                    output_dir, topic_id, paraphrasing_id, user_tag, turn, "wav"
                ),
            )
            _write_text(
                _turn_path(output_dir, topic_id, paraphrasing_id, user_tag, turn, "txt"),
                user_text,
            )
            transcript.append({"speaker": f"{user_key} (user)", "text": user_text})

            wav_bytes, assistant_text = await gemini_live_call(session, audio_path=user_audio_path)
            _save_assistant_turn(
                output_dir, topic_id, paraphrasing_id, assistant_tag, turn,
                wav_bytes, assistant_text, transcript, assistant_speaker_label,
            )

            if user_text:
                assistant_history.append({"role": "user", "content": user_text})
            elif user_audio_path:
                assistant_history.append({
                    "role": "user",
                    "content": [{"type": "audio", "audio": str(user_audio_path)}],
                })
            assistant_history.append({
                "role": "assistant",
                "content": assistant_text or "<audio_only>",
            })
            user_history.append({"role": "user", "content": assistant_text})

            _dump_history(
                os.path.join(output_dir, f"topic{topic_id}_par{paraphrasing_id}_assistant_history.json"),
                assistant_history,
            )
            _dump_history(
                os.path.join(output_dir, f"topic{topic_id}_par{paraphrasing_id}_user_history.json"),
                user_history,
            )

    with open(
        os.path.join(output_dir, f"topic{topic_id}_par{paraphrasing_id}_transcript.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


async def run_batch_async(
    chat_opener_audios,
    all_topics,
    system_messages,
    output_dir,
    eval_model,
    max_turns=4,
    instruction_location="system",
    instruction_index=None,
):
    os.makedirs(output_dir, exist_ok=True)

    for sample_idx, chat_opener_audio in enumerate(tqdm(chat_opener_audios)):
        for key, sys_msg_list in system_messages.items():
            out_dir = os.path.join(output_dir, key)
            for idx, sys_msg in enumerate(sys_msg_list):
                if instruction_index is not None:
                    assert len(sys_msg_list) == 1
                    idx = instruction_index

                while True:
                    try:
                        await run_multiturn_discussion_async(
                            chat_opener_audio=chat_opener_audio,
                            chat_opener_text=all_topics[sample_idx],
                            paraphrasing_id=idx,
                            output_dir=out_dir,
                            instruction=sys_msg,
                            chatbot_system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
                            eval_model=eval_model,
                            max_turns=max_turns,
                            instruction_location=instruction_location,
                        )
                        break
                    except Exception as e:
                        print(f"[{key}] run_multiturn_discussion_async failed, retrying... error: {e}")
                        await asyncio.sleep(30)
                await asyncio.sleep(5)


def main():
    parser = parse_base_args()
    parser.add_argument("--model", type=str, default="gemini-live", choices=MODEL_MAP.keys())
    args = parser.parse_args()

    fix_random_seed(args.seed)

    instructions = load_instructions(args)
    if args.instruction_location == "user":
        # Swap text instructions for the corresponding instruction-audio file paths.
        instructions = {
            key: [
                os.path.join(args.instruction_audio_dir, f"{key}_{idx}.wav")
                for idx in range(len(paraphrasing_list))
            ]
            for key, paraphrasing_list in instructions.items()
        }
    instructions = apply_instruction_index(args, instructions)

    eval_model = args.model
    args.output_dir = resolve_output_dir(args, eval_model)
    chat_opener_audios, all_topics = load_topics(args)

    asyncio.run(
        run_batch_async(
            chat_opener_audios=chat_opener_audios,
            all_topics=all_topics,
            system_messages=instructions,
            output_dir=args.output_dir,
            eval_model=eval_model,
            max_turns=args.max_turns,
            instruction_location=args.instruction_location,
            instruction_index=args.instruction_index,
        )
    )


if __name__ == "__main__":
    main()
