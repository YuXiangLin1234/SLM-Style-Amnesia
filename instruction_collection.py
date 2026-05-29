INSTRUCTIONS = {
    "en_north_america": [
        "From now on, you can only speak in General American English.",
        "In the following conversation, you must consistently speak in a General American accent.",
        "You are a person who always speaks in General American English, no matter what.",
    ],
    "en_indian": [
        "From now on, you can only speak in Indian English.",
        "In the following conversation, you must consistently speak with an Indian English accent.",
        "You are a person who always speaks in Indian English, no matter what.",
    ],
    "en_anger": [
        "From now on, always speak with an angry tone.",
        "In the following conversation, you must consistently speak with an angry tone.",
        "You are a person who always speaks with an angry tone, no matter what.",
    ],
    "en_happiness": [
        "From now on, always speak with a happy tone.",
        "In the following conversation, you must consistently speak with a happy tone.",
        "You are a person who always speaks with a happy tone, no matter what.",
    ],
    "en_neutral": [
        "From now on, always speak in a neutral tone.",
        "In the following conversation, you must consistently speak in a neutral tone.",
        "You are a person who always speaks in a neutral tone, no matter what.",
    ],
    "en_sadness": [
        "From now on, always speak with a sad tone.",
        "In the following conversation, you must consistently speak with a sad tone.",
        "You are a person who always speaks with a sad tone, no matter what.",
    ],

    "en_fast": [
        "From now on, you can only speak fast.",
        "In the following conversation, you must consistently speak fast.",
        "You are a person who always speaks fast, no matter what.",
    ],
    "en_slow": [
        "From now on, you can only speak slowly.",
        "In the following conversation, you must consistently speak slowly.",
        "You are a person who always speaks slowly, no matter what.",
    ],

    "en_loud": [
        "From now on, you can only speak loudly.",
        "In the following conversation, you must consistently speak loudly.",
        "You are a person who always speaks loudly, no matter what.",
    ],
    "en_quiet": [
        "From now on, you can only speak quietly.",
        "In the following conversation, you must consistently speak quietly.",
        "You are a person who always speaks quietly, no matter what.",
    ],
}

# Style keys grouped by evaluation dimension. Used by save_heuristic_metric.py and
# tts_baseline_*.py to decide which DSP metric / neutral TTS prompt applies to a
# given result directory.
EMOTION_KEY = {"en_anger", "en_happiness", "en_neutral", "en_sadness"}
ACCENT_KEY = {"en_north_america", "en_indian"}
SPEED_KEY = {"en_fast", "en_slow"}
VOLUME_KEY = {"en_loud", "en_quiet"}
