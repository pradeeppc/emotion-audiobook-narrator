"""Text + emotion -> audio bytes (WAV), backed by Piper (CPU, fixed voice).

Exposes one function, `synthesize`, so swapping the TTS engine later only
touches this file."""

import io
import wave
from pathlib import Path

from piper.voice import PiperVoice
from piper.config import SynthesisConfig

_MODEL_PATH = Path(__file__).parent / "models" / "en_US-lessac-medium.onnx"

_voice = None

# length_scale (speed) is fixed across all emotions -- varying it made the
# narrator sound like a different person. Only noise_scale/noise_w_scale
# (subtle texture) vary here; volume (in pipeline.py) carries most of the
# emotional signal.
_FIXED_LENGTH_SCALE = 1.0

_EMOTION_PARAMS = {
    "sadness": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.5, noise_w_scale=0.6),
    "fear": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.8, noise_w_scale=0.9),
    "anger": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.8, noise_w_scale=0.85),
    "joy": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.72, noise_w_scale=0.75),
    "surprise": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.8, noise_w_scale=0.85),
    "disgust": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.6, noise_w_scale=0.65),
    "neutral": dict(length_scale=_FIXED_LENGTH_SCALE, noise_scale=0.667, noise_w_scale=0.8),
}


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(_MODEL_PATH)
    return _voice


def synthesize(text: str, emotion: str = "neutral") -> bytes:
    """Convert one sentence into WAV audio bytes for the given emotion."""
    voice = _get_voice()
    params = _EMOTION_PARAMS.get(emotion, _EMOTION_PARAMS["neutral"])
    syn_config = SynthesisConfig(**params)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    return buffer.getvalue()
