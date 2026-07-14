"""Ties emotion.py and tts.py together: classify emotions, turn them into
pause/volume cues, synthesize speech, stitch it into one file.

Piper is a single fixed-identity voice -- pitch/speed changes make it sound
like a different person, so pace is fixed and only pause/volume vary."""

from io import BytesIO
from typing import TypedDict

from pydub import AudioSegment

import emotion
import tts
from emotion import EmotionResult


# ----------------------------------------------------------------------
# Emotion profiles: family -> baseline pause (ms) and volume (dB change)
# ----------------------------------------------------------------------

class EmotionProfile(TypedDict):
    pause: int
    volume: float


# Fallback for an unrecognized family, and the base for "neutral".
DEFAULT_PROFILE: EmotionProfile = {"pause": 220, "volume": 0}

EMOTION_PROFILE: dict[str, EmotionProfile] = {
    "sadness": {"pause": 650, "volume": -2},
    "fear": {"pause": 450, "volume": -1},
    "anger": {"pause": 120, "volume": 1.5},
    "joy": {"pause": 120, "volume": 1},
    "surprise": {"pause": 200, "volume": 1.5},
    "disgust": {"pause": 300, "volume": -1},
    "neutral": dict(DEFAULT_PROFILE),
}

# Confidence/length/punctuation scaling below can compound, so clamp the
# final pause to a sane range.
_MIN_PAUSE_MS = 80
_MAX_PAUSE_MS = 900

# Tiny fades to avoid an audible click at segment boundaries.
_FADE_IN_MS = 8
_FADE_OUT_MS = 15

# Safety net in case a future tuning pass adds too large a gain.
_MIN_VOLUME_DB = -3
_MAX_VOLUME_DB = 3


def _get_profile(family: str) -> EmotionProfile:
    return EMOTION_PROFILE.get(family, DEFAULT_PROFILE)


def _blend_profile(item: EmotionResult) -> EmotionProfile:
    """Blend pause/volume across the top-N detected emotions (item["emotions"]),
    weighted by their scores, instead of committing fully to just the single
    top label's family. GoEmotions is multi-label (see emotion.py), so a
    sentence scoring e.g. curiosity 0.75 + sadness 0.43 is genuinely a mix --
    this gives it an in-between pause/volume rather than a hard jump to one
    category. Uses the *unsmoothed* per-sentence scores (item["emotions"]
    isn't touched by smooth_emotions), while "family" (smoothed) still picks
    which TTS voice settings to use -- see _build_audio."""
    weighted = item.get("emotions") or [(item["emotion"], item["confidence"])]
    total = sum(score for _, score in weighted) or 1.0

    pause = 0.0
    volume = 0.0
    for label, score in weighted:
        profile = _get_profile(emotion.to_family(label))
        weight = score / total
        pause += profile["pause"] * weight
        volume += profile["volume"] * weight

    return {"pause": pause, "volume": volume}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _confidence_scale(confidence: float) -> float:
    """confidence=0.0 -> 0.70x, confidence=1.0 -> 1.30x pause length."""
    confidence = max(0.0, min(confidence, 1.0))
    return 0.7 + (confidence * 0.6)


def _sentence_scale(text: str) -> float:
    """Longer sentences get a slightly longer pause after them."""
    words = len(text.split())
    if words <= 5:
        return 0.75
    if words >= 25:
        return 1.20
    return 1.0


def _punctuation_pause(text: str) -> int:
    """Extra pause (ms) based on sentence-ending punctuation. Strips
    trailing quotes first so 'she said "Stop!"' still matches "!"."""
    text = text.rstrip(' \t\n\r"\'')
    if text.endswith("..."):
        return 300
    if text.endswith("?"):
        return 120
    if text.endswith("!"):
        return 40
    return 0


def _compute_pause(item: EmotionResult, profile: EmotionProfile) -> int:
    # None-check instead of `.get("confidence") or 0.5` -- `or` would also
    # override a legitimate confidence of 0.0.
    confidence = item.get("confidence")
    confidence = 0.5 if confidence is None else float(confidence)

    pause = profile["pause"]
    pause *= _confidence_scale(confidence)
    pause *= _sentence_scale(item["text"])
    pause += _punctuation_pause(item["text"])

    return max(_MIN_PAUSE_MS, min(int(pause), _MAX_PAUSE_MS))


def _apply_volume(segment: AudioSegment, profile: EmotionProfile) -> AudioSegment:
    volume = max(_MIN_VOLUME_DB, min(profile["volume"], _MAX_VOLUME_DB))
    return segment + volume  # pydub overloads "+" as a dB gain change


# ----------------------------------------------------------------------
# Audio assembly
# ----------------------------------------------------------------------

# Small micro-optimization: reuse silent segments instead of reallocating
# the same duration repeatedly.
_silence_cache: dict[int, AudioSegment] = {}


def _silence(ms: int) -> AudioSegment:
    if ms not in _silence_cache:
        _silence_cache[ms] = AudioSegment.silent(duration=ms)
    return _silence_cache[ms]


def _build_audio(tagged_sentences: list[EmotionResult]) -> AudioSegment:
    combined = AudioSegment.empty()

    for item in tagged_sentences:
        family = item["family"]
        profile = _blend_profile(item)

        wav = tts.synthesize(item["text"], family)
        segment = AudioSegment.from_wav(BytesIO(wav))
        segment = segment.fade_in(_FADE_IN_MS).fade_out(_FADE_OUT_MS)
        segment = _apply_volume(segment, profile)

        combined += segment
        combined += _silence(_compute_pause(item, profile))

    return combined


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def narrate(text: str) -> bytes:
    """Convert text into narrated WAV audio."""
    tagged = emotion.analyze(text)
    audio = _build_audio(tagged)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


def narrate_with_report(text: str) -> tuple[bytes, list[EmotionResult]]:
    """Same as narrate(), but also returns emotion metadata."""
    tagged = emotion.analyze(text)
    audio = _build_audio(tagged)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue(), tagged
