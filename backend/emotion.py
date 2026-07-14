"""Split text into sentences, classify the emotion of each, smooth the results."""

from typing import TypedDict

import nltk
from transformers import pipeline


class EmotionResult(TypedDict):
    """Shape returned by classify_sentences/smooth_emotions/analyze."""

    text: str
    emotion: str
    family: str
    confidence: float
    emotions: list[tuple[str, float]]

# 28 fine-grained emotions (GoEmotions dataset), multi-label (scores don't
# sum to 1 -- a sentence can be both "joy" and "gratitude").
_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# tts.py/pipeline.py only handle 7 broad prosody families, so we collapse
# the 28 fine-grained labels down to those for anything audio-related.
_FAMILY_MAP = {
    "admiration": "joy", "amusement": "joy", "approval": "joy", "caring": "joy",
    "desire": "joy", "excitement": "joy", "gratitude": "joy", "joy": "joy",
    "love": "joy", "optimism": "joy", "pride": "joy", "relief": "joy",
    "disappointment": "sadness", "grief": "sadness", "remorse": "sadness",
    "sadness": "sadness", "embarrassment": "sadness",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "fear": "fear", "nervousness": "fear",
    "disgust": "disgust",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    "curiosity": "surprise",
    "neutral": "neutral",
}

# Loaded once per process (loading a model is slow; reusing it is fast).
_classifier = None

nltk.download("punkt_tab", quiet=True)


def to_family(emotion_label: str) -> str:
    """Collapse a fine-grained GoEmotions label into one of 7 prosody families."""
    return _FAMILY_MAP.get(emotion_label, "neutral")


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            top_k=None,  # return scores for every emotion label, not just the top one
            device="cpu",  # force CPU explicitly -- on HF's ZeroGPU hardware,
            # torch.cuda.is_available() reports True even outside a
            # @spaces.GPU-decorated function, so auto-detection would try
            # (and fail) to move the model to CUDA here.
        )
    return _classifier


def split_sentences(text: str) -> list[str]:
    """Break a block of text into individual sentences."""
    sentences = nltk.tokenize.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


_TOP_N_EMOTIONS = 3


def classify_sentences(sentences: list[str]) -> list[EmotionResult]:
    """Run each sentence through the emotion classifier.

    Returns EmotionResult dicts: emotion/family/confidence are the top
    label, "emotions" is the top 3 labels with scores -- pipeline.py's
    _blend_profile() uses this to blend pause/volume across all 3 instead
    of committing fully to just the top label."""
    classifier = _get_classifier()
    results = []
    for sentence in sentences:
        scores = classifier(sentence)[0]  # list of {"label": ..., "score": ...}
        ranked = sorted(scores, key=lambda s: s["score"], reverse=True)
        top = ranked[0]
        top_n = [(s["label"], round(s["score"], 3)) for s in ranked[:_TOP_N_EMOTIONS]]
        results.append(
            {
                "text": sentence,
                "emotion": top["label"],
                "family": to_family(top["label"]),
                "confidence": round(top["score"], 3),
                "emotions": top_n,
            }
        )
    return results


def smooth_emotions(
    tagged: list[EmotionResult], confidence_threshold: float = 0.4
) -> list[EmotionResult]:
    """Only switch the active prosody family when confidence is high enough;
    otherwise inherit the previous sentence's family, so narration doesn't
    flip-flop tone every sentence."""
    if not tagged:
        return tagged

    smoothed = []
    active_family = tagged[0]["family"]

    for item in tagged:
        if item["confidence"] >= confidence_threshold:
            active_family = item["family"]
        smoothed.append({**item, "family": active_family})

    return smoothed


def analyze(text: str) -> list[EmotionResult]:
    """Full stage 2+3: text -> list of smoothed, emotion-tagged sentences."""
    sentences = split_sentences(text)
    tagged = classify_sentences(sentences)
    return smooth_emotions(tagged)
