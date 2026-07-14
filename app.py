"""Gradio UI for Hugging Face Spaces (free tier here doesn't support Docker,
so this is an alternative to backend/main.py's FastAPI+HTML version, which
is still used for local dev / Docker deployments elsewhere). Reuses
backend/pipeline.py and backend/emotion.py unchanged -- only the UI layer
differs.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import gradio as gr
import spaces

import emotion
import pipeline


@spaces.GPU
def _zerogpu_placeholder():
    """No-op. This Space's only available free hardware tier is ZeroGPU,
    which requires at least one @spaces.GPU-decorated function to exist or
    it refuses to start -- CPU Basic (what we'd actually use) requires a
    PRO subscription to switch to on this account. Our app never touches a
    GPU (Piper/onnxruntime run CPU-only); this function exists solely to
    satisfy that platform requirement and is never called."""

_MODELS_DIR = Path(__file__).parent / "backend" / "models"
_MODEL_FILE = _MODELS_DIR / "en_US-lessac-medium.onnx"

# backend/models/ isn't committed to git (see .gitignore) -- fetch the Piper
# voice on first run instead, same as a fresh local checkout would need to.
if not _MODEL_FILE.exists():
    from piper.download_voices import download_voice

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_voice("en_US-lessac-medium", _MODELS_DIR)


def preview_emotions(text: str):
    if not text.strip():
        return []
    tagged = emotion.analyze(text)
    return [[t["text"], t["emotion"], t["family"], t["confidence"]] for t in tagged]


def generate_narration(text: str):
    if not text.strip():
        return None
    audio_bytes = pipeline.narrate(text)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


with gr.Blocks(title="Emotion-Aware Audiobook Narrator") as demo:
    gr.Markdown("# Emotion-Aware Audiobook Narrator")
    gr.Markdown(
        "Paste text below. Sentences are narrated with pacing/volume that "
        "reflects each sentence's detected emotion."
    )

    text_input = gr.Textbox(
        label="Text", lines=8, placeholder="Paste a paragraph or short story here..."
    )

    with gr.Row():
        analyze_btn = gr.Button("Preview detected emotions")
        narrate_btn = gr.Button("Generate narration")

    emotion_table = gr.Dataframe(
        headers=["Sentence", "Emotion", "Family", "Confidence"],
        label="Detected emotions",
    )
    audio_output = gr.Audio(label="Narration", type="filepath")

    analyze_btn.click(fn=preview_emotions, inputs=text_input, outputs=emotion_table)
    narrate_btn.click(fn=generate_narration, inputs=text_input, outputs=audio_output)


if __name__ == "__main__":
    demo.launch()
