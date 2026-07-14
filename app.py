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
    """No-op, required so this ZeroGPU-tier Space starts (needs at least
    one @spaces.GPU function). App is CPU-only; never actually called."""

_MODELS_DIR = Path(__file__).parent / "backend" / "models"
_MODEL_FILE = _MODELS_DIR / "en_US-lessac-medium.onnx"

# Not committed to git -- fetch on first run instead.
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
