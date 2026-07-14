"""FastAPI backend: wires up HTTP endpoints to pipeline.py.

Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000/ for the app, or /docs for the API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline

app = FastAPI(title="Emotion-Aware Audiobook Narrator")

# Lets the frontend call this API from a different origin during local dev
# (e.g. opening frontend/index.html as a file:// URL). Not needed once
# deployed, since the StaticFiles mount below serves both from one origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NarrateRequest(BaseModel):
    text: str


@app.post("/narrate")
def narrate(req: NarrateRequest):
    """Text in, WAV audio bytes out."""
    audio_bytes = pipeline.narrate(req.text)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/analyze")
def analyze(req: NarrateRequest):
    """Text in, per-sentence emotion tags out (no audio) -- useful for
    previewing/debugging what the AI detected before generating audio."""
    import emotion

    return {"sentences": emotion.analyze(req.text)}


@app.get("/health")
def health():
    return {"status": "ok"}


# Mounted last so the API routes above take priority; this then serves the
# frontend for everything else, including "/" (html=True -> index.html).
_frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
