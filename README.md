Try it here - https://deep765-emotion-audiobook-narrator.hf.space/?

<img width="1914" height="878" alt="image" src="https://github.com/user-attachments/assets/fef449bf-db9e-4179-820a-7efa3837db24" />


---
title: Emotion Audiobook Narrator
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Emotion-Aware Audiobook Narrator (MVP v1)

Turns plain text into narration where pacing/tone shifts based on each
sentence's detected emotion. An NLP emotion classifier
(`SamLowe/roberta-base-go_emotions`) tags each sentence, and a Piper-based
TTS pipeline (`backend/tts.py`/`backend/pipeline.py`) turns that into
narration with emotion-appropriate pacing and volume, staying on one
consistent CPU-only voice throughout.

## The pipeline, end to end

```
 Raw text
    |
    v
[1] Text segmentation        -> break into sentences/paragraphs
    |
    v
[2] Emotion classification   -> tag each segment: sad / happy / angry / tense / neutral...
    |
    v
[3] Context smoothing        -> avoid emotion flip-flopping sentence to sentence
    |
    v
[4] Emotion -> voice mapping -> convert emotion tag into TTS control params
    |
    v
[5] TTS synthesis            -> generate audio per segment
    |
    v
[6] Audio stitching          -> join segments with natural pacing/pauses
    |
    v
 Final audio file (.wav/.mp3)
```

## Project structure

```
emotion-audiobook-narrator/
  backend/
    main.py       FastAPI app (HTTP endpoints, serves the frontend too)
    pipeline.py   orchestrates emotion tagging -> TTS -> audio stitching
    emotion.py    text -> emotion tags
    tts.py        text + emotion -> audio
    models/       downloaded Piper voice files
    venv/         Python virtual environment (not committed)
  frontend/
    index.html, style.css, script.js   plain HTML/JS UI, no build step
  samples/
    sample_text.txt   a short story to test with
```

## Running it

### 1. Backend

```bash
cd backend
source venv/bin/activate      # created already; if missing: python3 -m venv venv
uvicorn main:app --reload
```

This starts the API at `http://127.0.0.1:8000`, which also serves the
frontend directly.

The **first** request will be slow (~30-60s) — it downloads the emotion
classification model (~500MB) the first time. After that it's cached and
fast.

### 2. Try it

Open `http://127.0.0.1:8000/` in a browser, paste a short paragraph with
some emotional variety (see `samples/sample_text.txt` for an example),
click **Preview detected emotions** to see the per-sentence tags, then
**Generate narration** to hear it.

## Current limitations

- **The TTS voice has no native emotion control.** Emotional delivery is
  approximated with pause length, volume, and pacing rather than true
  prosody control — noticeably better than flat narration, but with a
  real ceiling compared to an emotion-trained TTS model.
- **Sentence-by-sentence classification is stateless** beyond a simple
  smoothing pass — long-range plot/mood arcs aren't modeled.
- **No async job queue yet** — a long chapter will make the browser wait
  for the full generation before responding.

## Next steps (roadmap)

1. Add character/dialogue voice differentiation.
2. Add async job handling for longer texts.
3. Deploy on a zero/near-zero-cost host — the whole pipeline is CPU-only,
   so it runs on a free tier or a cheap VPS with no GPU required.
