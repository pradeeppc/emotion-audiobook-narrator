# Deploys the emotion-aware audiobook narrator as a single container: one
# FastAPI process serves both the API and the frontend's static files (see
# backend/main.py's StaticFiles mount). Built for Hugging Face Spaces
# (Docker SDK) -- see docs/07-deployment.md.

FROM python:3.12-slim

# NOTE: no ffmpeg installed here, deliberately. pydub normally shells out to
# ffmpeg for audio processing, but this project only ever reads/writes WAV
# with no codec/parameters -- verified against pydub's source that this
# specific usage (from_wav / export(format="wav") / +gain / fade_in/out)
# takes pydub's pure-Python `wave`-module code path and never invokes
# ffmpeg. If mp3 export or non-WAV input is ever added, ffmpeg would need
# to be installed here first.

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt

# CPU-only torch build first (avoids pulling ~1GB+ of unused CUDA
# libraries) -- see docs/01-overview.md on why this project stays CPU-only.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

# Bake models into the image at build time, so a reviewer's very first
# request isn't a slow cold-download (Piper voice ~65MB, emotion classifier
# ~500MB) -- see docs/07-deployment.md.
RUN python -m piper.download_voices en_US-lessac-medium --download-dir backend/models

RUN python -c "\
import sys; sys.path.insert(0, 'backend'); \
import emotion; \
emotion._get_classifier(); \
emotion.split_sentences('Warm-up sentence to trigger the nltk punkt_tab download.')"

WORKDIR /app/backend

# Hugging Face Spaces' Docker SDK expects the app to listen on 7860 by
# default.
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
