FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

RUN python -m piper.download_voices en_US-lessac-medium --download-dir backend/models

RUN python -c "\
import sys; sys.path.insert(0, 'backend'); \
import emotion; \
emotion._get_classifier(); \
emotion.split_sentences('Warm-up sentence to trigger the nltk punkt_tab download.')"

WORKDIR /app/backend

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
