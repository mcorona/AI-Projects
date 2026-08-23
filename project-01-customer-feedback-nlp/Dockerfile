# Serves src/api.py (FastAPI inference server) only -- not training, not
# the Streamlit demo. Uses requirements-api.txt (a minimal subset of
# requirements.txt) so the image doesn't ship Jupyter/Streamlit/pytest/etc.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/

EXPOSE 8000

# No checkpoint is baked into the image -- model artifacts are gitignored
# and shouldn't live in a Docker image either. Mount one at runtime:
#
#   docker build -t financial-sentiment .
#   docker run -p 8000:8000 \
#     -v "$(pwd)/output/models:/app/output/models" \
#     financial-sentiment
#
# src/api.py auto-discovers the newest output/models/best-*.ckpt; to pin a
# specific checkpoint instead, add -e MODEL_CKPT_PATH=/app/output/models/<file>.ckpt
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
