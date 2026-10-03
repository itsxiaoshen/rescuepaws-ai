FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# The pip in the base image is old and can fail on newer packages, so upgrade it first.
# CPU-only PyTorch: the default Linux build bundles GPU libraries and is several GB larger
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding models at build time, so the container starts without fetching them
# (names must match MODEL_NAME in src/matching.py and POLICY_MODEL_NAME in src/rag.py)
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2'); S('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

COPY src/ src/
COPY app/ app/
COPY data/ data/

# Don't run the server as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
