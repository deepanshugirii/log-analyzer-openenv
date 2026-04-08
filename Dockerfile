# Dockerfile for Log Analyzer OpenEnv
# Compatible with Hugging Face Spaces (SDK: docker)

FROM python:3.11-slim

WORKDIR /app

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# ── Enable Git LFS ────────────────────────────────────────────
RUN git lfs install

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────
COPY . .

RUN git lfs pull

# ── Prepare task data ─────────────────────────────────────────
RUN python prepare_data.py \
    --log    data/raw/HDFS_2k.log \
    --struct data/raw/HDFS_2k_log_structured.csv \
    --labels data/raw/anomaly_label.csv \
    --out    env/data

# ── HuggingFace Spaces: must listen on 7860 ───────────────────
EXPOSE 7860

# ── Non-root user (HF Spaces requirement) ────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "server.app"]
