# ---------------------------------------------------------------------------
# FinFarm AI — HuggingFace Docker Space
# CPU image. Models are pulled from the Hub at runtime into a writable cache.
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# System deps: git (clone IndicLID / IndicTransToolkit), build tools for
# fasttext, libsndfile for audio (kept light; audio path deferred).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential wget unzip \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces convention: run as a non-root user with HOME=/home/user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # Cache models on the writable persistent volume.
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PORT=7860

WORKDIR /home/user/app

# --- Python dependencies (cached layer) ---
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- External AI4Bharat tooling (git-only installs) ---
# IndicTransToolkit is an installable package; IndicLID is imported by path
# (sys.path.append "./IndicLID/Inference" in shared/translation.py).
RUN git clone --depth 1 https://github.com/VarunGumma/IndicTransToolkit.git \
    && pip install --no-cache-dir -e ./IndicTransToolkit \
    && git clone --depth 1 https://github.com/AI4Bharat/IndicLID.git

# --- Download the IndicLID model binaries into ./models ---
# (these are the .bin / .pt files referenced by INDICLID_*_PATH in settings.py)
RUN mkdir -p models \
    && wget -q https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftn.zip -O models/ftn.zip \
    && wget -q https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip -O models/ftr.zip \
    && wget -q https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip -O models/bert.zip \
    && cd models \
    && for z in ftn ftr bert; do unzip -o $z.zip && rm $z.zip; done || true

# --- Application code ---
COPY --chown=user . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
