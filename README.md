---
title: FinFarm AI
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Farm AI & FinAI Project

Multilingual **Agro / Banking / General** query assistant. A user submits a
query in any supported Indic language and picks a domain; the pipeline detects
the language, classifies + retrieves relevant answers, synthesises a reply with
Groq, and translates it back to the original language.

## Deploying to a HuggingFace Docker Space

1. Create a **Docker** Space and push this folder to it.
2. In **Settings → Variables and secrets**, add (as *Secrets*):
   - `GROQ_API_KEY`
   - `HF_TOKEN`
   - `DATAGOV_API_KEY` (only if re-fetching KCC data)
   > ⚠️ The fallback values currently hard-coded in `config/settings.py` are
   > exposed and must be **rotated**. Production reads them from env only.
3. The Space builds the `Dockerfile`, warms the models on startup, and serves
   on port `7860`.

## API

`POST /query`
```json
{ "query": "மழைக்கு பிறகு நெல் எப்போது விதைக்க வேண்டும்?",
  "domain": "agro",
  "state_name": "TAMILNADU" }
```
Returns `{ final_answer, detected_lang, domain, intent, labels, answers_en }`.
Interactive docs at `/docs`; liveness at `/health`.

## Architecture (unified pipeline)

```
START → detect_language → translate_query → route by domain
   agro    → classify (native) → retrieve KCC → translate answers→EN ┐
   banking → translate query→EN → classify → retrieve (EN)           ├→ Groq synthesis
   general → (no retrieval)                                          ┘
                                          → back-translate (EN→original) → END
```
Orchestrated with **LangGraph** in `pipeline/graph.py`.

## Project Structure

```
farm_ai_project/
│
├── config/
│   └── settings.py               # All API keys, constants, paths
│
├── farm_gpu/                      # Farmer Query Classification (KCC Dataset)
│   ├── data_extraction.py         # Pulls data from data.gov.in API
│   ├── data_preprocessing.py      # Cleaning, filtering, merging, encoding
│   ├── model_training.py          # IndicBERT fine-tuning with HuggingFace Trainer
│   ├── inference.py               # Predict label + probabilities
│   ├── language_detection.py      # IndicLID language detection
│   └── cosine_retrieval.py        # Sentence-BERT cosine similarity retrieval
│
├── fin_ai/                        # Financial / Banking Intent Classifier
│   ├── data_extraction.py         # Load Bitext banking dataset from HuggingFace
│   ├── data_preprocessing.py      # Label encoding, tokenization, train/val split
│   ├── model_training.py          # BERT fine-tuning with HuggingFace Trainer
│   ├── inference.py               # Predict intent + probabilities
│   ├── audio_transcription.py     # Whisper STT (translate to English)
│   ├── language_detection.py      # IndicLID + IndicTrans2 translation
│   └── cosine_retrieval.py        # MiniLM cosine similarity + Groq LLM synthesis
│
└── shared/
    └── utils.py                   # Shared utilities
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Each module can be run independently or imported as a library.
