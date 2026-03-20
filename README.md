# Farm AI & FinAI Project

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
