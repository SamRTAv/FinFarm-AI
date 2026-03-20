"""
config/settings.py

Centralised configuration for the Farm AI & FinAI project.
Replace placeholder values with real secrets via environment variables
or a secrets manager before deploying to production.
"""

import os

# ---------------------------------------------------------------------------
# Farm GPU (KCC Dataset) – data.gov.in
# ---------------------------------------------------------------------------

DATAGOV_API_KEY: str = os.getenv("DATAGOV_API_KEY", "579b464db66ec23bdd000001885b48058a2b4ee3749bf724ad8d25f0")
DATAGOV_RESOURCE_ID: str = os.getenv("DATAGOV_RESOURCE_ID", "cef25fe2-9231-4128-8aec-2c948fedd43f")
DATAGOV_BASE_URL: str = "https://api.data.gov.in/resource/"

# States to pull data from
TARGET_STATES: list[str] = ["GUJARAT", "TAMILNADU"]

# Per-state request limit (API max is usually 1000 per call; set higher and paginate)
DATA_FETCH_LIMIT: int = 30000

# Year filter for data freshness
DATA_YEAR: str = "2025"

# Valid QueryType categories (top-N by frequency)
VALID_QUERY_TYPES_COUNT: int = 11

# Excluded categories
EXCLUDED_QUERY_TYPES: list[str] = ["Government Schemes", "Training and Exposure Visits"]

# Excluded sectors
EXCLUDED_SECTORS: list[str] = ["ANIMAL HUSBANDRY"]

# Columns to drop during preprocessing
COLUMNS_TO_DROP: list[str] = [
    "DistrictName", "BlockName", "Season", "Category",
    "Crop", "QueryText", "CreatedOn",
]

# ---------------------------------------------------------------------------
# Farm GPU – IndicBERT model
# ---------------------------------------------------------------------------

INDICBERT_MODEL_NAME: str = "ai4bharat/indic-bert"
INDICBERT_OUTPUT_DIR: str = "./indicbert-finetuned"
INDICBERT_NUM_LABELS: int = 31
INDICBERT_MAX_LENGTH: int = 128
INDICBERT_TRAIN_EPOCHS: int = 3
INDICBERT_BATCH_SIZE: int = 16
INDICBERT_TEST_SIZE: float = 0.1
INDICBERT_HUB_MODEL_REPO: str = "sam1609/indicbert-finetuned"
INDICBERT_HUB_TOKENIZER_REPO: str = "sam1609/indicbert-tokenizer"

# ---------------------------------------------------------------------------
# Farm GPU – Cosine Retrieval
# ---------------------------------------------------------------------------

INDIC_SBERT_MODEL: str = "l3cube-pune/indic-sentence-similarity-sbert"
TOP_K_RETRIEVAL: int = 50
COSINE_THRESHOLD: float = 0.09

# ---------------------------------------------------------------------------
# Farm GPU – Intermediate data
# ---------------------------------------------------------------------------

RAW_CSV_PATH: str = "filename.csv"
API_DATA_CSV_PATH: str = "apidata.csv"

# ---------------------------------------------------------------------------
# FinAI – Banking dataset
# ---------------------------------------------------------------------------

BITEXT_PARQUET_URL: str = (
    "hf://datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/"
    "bitext-retail-banking-llm-chatbot-training-dataset.parquet"
)
BITEXT_API_URL: str = (
    "https://huggingface.co/api/datasets/bitext/"
    "Bitext-retail-banking-llm-chatbot-training-dataset/parquet/default/train"
)

# ---------------------------------------------------------------------------
# FinAI – BERT model
# ---------------------------------------------------------------------------

BERT_MODEL_NAME: str = "bert-base-uncased"
BERT_OUTPUT_DIR: str = "./bert-finetuned-intent"
BERT_MAX_LENGTH: int = 128
BERT_TRAIN_EPOCHS: int = 3
BERT_BATCH_SIZE: int = 16
BERT_TEST_SIZE: float = 0.1

# ---------------------------------------------------------------------------
# FinAI – Sentence-BERT retrieval
# ---------------------------------------------------------------------------

MINIML_MODEL: str = "all-MiniLM-L6-v2"
TOP_RESPONSES: int = 3

# ---------------------------------------------------------------------------
# FinAI – Groq LLM
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL: str = "llama3-8b-8192"
GROQ_TEMPERATURE: float = 0.7
GROQ_MAX_TOKENS: int = 300

# ---------------------------------------------------------------------------
# Shared – IndicLID
# ---------------------------------------------------------------------------

INDICLID_INPUT_THRESHOLD: float = 0.5
INDICLID_ROMAN_THRESHOLD: float = 0.6
INDICLID_FTN_PATH: str = "models/indiclid-ftn/model_baseline_roman.bin"
INDICLID_FTR_PATH: str = "models/indiclid-ftr/model_baseline_roman.bin"
INDICLID_BERT_PATH: str = "models/indiclid-bert/basline_nn_simple.pt"

# ---------------------------------------------------------------------------
# Shared – IndicTrans2
# ---------------------------------------------------------------------------

INDICTRANS2_MODEL: str = "ai4bharat/indictrans2-indic-en-1B"
INDICTRANS2_MAX_LENGTH: int = 256
INDICTRANS2_NUM_BEAMS: int = 5

SUPPORTED_INDIC_LANGS: list[str] = [
    "hin_Deva", "ben_Beng", "guj_Gujr", "tam_Taml", "tel_Telu", "mar_Deva",
    "kan_Knda", "mal_Mlym", "pan_Guru", "ory_Orya", "asm_Beng", "urd_Arab",
    "npi_Deva", "san_Deva", "kas_Deva", "snd_Deva", "gom_Deva", "mai_Deva",
    "mni_Beng", "sat_Olck",
]

# ---------------------------------------------------------------------------
# Shared – Whisper
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = "openai/whisper-large-v2"
WHISPER_TARGET_SAMPLE_RATE: int = 16000

# ---------------------------------------------------------------------------
# HuggingFace Hub
# ---------------------------------------------------------------------------

HF_TOKEN: str = os.getenv("HF_TOKEN", "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")
