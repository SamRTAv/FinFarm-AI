"""
shared/registry.py

Lazy, cached singletons for every heavy object in the pipeline.

Each model / dataset is built at most once per process (functools.lru_cache),
so the LangGraph nodes can call e.g. registry.get_translation() freely without
reloading multi-GB models. Construct nothing at import time — the first call
triggers the load, and app startup can warm them by calling warmup().

Model source resolution: prefer a local fine-tuned directory if it exists
(local dev), otherwise fall back to the HuggingFace Hub repo (HF Spaces).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import functools

import pandas as pd

from config.settings import (
    API_DATA_CSV_PATH,
    INDICBERT_OUTPUT_DIR,
    INDICBERT_HUB_MODEL_REPO,
    INDICBERT_HUB_TOKENIZER_REPO,
    BERT_OUTPUT_DIR,
    BERT_HUB_MODEL_REPO,
)
from shared.utils import get_logger

logger = get_logger(__name__)


def _resolve(local_dir: str, hub_repo: str) -> str:
    """Use the local fine-tuned dir if present, else the Hub repo id."""
    return local_dir if os.path.isdir(local_dir) else hub_repo


# ---------------------------------------------------------------------------
# Translation / detection
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_translation():
    from shared.translation import TranslationService
    return TranslationService()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_kcc_df() -> pd.DataFrame:
    """The KCC answer corpus used for Agro retrieval (apidata.csv)."""
    logger.info("Loading KCC corpus from %s", API_DATA_CSV_PATH)
    return pd.read_csv(API_DATA_CSV_PATH)


@functools.lru_cache(maxsize=1)
def get_banking_df() -> pd.DataFrame:
    """The Bitext banking corpus (intent + response) for Banking retrieval."""
    from fin_ai.data_extraction import load_banking_dataset
    return load_banking_dataset()


# ---------------------------------------------------------------------------
# Label mappings (int id -> human-readable label string)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _farm_id2label() -> dict[int, str]:
    """
    Reconstruct the Agro id->label map from the KCC corpus.

    Training used `df["label"].astype("category").cat.codes`, and pandas
    assigns category codes by *sorted* category order — so sorting the unique
    QueryType values reproduces the same index->label mapping.
    """
    df = get_kcc_df()
    cats = sorted(df["QueryType"].dropna().unique().tolist())
    return dict(enumerate(cats))


@functools.lru_cache(maxsize=1)
def get_banking_id2intent() -> dict[int, str]:
    """
    Banking id->intent map.

    NOTE: fin_ai.data_preprocessing.encode_labels returns the variables in a
    confusingly named order — its *first* returned dict (`label2id`) is in fact
    {int_id: intent_str}, which is what BankingRetriever / the classifier need.
    We take that one and ignore the inverted second dict to avoid the latent
    bug in the original __main__ blocks.
    """
    from fin_ai.data_preprocessing import encode_labels
    _df, id_to_intent, _intent_to_id = encode_labels(get_banking_df())
    return dict(id_to_intent)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_farm_classifier():
    """IndicBERT KCC query-type classifier (operates on native Indic text)."""
    from transformers import AutoTokenizer
    from farm_gpu.inference import FarmQueryClassifier

    src = _resolve(INDICBERT_OUTPUT_DIR, INDICBERT_HUB_MODEL_REPO)
    clf = FarmQueryClassifier(model_dir=src)

    # The tokenizer was pushed to a separate Hub repo, so load it explicitly
    # when we resolved to the Hub (a local dir bundles its own tokenizer).
    if src == INDICBERT_HUB_MODEL_REPO:
        clf.tokenizer = AutoTokenizer.from_pretrained(INDICBERT_HUB_TOKENIZER_REPO)

    clf.set_label_mapping(_farm_id2label())
    return clf


@functools.lru_cache(maxsize=1)
def get_banking_classifier():
    """BERT banking-intent classifier (operates on English text)."""
    from fin_ai.inference import BankingIntentClassifier

    src = _resolve(BERT_OUTPUT_DIR, BERT_HUB_MODEL_REPO)
    clf = BankingIntentClassifier(model_dir=src)
    clf.set_label_mapping(get_banking_id2intent())
    return clf


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_farm_retriever():
    from farm_gpu.cosine_retrieval import KCCRetriever
    return KCCRetriever()


@functools.lru_cache(maxsize=1)
def get_banking_retriever():
    from fin_ai.cosine_retrieval import BankingRetriever
    return BankingRetriever(get_banking_df(), get_banking_id2intent())


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup() -> None:
    """Eagerly build everything (call once at FastAPI startup)."""
    logger.info("Warming up all models …")
    get_translation()
    get_farm_classifier()
    get_farm_retriever()
    get_banking_classifier()
    get_banking_retriever()
    logger.info("Warmup complete.")
