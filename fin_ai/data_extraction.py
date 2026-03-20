"""
fin_ai/data_extraction.py

Loads the Bitext retail-banking LLM chatbot training dataset from HuggingFace
and returns it as a pandas DataFrame.

Usage:
    from fin_ai.data_extraction import load_banking_dataset
    df = load_banking_dataset()
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import requests
import pandas as pd

from config.settings import BITEXT_PARQUET_URL, BITEXT_API_URL
from shared.utils import get_logger

logger = get_logger(__name__)


def load_banking_dataset() -> pd.DataFrame:
    """
    Load the Bitext banking dataset directly from HuggingFace Hub.

    Tries the direct Parquet URL first; falls back to the HF API endpoint
    to discover the Parquet URL dynamically.

    Returns
    -------
    pd.DataFrame
    """
    try:
        logger.info("Loading banking dataset from HuggingFace Hub …")
        df = pd.read_parquet(BITEXT_PARQUET_URL, engine="pyarrow")
        logger.info("Loaded %d rows from direct Parquet URL.", len(df))
        return df
    except Exception as primary_error:
        logger.warning("Direct Parquet load failed: %s — trying API endpoint.", primary_error)

    try:
        response = requests.get(BITEXT_API_URL, timeout=30)
        response.raise_for_status()
        parquet_urls = response.json()
        parquet_url = parquet_urls[0]
        logger.info("Discovered Parquet URL: %s", parquet_url)
        df = pd.read_parquet(parquet_url, engine="pyarrow")
        logger.info("Loaded %d rows via API endpoint.", len(df))
        return df
    except Exception as fallback_error:
        raise RuntimeError(
            "Failed to load banking dataset from HuggingFace Hub."
        ) from fallback_error


if __name__ == "__main__":
    df = load_banking_dataset()
    print(df.shape)
    print(df.head())
