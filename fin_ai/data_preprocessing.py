"""
fin_ai/data_preprocessing.py

Prepares the Bitext banking dataset for BERT fine-tuning:
  - Select and clean columns
  - Encode intent labels
  - Tokenise with BertTokenizer
  - Split into train / validation sets

Usage:
    from fin_ai.data_preprocessing import build_tokenized_dataset
    tokenized_dataset, label2id, id2label = build_tokenized_dataset(df)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer

from config.settings import (
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_TEST_SIZE,
)
from shared.utils import get_logger

logger = get_logger(__name__)


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Encode the 'intent' column as integer category codes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an 'intent' column.

    Returns
    -------
    df : pd.DataFrame
        With a new 'label' integer column.
    label2id : dict[int, str]
        Maps integer code → intent string.
    id2label : dict[str, int]
        Maps intent string → integer code.
    """
    df = df[["instruction", "intent"]].dropna().copy()
    df["intent"] = df["intent"].astype("category")
    df["label"] = df["intent"].cat.codes
    label2id = dict(enumerate(df["intent"].cat.categories))
    id2label = {v: k for k, v in label2id.items()}
    logger.info("Unique intent classes: %d", len(label2id))
    return df, label2id, id2label


def build_tokenized_dataset(
    df: pd.DataFrame,
) -> tuple[DatasetDict, dict, dict]:
    """
    Full preprocessing pipeline: encode labels → tokenise → split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw banking DataFrame (from data_extraction.load_banking_dataset).

    Returns
    -------
    tokenized_dataset : DatasetDict
        With 'train' and 'test' splits.
    label2id : dict[int, str]
    id2label : dict[str, int]
    """
    df, label2id, id2label = encode_labels(df)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df[["instruction", "label"]])

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def tokenize_function(example):
        return tokenizer(
            example["instruction"],
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    split = tokenized.train_test_split(test_size=BERT_TEST_SIZE, shuffle=True)

    def _drop_index(ds):
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])
        return ds

    tokenized_dataset = DatasetDict(
        train=_drop_index(split["train"]),
        test=_drop_index(split["test"]),
    )
    logger.info("Train: %d | Test: %d",
                len(tokenized_dataset["train"]), len(tokenized_dataset["test"]))
    return tokenized_dataset, label2id, id2label


if __name__ == "__main__":
    from fin_ai.data_extraction import load_banking_dataset
    df = load_banking_dataset()
    ds, l2i, i2l = build_tokenized_dataset(df)
    print(ds)
    print("label2id:", l2i)
