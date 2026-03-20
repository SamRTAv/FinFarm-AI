"""
farm_gpu/data_preprocessing.py

Cleans, filters, merges, and encodes KCC DataFrames produced by
data_extraction.py into a model-ready DataFrame.

Usage:
    from farm_gpu.data_preprocessing import build_combined_dataset
    df = build_combined_dataset(state_dfs)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config.settings import (
    VALID_QUERY_TYPES_COUNT,
    EXCLUDED_QUERY_TYPES,
    EXCLUDED_SECTORS,
    COLUMNS_TO_DROP,
    RAW_CSV_PATH,
)
from shared.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-state filtering helpers
# ---------------------------------------------------------------------------

def _get_valid_query_types(df: pd.DataFrame, top_n: int) -> list[str]:
    """Return the top-N QueryType values by frequency."""
    return df["QueryType"].value_counts().index[:top_n].tolist()


def filter_gujarat_data(df_gujarat: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Gujarat-specific filtering rules:
      - Keep only top-N query types.
      - Drop 'Government Schemes'.
    """
    valid = _get_valid_query_types(df_gujarat, VALID_QUERY_TYPES_COUNT)
    df = df_gujarat[df_gujarat["QueryType"].isin(valid)].copy()
    df = df[df["QueryType"] != "Government Schemes"]
    logger.info("Gujarat after filtering: %d rows", len(df))
    return df


def filter_tamilnadu_data(df_tamil: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Tamil Nadu-specific filtering rules:
      - Drop 'Training and Exposure Visits'.
      - Keep only top-N query types.
    """
    df = df_tamil[df_tamil["QueryType"] != "Training and Exposure Visits"].copy()
    valid = _get_valid_query_types(df, VALID_QUERY_TYPES_COUNT - 2)  # top 9
    df = df[df["QueryType"].isin(valid)]
    logger.info("Tamil Nadu after filtering: %d rows", len(df))
    return df


def filter_older_state_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a generic older-year state DataFrame to only retain
    manually curated query types for consistency across years.
    """
    keep = [
        "Fertilizer Use and Availability",
        "Cultural Practices",
        "Sowing Time and Weather",
        "Varieties",
        "Nutrient Management",
        "\tWater Management\t",
        "Weed Management",
        "Market Information",
    ]
    return df[df["QueryType"].isin(keep)].copy()


# ---------------------------------------------------------------------------
# Merge & final cleaning
# ---------------------------------------------------------------------------

def merge_state_frames(filtered_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate filtered state DataFrames and shuffle."""
    df = pd.concat(filtered_frames, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    logger.info("Combined dataset: %d rows", len(df))
    return df


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning step:
      - Remove unwanted sectors.
      - Drop irrelevant columns.
      - Rename columns for the model pipeline.
      - Add one-hot encoded label columns.
    """
    # Drop excluded sectors
    df = df[~df["Sector"].isin(EXCLUDED_SECTORS)].copy()

    # Drop unnecessary columns
    existing_cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    # Rename for downstream use
    df.rename(columns={"KccAns": "Query", "QueryType": "label"}, inplace=True)

    # One-hot encoding
    one_hot = pd.get_dummies(df["label"])
    df = pd.concat([df, one_hot], axis=1)

    logger.info("After clean_and_encode: %d rows, columns=%s", len(df), df.columns.tolist())
    return df


def build_label_mappings(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build id2label and label2id mappings from the 'label' column.

    Returns
    -------
    id2label : dict[int, str]
    label2id : dict[str, int]
    """
    categories = df["label"].astype("category").cat.categories
    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_combined_dataset(state_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    state_dfs : dict[str, pd.DataFrame]
        Raw DataFrames keyed by state name (from data_extraction.fetch_kcc_data).

    Returns
    -------
    pd.DataFrame
        Cleaned, merged, and encoded DataFrame ready for tokenisation.
    """
    filtered = []
    for state, raw_df in state_dfs.items():
        state_upper = state.upper()
        if state_upper == "GUJARAT":
            filtered.append(filter_gujarat_data(raw_df))
        elif state_upper == "TAMILNADU":
            filtered.append(filter_tamilnadu_data(raw_df))
        else:
            filtered.append(filter_older_state_data(raw_df))

    # Align valid categories across all states before merging
    # Use Tamil Nadu's valid set as the master (as done in the notebook)
    tamil_df = next((f for s, f in zip(state_dfs.keys(), filtered)
                     if s.upper() == "TAMILNADU"), None)
    if tamil_df is not None:
        master_valid = tamil_df["QueryType"].unique().tolist() if "QueryType" in tamil_df.columns \
            else tamil_df.columns.tolist()
        # Re-filter each frame to the shared valid set
        filtered = [f[f["QueryType"].isin(master_valid)] if "QueryType" in f.columns else f
                    for f in filtered]

    combined = merge_state_frames(filtered)
    combined = clean_and_encode(combined)
    combined.to_csv(RAW_CSV_PATH, index=False)
    logger.info("Saved raw combined CSV to %s", RAW_CSV_PATH)
    return combined


if __name__ == "__main__":
    from farm_gpu.data_extraction import fetch_kcc_data
    state_dfs = fetch_kcc_data()
    df = build_combined_dataset(state_dfs)
    print(df["label"].value_counts())
