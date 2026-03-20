"""
farm_gpu/data_extraction.py

Pulls farmer-query records from the data.gov.in KCC API for the configured
states and year, then returns a combined raw DataFrame.

Usage:
    from farm_gpu.data_extraction import fetch_kcc_data
    df = fetch_kcc_data()
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import requests
import pandas as pd

from config.settings import (
    DATAGOV_API_KEY,
    DATAGOV_RESOURCE_ID,
    DATAGOV_BASE_URL,
    TARGET_STATES,
    DATA_FETCH_LIMIT,
    DATA_YEAR,
)
from shared.utils import get_logger

logger = get_logger(__name__)


def fetch_state_data(state: str) -> pd.DataFrame:
    """
    Fetch KCC records for a single state from data.gov.in.

    Parameters
    ----------
    state : str
        State name as expected by the API (e.g. "GUJARAT").

    Returns
    -------
    pd.DataFrame
        Raw records for the given state.
    """
    url = DATAGOV_BASE_URL + DATAGOV_RESOURCE_ID
    params = {
        "api-key": DATAGOV_API_KEY,
        "format": "json",
        "limit": DATA_FETCH_LIMIT,
        "filters[StateName]": state,
        "filters[year]": DATA_YEAR,
    }

    logger.info("Fetching data for state=%s year=%s ...", state, DATA_YEAR)
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    records = data.get("records", [])
    logger.info("Received %d records for state=%s", len(records), state)
    return pd.DataFrame(records)


def fetch_kcc_data(states: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Fetch KCC data for all target states.

    Parameters
    ----------
    states : list[str] | None
        Override list of states. Defaults to TARGET_STATES from settings.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of state name → raw DataFrame.
    """
    states = states or TARGET_STATES
    result: dict[str, pd.DataFrame] = {}
    for state in states:
        result[state] = fetch_state_data(state)
    return result


if __name__ == "__main__":
    dfs = fetch_kcc_data()
    for state, df in dfs.items():
        print(f"{state}: {len(df)} rows")
