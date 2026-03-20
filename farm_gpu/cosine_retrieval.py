"""
farm_gpu/cosine_retrieval.py

Given a user query and a filtered DataFrame of KCC records, uses
Sentence-BERT (l3cube-pune/indic-sentence-similarity-sbert) to find
the top-K most semantically similar answers.

Usage:
    from farm_gpu.cosine_retrieval import KCCRetriever
    retriever = KCCRetriever()
    top_results = retriever.retrieve(query, df_filtered)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from config.settings import (
    INDIC_SBERT_MODEL,
    TOP_K_RETRIEVAL,
    API_DATA_CSV_PATH,
)
from shared.utils import get_logger

logger = get_logger(__name__)


class KCCRetriever:
    """
    Semantic retrieval over KCC answer records using cosine similarity.

    Parameters
    ----------
    model_name : str
        Sentence-BERT model identifier.
    top_k : int
        Number of top results to return.
    """

    def __init__(self, model_name: str = INDIC_SBERT_MODEL, top_k: int = TOP_K_RETRIEVAL):
        logger.info("Loading Sentence-BERT: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def retrieve(self, query: str, df: pd.DataFrame, text_col: str = "KccAns") -> pd.DataFrame:
        """
        Find the top-K rows in df most similar to query.

        Parameters
        ----------
        query : str
            User query (any Indic language or English).
        df : pd.DataFrame
            DataFrame of candidate answers (filtered by state and query type).
        text_col : str
            Column in df containing the answer text to embed.

        Returns
        -------
        pd.DataFrame
            Top-K rows from df, ordered by cosine similarity (descending).
        """
        texts = df[text_col].tolist()

        logger.info("Encoding %d candidate answers …", len(texts))
        corpus_embeddings = self.model.encode(texts, convert_to_tensor=True)
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=min(self.top_k, len(texts)))

        top_df = df.iloc[top_results.indices.tolist()].copy()
        top_df["cosine_score"] = top_results.values.tolist()
        top_df = top_df.sort_values("cosine_score", ascending=False).reset_index(drop=True)

        logger.info("Returning top-%d results.", len(top_df))
        return top_df

    def retrieve_from_csv(self, query: str, state: str, target_labels: list[str]) -> pd.DataFrame:
        """
        Load the apidata CSV, filter by state and target QueryTypes,
        then run cosine retrieval.

        Parameters
        ----------
        query : str
        state : str
            State name to filter on (e.g. "TAMILNADU").
        target_labels : list[str]
            QueryType labels predicted by the classifier.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_csv(API_DATA_CSV_PATH)
        df = df[df["StateName"] == state]
        df = df[df["QueryType"].isin(target_labels)]
        return self.retrieve(query, df)


if __name__ == "__main__":
    retriever = KCCRetriever()
    test_query = "வரவிருக்கும் குளிர்காலத்தில் எந்த மாதத்தில் நெல் விதைகளை விதைக்க வேண்டும்?"
    results = retriever.retrieve_from_csv(
        query=test_query,
        state="TAMILNADU",
        target_labels=["Sowing Time and Weather", "Varieties"],
    )
    print(results[["KccAns", "cosine_score"]].head(10))
