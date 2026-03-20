"""
fin_ai/cosine_retrieval.py

Given a banking query and the Bitext dataset, this module:
  1. Filters responses by predicted intent.
  2. Embeds query + responses with all-MiniLM-L6-v2 (Sentence-BERT).
  3. Returns the top-K most similar responses.
  4. Synthesises them into a single coherent answer using the Groq LLM API.

Usage:
    from fin_ai.cosine_retrieval import BankingRetriever
    retriever = BankingRetriever(df, id2label)
    answer = retriever.answer("I forgot my ATM PIN", predicted_label_id=12)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch
import requests
from sentence_transformers import SentenceTransformer, util

from config.settings import (
    MINIML_MODEL,
    TOP_RESPONSES,
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
)
from shared.utils import get_logger

logger = get_logger(__name__)


class BankingRetriever:
    """
    Retrieval-augmented answering for banking queries.

    Parameters
    ----------
    df : pd.DataFrame
        Full Bitext dataset (must have 'intent' and 'response' columns).
    id2label : dict[int, str]
        Maps class index → intent string.
    model_name : str
        Sentence-BERT model for similarity.
    top_k : int
        Number of top responses to retrieve.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id2label: dict[int, str],
        model_name: str = MINIML_MODEL,
        top_k: int = TOP_RESPONSES,
    ):
        logger.info("Loading Sentence-BERT: %s", model_name)
        self.sbert = SentenceTransformer(model_name)
        self.df = df
        self.id2label = id2label
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, predicted_label_id: int) -> list[str]:
        """
        Retrieve the top-K most similar responses for a given intent.

        Parameters
        ----------
        query : str
            User query in English.
        predicted_label_id : int
            Predicted class index from BankingIntentClassifier.

        Returns
        -------
        list[str]
            Top-K response strings, ordered by similarity.
        """
        intent_str = self.id2label.get(predicted_label_id, "")
        intent_responses = self.df[self.df["intent"] == intent_str]["response"].tolist()

        if not intent_responses:
            logger.warning("No responses found for intent '%s'", intent_str)
            return []

        query_emb = self.sbert.encode(query, convert_to_tensor=True)
        resp_embs = self.sbert.encode(intent_responses, convert_to_tensor=True)

        if torch.cuda.is_available():
            query_emb = query_emb.to("cuda")
            resp_embs = resp_embs.to("cuda")

        cos_scores = util.pytorch_cos_sim(query_emb, resp_embs)[0]
        top_results = cos_scores.topk(min(self.top_k, len(intent_responses)))

        responses = [intent_responses[idx] for idx in top_results[1]]
        logger.info("Retrieved %d responses for intent '%s'", len(responses), intent_str)
        return responses

    # ------------------------------------------------------------------
    # Groq synthesis
    # ------------------------------------------------------------------

    def synthesise_with_groq(self, responses: list[str]) -> str:
        """
        Send the retrieved responses to Groq (LLaMA-3) to produce a
        single, concise, coherent answer.

        Parameters
        ----------
        responses : list[str]

        Returns
        -------
        str
            Synthesised final answer from the LLM.
        """
        joined = "\n".join(f"- {r}" for r in responses)
        prompt = (
            "Given the following related answers from a banking dataset, combine them into "
            "one short, clear, and helpful response:\n\n"
            f"{joined}\n\n"
            "Give the final answer in one paragraph:"
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": GROQ_TEMPERATURE,
            "max_tokens": GROQ_MAX_TOKENS,
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            logger.info("Groq synthesis successful.")
            return answer
        else:
            logger.error("Groq API error %d: %s", response.status_code, response.text)
            return "\n".join(responses)  # Fallback: return raw responses joined

    # ------------------------------------------------------------------
    # End-to-end answer
    # ------------------------------------------------------------------

    def answer(self, query: str, predicted_label_id: int) -> str:
        """
        Full pipeline: retrieve → synthesise → return final answer.

        Parameters
        ----------
        query : str
            English user query.
        predicted_label_id : int
            Class index from BankingIntentClassifier.

        Returns
        -------
        str
            Final synthesised answer.
        """
        responses = self.retrieve(query, predicted_label_id)
        if not responses:
            return "I'm sorry, I couldn't find relevant information for your query."
        return self.synthesise_with_groq(responses)


if __name__ == "__main__":
    from fin_ai.data_extraction import load_banking_dataset
    from fin_ai.data_preprocessing import encode_labels
    from fin_ai.inference import BankingIntentClassifier

    df_raw = load_banking_dataset()
    df, label2id, id2label = encode_labels(df_raw)

    clf = BankingIntentClassifier()
    clf.set_label_mapping(id2label)

    query = "I forgot my ATM PIN, what should I do?"
    pred_id = clf.predict(query)

    retriever = BankingRetriever(df_raw, id2label)
    answer = retriever.answer(query, pred_id)
    print("Answer:", answer)
