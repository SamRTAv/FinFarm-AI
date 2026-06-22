"""
shared/groq_client.py

Query-grounded answer synthesis using the Groq (LLaMA-3) API.

This is the convergence point of the diagram: every branch feeds the
user's English query plus its Top-K retrieved answers here, and Groq returns
one clean English answer. The General branch passes an empty answer list,
in which case Groq answers the question directly from its own knowledge.

Note: this differs from the original fin_ai prompt, which merged answers
without the user's question. Here the answer is explicitly grounded in the
query so the synthesis stays on-topic.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import requests

from config.settings import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
)
from shared.utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant for Indian users asking agriculture, banking, "
    "and general questions. Answer accurately, concisely, and in plain language."
)


def _build_user_prompt(query_en: str, answers: list[str], domain: str) -> str:
    if answers:
        context = "\n".join(f"- {a}" for a in answers)
        return (
            f"User question:\n{query_en}\n\n"
            f"Reference answers retrieved from a {domain} knowledge base:\n{context}\n\n"
            "Using only the relevant reference answers, write one clear, accurate, and "
            "helpful reply to the user's question in a single short paragraph. "
            "If the references do not address the question, say so briefly."
        )
    return (
        f"User question:\n{query_en}\n\n"
        "Answer clearly and helpfully in one short paragraph."
    )


def synthesize(query_en: str, answers: list[str], domain: str = "general") -> str:
    """
    Produce the final English answer.

    Parameters
    ----------
    query_en : str
        The user's question, in English.
    answers : list[str]
        Top-K retrieved answers (English). Empty for the General branch.
    domain : str
        "agro" | "banking" | "general" — used only to phrase the prompt.

    Returns
    -------
    str
        The synthesised English answer. On API failure, degrades to the
        joined raw answers (or an apology if there were none).
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query_en, answers, domain)},
        ],
        "temperature": GROQ_TEMPERATURE,
        "max_tokens": GROQ_MAX_TOKENS,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        logger.info("Groq synthesis successful (%d chars).", len(answer))
        return answer
    except Exception as exc:
        logger.error("Groq synthesis failed: %s", exc)
        if answers:
            return "\n".join(answers)
        return "Sorry, I couldn't generate an answer right now. Please try again."
