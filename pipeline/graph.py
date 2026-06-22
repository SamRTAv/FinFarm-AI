"""
pipeline/graph.py

The complete FinFarm inference pipeline as a LangGraph StateGraph — the entire
"Currently Achieved Architecture" diagram in one compiled graph.

Flow
----
                              ┌── agro ───►  agro_classify ─► agro_retrieve ─┐
START ─► detect_language ─►   │                                             │
        translate_query ──►(route by domain) ── banking ─► banking_classify ─► banking_retrieve ─┤
                              │                                             │
                              └── general ─► general_prepare ───────────────┤
                                                                            ▼
                                                                       synthesize (Groq)
                                                                            │
                                                                       back_translate
                                                                            │
                                                                           END

Key design points
-----------------
* The domain (agro / banking / general) is chosen by the USER and supplied as
  input — there is no automatic domain router (matches the diagram's
  "Asking from the user for any one of them").
* Language is detected ONCE up front. Every branch needs the source language:
  banking/general translate the query to English; agro classifies natively but
  still needs the language to back-translate the final answer.
* The two branches differ in *when* they translate (matching the existing code):
    - agro:    classify native text → retrieve → translate ANSWERS to English
    - banking: translate QUERY to English → classify → retrieve (already English)
* synthesize + back_translate are the shared tail for all three branches.

Usage:
    from pipeline.graph import run
    out = run("மழைக்கு பிறகு நெல் எப்போது விதைக்க வேண்டும்?", domain="agro",
              state_name="TAMILNADU")
    print(out["final_answer"])
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, START, END

from config.settings import TOP_K_FINAL, COSINE_THRESHOLD
from shared import registry, groq_client
from shared.translation import ENGLISH_CODE
from shared.utils import get_logger

logger = get_logger(__name__)

Domain = Literal["agro", "banking", "general"]


# ---------------------------------------------------------------------------
# Shared state passed between nodes
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # --- inputs ---
    query: str                 # raw user query, original language
    domain: Domain             # user-selected branch
    state_name: Optional[str]  # optional KCC state filter for the agro branch

    # --- detection / translation ---
    detected_lang: str         # IndicTrans2 code, e.g. "tam_Taml" / "eng_Latn"
    confidence: float
    query_en: str              # query translated to English (for Groq + banking)

    # --- agro branch ---
    labels: list[str]          # predicted KCC query-types

    # --- banking branch ---
    intent_id: int
    intent: str

    # --- convergence ---
    answers_en: list[str]      # Top-K retrieved answers, in English
    english_answer: str        # Groq-synthesised answer (English)
    final_answer: str          # back-translated to the original language


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def detect_language(state: PipelineState) -> dict:
    lang, conf = registry.get_translation().detect(state["query"])
    logger.info("Detected language=%s confidence=%.2f", lang, conf)
    return {"detected_lang": lang, "confidence": conf}


def translate_query(state: PipelineState) -> dict:
    """Translate the query to English (no-op if already English). Needed by
    the Groq prompt in every branch, and by the banking classifier."""
    ts = registry.get_translation()
    return {"query_en": ts.to_english(state["query"], state["detected_lang"])}


def route_domain(state: PipelineState) -> str:
    """Conditional edge: send the query down the user-selected branch."""
    return state.get("domain", "general")


# ---- Agro branch -----------------------------------------------------------

def agro_classify(state: PipelineState) -> dict:
    """Classify the NATIVE query text with IndicBERT (multilingual)."""
    clf = registry.get_farm_classifier()
    labels = clf.predict_top_labels(state["query"], threshold=COSINE_THRESHOLD)
    logger.info("Agro predicted labels: %s", labels)
    return {"labels": labels}


def agro_retrieve(state: PipelineState) -> dict:
    """Filter the KCC corpus by predicted query-types (+ optional state),
    rank by cosine similarity, then translate the Top-K answers to English."""
    df = registry.get_kcc_df()
    sub = df[df["QueryType"].isin(state.get("labels", []))]
    if state.get("state_name"):
        sub = sub[sub["StateName"] == state["state_name"]]

    if sub.empty:
        logger.warning("Agro: no KCC rows matched labels=%s state=%s",
                       state.get("labels"), state.get("state_name"))
        return {"answers_en": []}

    top = registry.get_farm_retriever().retrieve(state["query"], sub, text_col="KccAns")
    answers_native = top["KccAns"].head(TOP_K_FINAL).tolist()
    answers_en = registry.get_translation().to_english_batch(
        answers_native, state["detected_lang"]
    )
    return {"answers_en": answers_en}


# ---- Banking branch --------------------------------------------------------

def banking_classify(state: PipelineState) -> dict:
    """Classify the ENGLISH query with the fine-tuned BERT intent model."""
    clf = registry.get_banking_classifier()
    pid = clf.predict(state["query_en"])
    intent = clf.id_to_label(pid)
    logger.info("Banking predicted intent=%s (id=%d)", intent, pid)
    return {"intent_id": pid, "intent": intent}


def banking_retrieve(state: PipelineState) -> dict:
    """Retrieve the Top-K most similar banking responses for the intent
    (responses are already in English)."""
    answers = registry.get_banking_retriever().retrieve(
        state["query_en"], state["intent_id"]
    )[:TOP_K_FINAL]
    return {"answers_en": answers}


# ---- General branch --------------------------------------------------------

def general_prepare(state: PipelineState) -> dict:
    """No retrieval: Groq answers the translated query from its own knowledge."""
    return {"answers_en": []}


# ---- Convergence -----------------------------------------------------------

def synthesize(state: PipelineState) -> dict:
    answer = groq_client.synthesize(
        state["query_en"], state.get("answers_en", []), state.get("domain", "general")
    )
    return {"english_answer": answer}


def back_translate(state: PipelineState) -> dict:
    """Translate the English answer back to the user's original language."""
    if state["detected_lang"] == ENGLISH_CODE:
        return {"final_answer": state["english_answer"]}
    final = registry.get_translation().to_indic(
        state["english_answer"], state["detected_lang"]
    )
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    g = StateGraph(PipelineState)

    g.add_node("detect_language", detect_language)
    g.add_node("translate_query", translate_query)
    g.add_node("agro_classify", agro_classify)
    g.add_node("agro_retrieve", agro_retrieve)
    g.add_node("banking_classify", banking_classify)
    g.add_node("banking_retrieve", banking_retrieve)
    g.add_node("general_prepare", general_prepare)
    g.add_node("synthesize", synthesize)
    g.add_node("back_translate", back_translate)

    g.add_edge(START, "detect_language")
    g.add_edge("detect_language", "translate_query")

    g.add_conditional_edges(
        "translate_query",
        route_domain,
        {
            "agro": "agro_classify",
            "banking": "banking_classify",
            "general": "general_prepare",
        },
    )

    # agro branch
    g.add_edge("agro_classify", "agro_retrieve")
    g.add_edge("agro_retrieve", "synthesize")
    # banking branch
    g.add_edge("banking_classify", "banking_retrieve")
    g.add_edge("banking_retrieve", "synthesize")
    # general branch
    g.add_edge("general_prepare", "synthesize")

    # shared tail
    g.add_edge("synthesize", "back_translate")
    g.add_edge("back_translate", END)

    return g.compile()


# Module-level compiled graph (built lazily on first run()).
_GRAPH = None


def run(query: str, domain: Domain, state_name: Optional[str] = None) -> dict:
    """
    Execute the full pipeline for a single query.

    Parameters
    ----------
    query : str
        User query in any supported language.
    domain : "agro" | "banking" | "general"
        The branch the user selected.
    state_name : str | None
        Optional KCC state filter (agro branch only), e.g. "TAMILNADU".

    Returns
    -------
    dict
        The final PipelineState, including "final_answer".
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    init: PipelineState = {"query": query, "domain": domain}
    if state_name:
        init["state_name"] = state_name
    return _GRAPH.invoke(init)


if __name__ == "__main__":
    demo = run(
        "வரவிருக்கும் குளிர்காலத்தில் எந்த மாதத்தில் நெல் விதைகளை விதைக்க வேண்டும்?",
        domain="agro",
        state_name="TAMILNADU",
    )
    print("Detected:", demo.get("detected_lang"))
    print("Labels:", demo.get("labels"))
    print("Answer:", demo.get("final_answer"))
