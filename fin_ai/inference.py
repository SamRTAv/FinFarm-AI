"""
fin_ai/inference.py

Loads the fine-tuned BERT intent classifier and exposes prediction helpers.

Usage:
    from fin_ai.inference import BankingIntentClassifier
    clf = BankingIntentClassifier()
    intent = clf.predict_label("I want to activate my card")
    top = clf.predict_with_probs("What is a debit card?")
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

from config.settings import BERT_OUTPUT_DIR
from shared.utils import get_logger

logger = get_logger(__name__)


class BankingIntentClassifier:
    """
    Inference wrapper for the fine-tuned BERT banking intent classifier.

    Parameters
    ----------
    model_dir : str
        Path to saved model directory.
    id2label : dict[int, str] | None
        Optional label mapping; supply after training.
    """

    def __init__(self, model_dir: str = BERT_OUTPUT_DIR, id2label: dict | None = None):
        logger.info("Loading BERT intent classifier from %s", model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._id2label: dict[int, str] = id2label or {}

    def set_label_mapping(self, id2label: dict[int, str]):
        self._id2label = id2label

    def id_to_label(self, idx: int) -> str:
        return self._id2label.get(idx, str(idx))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _get_probs(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return F.softmax(logits, dim=1).squeeze()

    def predict(self, text: str) -> int:
        """Return predicted class index."""
        return torch.argmax(self._get_probs(text)).item()

    def predict_label(self, text: str) -> str:
        """Return predicted intent string (requires set_label_mapping)."""
        return self.id_to_label(self.predict(text))

    def predict_with_probs(self, text: str) -> tuple[int, list[float]]:
        """Return (predicted_class_id, all_probabilities)."""
        probs = self._get_probs(text)
        return torch.argmax(probs).item(), probs.tolist()

    def predict_all_intents(self, text: str) -> list[tuple[str, float]]:
        """
        Return all intents sorted by probability descending.

        Returns
        -------
        list[tuple[str, float]]
            (intent_label, probability) pairs.
        """
        probs = self._get_probs(text).tolist()
        results = [(self.id_to_label(i), p) for i, p in enumerate(probs)]
        return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    clf = BankingIntentClassifier()
    for q in [
        "I would like to activate a card, can you help me?",
        "I forgot my ATM PIN, what should I do?",
        "What is debit and credit card",
    ]:
        print(f"Q: {q}")
        print(f"  → Class ID: {clf.predict(q)}\n")
