"""
farm_gpu/inference.py

Loads the fine-tuned IndicBERT model and provides functions to:
  - predict a single query-type label
  - return full softmax probability distribution
  - return top-k labels above a confidence threshold

Usage:
    from farm_gpu.inference import FarmQueryClassifier
    clf = FarmQueryClassifier()
    label = clf.predict("நெல் விதைகளை எப்போது விதைக்க வேண்டும்?")
    top_labels = clf.predict_top(text, threshold=0.09)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import INDICBERT_OUTPUT_DIR, COSINE_THRESHOLD
from shared.utils import get_logger

logger = get_logger(__name__)


class FarmQueryClassifier:
    """
    Wrapper around the fine-tuned IndicBERT model for inference.

    Parameters
    ----------
    model_dir : str
        Path to the saved model directory (defaults to INDICBERT_OUTPUT_DIR).
    id2label : dict[int, str] | None
        Mapping from class index to label string.
        If None, uses model.config.id2label.
    """

    def __init__(self, model_dir: str = INDICBERT_OUTPUT_DIR, id2label: dict | None = None):
        logger.info("Loading IndicBERT from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._id2label: dict[int, str] = id2label or {}

    # ------------------------------------------------------------------
    # Label mapping helpers
    # ------------------------------------------------------------------

    def set_label_mapping(self, id2label: dict[int, str]):
        """Attach a label mapping built from the training DataFrame."""
        self._id2label = id2label

    def id_to_label(self, idx: int) -> str:
        return self._id2label.get(idx, str(idx))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _get_probs(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return F.softmax(logits, dim=1).squeeze()

    def predict(self, text: str) -> int:
        """
        Return the predicted class index.

        Parameters
        ----------
        text : str

        Returns
        -------
        int
            Predicted class index.
        """
        probs = self._get_probs(text)
        return torch.argmax(probs).item()

    def predict_label(self, text: str) -> str:
        """Return the predicted label string (requires set_label_mapping)."""
        return self.id_to_label(self.predict(text))

    def predict_with_probs(self, text: str) -> tuple[int, list[float]]:
        """
        Return (predicted_class_id, list_of_all_probabilities).
        """
        probs = self._get_probs(text)
        predicted_id = torch.argmax(probs).item()
        return predicted_id, probs.tolist()

    def predict_top(self, text: str, threshold: float = COSINE_THRESHOLD) -> list[tuple[int, float]]:
        """
        Return all (class_id, probability) pairs whose probability exceeds threshold,
        sorted descending.

        Parameters
        ----------
        text : str
        threshold : float

        Returns
        -------
        list[tuple[int, float]]
        """
        probs = self._get_probs(text)
        sorted_probs = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)
        return [(idx, prob) for idx, prob in sorted_probs if prob > threshold]

    def predict_top_labels(self, text: str, threshold: float = COSINE_THRESHOLD) -> list[str]:
        """Return the top label strings above the threshold."""
        return [self.id_to_label(idx) for idx, _ in self.predict_top(text, threshold)]


if __name__ == "__main__":
    clf = FarmQueryClassifier()
    test_query = "வரவிருக்கும் குளிர்காலத்தில் எந்த மாதத்தில் நெல் விதைகளை விதைக்க வேண்டும்?"
    pred_id, probs = clf.predict_with_probs(test_query)
    print("Predicted class ID:", pred_id)
    print("Top predictions:", clf.predict_top(test_query))
