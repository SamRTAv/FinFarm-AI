"""
farm_gpu/language_detection.py

Wraps the IndicLID language-identification model using a patched loader
that avoids the weights_only unpickling error on newer PyTorch versions.

Usage:
    from farm_gpu.language_detection import IndicLIDDetector
    detector = IndicLIDDetector()
    lang_code = detector.detect("நெல் விதைகளை எப்போது விதைக்க வேண்டும்?")
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("./IndicLID/Inference")

import torch
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel

from config.settings import (
    INDICLID_INPUT_THRESHOLD,
    INDICLID_ROMAN_THRESHOLD,
    INDICLID_FTN_PATH,
    INDICLID_FTR_PATH,
    INDICLID_BERT_PATH,
)
from shared.utils import get_logger

logger = get_logger(__name__)


class IndicLIDDetector:
    """
    Language identifier using the IndicLID model.

    Patches torch.load to use weights_only=False for legacy model files.
    """

    def __init__(
        self,
        input_threshold: float = INDICLID_INPUT_THRESHOLD,
        roman_lid_threshold: float = INDICLID_ROMAN_THRESHOLD,
    ):
        torch.serialization.add_safe_globals([BertForSequenceClassification, BertModel])

        logger.info("Loading IndicLID BERT model from %s", INDICLID_BERT_PATH)
        lid_model = torch.load(INDICLID_BERT_PATH, weights_only=False)

        from ai4bharat.IndicLID import IndicLID
        import fasttext

        class _PatchedIndicLID(IndicLID):
            """IndicLID with pre-loaded BERT model to avoid unpickling errors."""

            def __init__(self_, threshold, roman_threshold):
                self_.input_threshold = threshold
                self_.roman_lid_threshold = roman_threshold
                self_.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                self_.IndicLID_FTN = fasttext.load_model(INDICLID_FTN_PATH)
                self_.IndicLID_FTR = fasttext.load_model(INDICLID_FTR_PATH)

                self_.IndicLID_BERT = lid_model
                self_.IndicLID_BERT.eval()
                self_.IndicLID_BERT_tokenizer = AutoTokenizer.from_pretrained(
                    "ai4bharat/IndicBERTv2-MLM-only"
                )

        self._lid = _PatchedIndicLID(input_threshold, roman_lid_threshold)
        logger.info("IndicLID loaded successfully.")

    def detect(self, text: str) -> str:
        """
        Detect the language of a single text string.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
            Detected language code (e.g. 'tam_Taml', 'hin_Deva').
        """
        results = self._lid.batch_predict([text], batch_size=1)
        return results[0][1]

    def detect_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Detect language and return confidence score.

        Returns
        -------
        tuple[str, float]
            (language_code, confidence)
        """
        results = self._lid.batch_predict([text], batch_size=1)
        return results[0][1], results[0][2]

    def batch_detect(self, texts: list[str], batch_size: int = 8) -> list[str]:
        """
        Detect languages for a list of texts.

        Returns
        -------
        list[str]
            Language codes in the same order as input.
        """
        results = self._lid.batch_predict(texts, batch_size)
        return [r[1] for r in results]


if __name__ == "__main__":
    detector = IndicLIDDetector()
    samples = [
        "आज के दिन का मौसम अत्यंत सुंदर है",
        "வரவிருக்கும் குளிர்காலத்தில் எந்த மாதத்தில்",
    ]
    for s in samples:
        lang, conf = detector.detect_with_confidence(s)
        print(f"[{lang} | {conf:.2f}] {s[:40]}")
