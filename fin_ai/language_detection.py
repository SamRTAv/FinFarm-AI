"""
fin_ai/language_detection.py

Detects the Indic language of a query (using IndicLID) and translates it
to English (using IndicTrans2 indic-en-1B), enabling the rest of the
FinAI pipeline to operate on English text regardless of input language.

Usage:
    from fin_ai.language_detection import AutoTranslator
    translator = AutoTranslator()
    result = translator.detect_and_translate("मुझे अपना ATM पिन भूल गया")
    print(result["translation"])
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("./IndicTransToolkit")
sys.path.append("./IndicLID/Inference")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel

from config.settings import (
    INDICLID_BERT_PATH,
    INDICLID_FTN_PATH,
    INDICLID_FTR_PATH,
    INDICLID_INPUT_THRESHOLD,
    INDICLID_ROMAN_THRESHOLD,
    INDICTRANS2_MODEL,
    INDICTRANS2_MAX_LENGTH,
    INDICTRANS2_NUM_BEAMS,
    SUPPORTED_INDIC_LANGS,
)
from shared.utils import get_logger

logger = get_logger(__name__)


class AutoTranslator:
    """
    Language detection + translation pipeline for Indic languages → English.

    Combines:
      - IndicLID for language identification
      - IndicTrans2 (indic-en-1B) for translation
    """

    def __init__(self):
        # ---- Patch torch.load for legacy model files ----
        torch.serialization.add_safe_globals([BertForSequenceClassification, BertModel])
        lid_model = torch.load(INDICLID_BERT_PATH, weights_only=False)

        # ---- Build patched IndicLID ----
        from ai4bharat.IndicLID import IndicLID
        import fasttext

        class _PatchedIndicLID(IndicLID):
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

        self.lid = _PatchedIndicLID(INDICLID_INPUT_THRESHOLD, INDICLID_ROMAN_THRESHOLD)

        # ---- Load IndicTrans2 ----
        logger.info("Loading IndicTrans2: %s", INDICTRANS2_MODEL)
        from IndicTransToolkit.processor import IndicProcessor

        self.tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_MODEL, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS2_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.ip = IndicProcessor(inference=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.lang_mapping = SUPPORTED_INDIC_LANGS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_translate(self, text: str) -> dict:
        """
        Auto-detect the language and translate to English.

        Parameters
        ----------
        text : str
            Input text in any supported Indic language.

        Returns
        -------
        dict with keys:
            status : "success" | "unsupported_language" | "low_confidence" | "error"
            detected_lang : str
            confidence : float
            translation : str
        """
        try:
            results = self.lid.batch_predict([text], batch_size=1)
            detected_lang = results[0][1]
            confidence = results[0][2]

            if detected_lang not in self.lang_mapping:
                return {
                    "status": "unsupported_language",
                    "detected_lang": detected_lang,
                    "confidence": confidence,
                    "translation": f"Unsupported language: {detected_lang}",
                }
            if confidence < INDICLID_INPUT_THRESHOLD:
                return {
                    "status": "low_confidence",
                    "detected_lang": detected_lang,
                    "confidence": confidence,
                    "translation": "Low confidence — cannot translate reliably.",
                }

            translation = self.translate_single(text, detected_lang)
            return {
                "status": "success",
                "detected_lang": detected_lang,
                "confidence": confidence,
                "translation": translation,
            }
        except Exception as exc:
            return {"status": "error", "detected_lang": "", "confidence": 0.0, "translation": str(exc)}

    def translate_single(
        self,
        text: str,
        source_lang: str,
        max_length: int = INDICTRANS2_MAX_LENGTH,
        num_beams: int = INDICTRANS2_NUM_BEAMS,
    ) -> str:
        """
        Translate a single text from source_lang to English.

        Parameters
        ----------
        text : str
        source_lang : str
            IndicTrans2 language code (e.g. 'tam_Taml').
        max_length : int
        num_beams : int

        Returns
        -------
        str
            English translation.
        """
        try:
            batch = self.ip.preprocess_batch([text], src_lang=source_lang, tgt_lang="eng_Latn")
            inputs = self.tokenizer(
                batch, truncation=True, padding="longest", return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=1,
                )
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return self.ip.postprocess_batch(decoded, lang="eng_Latn")[0]
        except Exception as exc:
            return f"Translation error: {exc}"


if __name__ == "__main__":
    translator = AutoTranslator()
    samples = [
        "मुझे अपना ATM पिन भूल गया",
        "வரவிருக்கும் குளிர்காலத்தில் எந்த மாதத்தில் நெல் விதைகளை விதைக்க வேண்டும்?",
    ]
    for s in samples:
        result = translator.detect_and_translate(s)
        print(result)
