"""
shared/translation.py

Unified language services for the FinFarm pipeline:

  - IndicLID language *detection*
  - IndicTrans2 translation in BOTH directions:
        indic -> en   (1B model      — input fidelity)
        en   -> indic  (distilled 200M — fast/low-RAM back-translation)

A single IndicLID instance and a single IndicProcessor are shared across
both translation directions. This module replaces the duplicated IndicLID
patching that previously lived in farm_gpu/ and fin_ai/ language_detection.py.

Usage:
    from shared.translation import TranslationService
    ts = TranslationService()
    lang, conf = ts.detect("मुझे अपना ATM पिन भूल गया")
    en   = ts.to_english("मुझे अपना ATM पिन भूल गया", lang)
    back = ts.to_indic("You can reset your ATM PIN at any branch.", lang)
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
    INDICTRANS2_EN_INDIC_MODEL,
    INDICTRANS2_MAX_LENGTH,
    INDICTRANS2_NUM_BEAMS,
)
from shared.utils import get_logger

logger = get_logger(__name__)

# IndicTrans2 / FLORES code for English.
ENGLISH_CODE = "eng_Latn"


def _load_indiclid(input_threshold: float, roman_threshold: float):
    """
    Build an IndicLID instance with the BERT model pre-loaded, side-stepping
    the `weights_only=True` unpickling error on newer PyTorch versions.
    """
    torch.serialization.add_safe_globals([BertForSequenceClassification, BertModel])
    logger.info("Loading IndicLID BERT from %s", INDICLID_BERT_PATH)
    lid_model = torch.load(INDICLID_BERT_PATH, weights_only=False)

    from ai4bharat.IndicLID import IndicLID
    import fasttext

    class _PatchedIndicLID(IndicLID):
        """IndicLID with a pre-loaded BERT model to avoid unpickling errors."""

        def __init__(self_):
            self_.input_threshold = input_threshold
            self_.roman_lid_threshold = roman_threshold
            self_.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self_.IndicLID_FTN = fasttext.load_model(INDICLID_FTN_PATH)
            self_.IndicLID_FTR = fasttext.load_model(INDICLID_FTR_PATH)
            self_.IndicLID_BERT = lid_model
            self_.IndicLID_BERT.eval()
            self_.IndicLID_BERT_tokenizer = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicBERTv2-MLM-only"
            )

    return _PatchedIndicLID()


class TranslationService:
    """
    Detection + bidirectional translation, loaded once and reused.

    All heavy models are loaded in __init__, so construct this exactly once
    (see shared.registry.get_translation()).
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # ---- IndicLID (language detection) ----
        self.lid = _load_indiclid(INDICLID_INPUT_THRESHOLD, INDICLID_ROMAN_THRESHOLD)

        # ---- Shared IndicTrans2 pre/post processor ----
        from IndicTransToolkit.processor import IndicProcessor
        self.ip = IndicProcessor(inference=True)

        # ---- indic -> en (1B) ----
        logger.info("Loading IndicTrans2 indic->en: %s", INDICTRANS2_MODEL)
        self.i2e_tokenizer = AutoTokenizer.from_pretrained(
            INDICTRANS2_MODEL, trust_remote_code=True
        )
        self.i2e_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                INDICTRANS2_MODEL, trust_remote_code=True, torch_dtype=self._dtype
            )
            .to(self.device)
            .eval()
        )

        # ---- en -> indic (distilled 200M) ----
        logger.info("Loading IndicTrans2 en->indic: %s", INDICTRANS2_EN_INDIC_MODEL)
        self.e2i_tokenizer = AutoTokenizer.from_pretrained(
            INDICTRANS2_EN_INDIC_MODEL, trust_remote_code=True
        )
        self.e2i_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                INDICTRANS2_EN_INDIC_MODEL, trust_remote_code=True, torch_dtype=self._dtype
            )
            .to(self.device)
            .eval()
        )
        logger.info("TranslationService ready.")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> tuple[str, float]:
        """Return (language_code, confidence) for a single text."""
        results = self.lid.batch_predict([text], batch_size=1)
        return results[0][1], results[0][2]

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _translate(self, texts: list[str], src: str, tgt: str, tokenizer, model) -> list[str]:
        batch = self.ip.preprocess_batch(texts, src_lang=src, tgt_lang=tgt)
        inputs = tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=INDICTRANS2_MAX_LENGTH,
                num_beams=INDICTRANS2_NUM_BEAMS,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return self.ip.postprocess_batch(decoded, lang=tgt)

    def to_english(self, text: str, src_lang: str) -> str:
        """Translate a single Indic string to English (pass-through if already English)."""
        if src_lang == ENGLISH_CODE:
            return text
        return self._translate([text], src_lang, ENGLISH_CODE, self.i2e_tokenizer, self.i2e_model)[0]

    def to_english_batch(self, texts: list[str], src_lang: str) -> list[str]:
        """Batch variant of to_english (used for translating retrieved KCC answers)."""
        if not texts or src_lang == ENGLISH_CODE:
            return texts
        return self._translate(texts, src_lang, ENGLISH_CODE, self.i2e_tokenizer, self.i2e_model)

    def to_indic(self, text: str, tgt_lang: str) -> str:
        """Translate a single English string back to the target Indic language."""
        if tgt_lang == ENGLISH_CODE:
            return text
        return self._translate([text], ENGLISH_CODE, tgt_lang, self.e2i_tokenizer, self.e2i_model)[0]


if __name__ == "__main__":
    ts = TranslationService()
    sample = "मुझे अपना ATM पिन भूल गया"
    lang, conf = ts.detect(sample)
    print(f"detected: {lang} ({conf:.2f})")
    en = ts.to_english(sample, lang)
    print("en:", en)
    print("back:", ts.to_indic(en, lang))
