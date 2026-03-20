"""
fin_ai/audio_transcription.py

Transcribes or translates audio files to English text using OpenAI Whisper
(openai/whisper-large-v2) via HuggingFace Transformers.

Two modes are supported:
  1. Auto-detect language → translate to English  (task="translate")
  2. Force translate from a specific language (e.g. Hindi) to English

Usage:
    from fin_ai.audio_transcription import WhisperTranscriber
    transcriber = WhisperTranscriber()
    text = transcriber.transcribe_and_translate("audio.wav")
    text = transcriber.translate_from_language("audio.wav", language="hi")
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config.settings import WHISPER_MODEL, WHISPER_TARGET_SAMPLE_RATE
from shared.utils import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    """
    Wrapper around Whisper for audio transcription and translation.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: openai/whisper-large-v2).
    """

    def __init__(self, model_name: str = WHISPER_MODEL):
        logger.info("Loading Whisper: %s", model_name)
        self._model_name = model_name
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and resample to 16kHz mono."""
        audio_input, sample_rate = torchaudio.load(audio_path)
        if sample_rate != WHISPER_TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=WHISPER_TARGET_SAMPLE_RATE
            )
            audio_input = resampler(audio_input)
        return audio_input.squeeze()

    def transcribe_and_translate(self, audio_path: str) -> str:
        """
        Auto-detect the language in the audio and translate to English.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (.wav, .mp3, etc.)

        Returns
        -------
        str
            English translation of the audio content.
        """
        processor = WhisperProcessor.from_pretrained(
            self._model_name, language="en", task="translate"
        )
        audio = self._load_audio(audio_path)
        inputs = processor(
            audio, sampling_rate=WHISPER_TARGET_SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs["attention_mask"],
            )
        translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info("Translated text: %s", translation[:100])
        return translation

    def translate_from_language(self, audio_path: str, language: str = "hi") -> str:
        """
        Force translation from a specific language to English.

        Parameters
        ----------
        audio_path : str
        language : str
            ISO 639-1 language code (e.g. 'hi' for Hindi, 'ta' for Tamil).

        Returns
        -------
        str
            English translation.
        """
        audio = self._load_audio(audio_path)
        inputs = self.processor(
            audio, sampling_rate=WHISPER_TARGET_SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="translate"
        )
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                forced_decoder_ids=forced_decoder_ids,
            )
        translation = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info("Forced translation (%s→en): %s", language, translation[:100])
        return translation


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_transcription.py <audio_file>")
        sys.exit(1)
    transcriber = WhisperTranscriber()
    result = transcriber.transcribe_and_translate(sys.argv[1])
    print("Translation:", result)
