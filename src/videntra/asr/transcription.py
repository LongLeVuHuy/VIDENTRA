"""Speech transcription interfaces with PhoWhisper and deterministic mock implementations."""
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Sequence
import numpy as np

from videntra.evidence.normalize import normalize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    confidence: float
    language: str = "vi"
    status: str = "ok"


class ASRTranscriber:
    """Base speech-to-text transcriber interface."""

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        raise NotImplementedError


class MockASRTranscriber(ASRTranscriber):
    """Deterministic ASR transcriber for test cases and fast pipelines."""

    def __init__(self, default_text: str = "Cảm ơn bạn rất nhiều", confidence: float = 0.94, language: str = "vi"):
        self.default_text = default_text
        self.confidence = confidence
        self.language = language

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        return TranscriptResult(
            text=normalize_text(self.default_text),
            confidence=self.confidence,
            language=self.language,
        )


class PhoWhisperTranscriber(ASRTranscriber):
    """Vietnamese specialist ASR model using PhoWhisper-small (vinai/PhoWhisper-small)."""

    DEFAULT_MODEL_ID = "vinai/PhoWhisper-small"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        local_files_only: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        env_asr = os.environ.get("VIDENTRA_ASR_MODEL_DIR")
        candidate_paths = [
            self.model_path,
            Path(env_asr) if env_asr else None,
            Path("models/asr/phowhisper_small"),
        ]
        target_path = None
        for c in candidate_paths:
            if c and c.is_dir() and (c / "config.json").exists():
                target_path = c
                break

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        if target_path is not None:
            self._processor = WhisperProcessor.from_pretrained(str(target_path), local_files_only=True)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                str(target_path), local_files_only=True
            ).to(self.device)
        elif not self.local_files_only and self.model_id:
            kwargs = {"revision": self.revision} if self.revision else {}
            self._processor = WhisperProcessor.from_pretrained(self.model_id, **kwargs)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_id, **kwargs).to(self.device)
        else:
            raise FileNotFoundError("PhoWhisper model checkpoint directory not found")

        self._model.eval()

    def transcribe_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        """Transcribe single-channel float32 audio chunk."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return TranscriptResult(text="", confidence=0.0, language="vi", status="empty")

        try:
            self._ensure_loaded()
            import torch

            inputs = self._processor(
                audio_chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            with torch.no_grad():
                predicted_ids = self._model.generate(input_features, language="vi", task="transcribe")

            transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            clean_text = normalize_text(transcription)

            return TranscriptResult(
                text=clean_text,
                confidence=0.90,
                language="vi",
                status="ok",
            )
        except Exception as exc:
            logger.warning("PhoWhisper transcription failed: %s", exc)
            return TranscriptResult(
                text="",
                confidence=0.0,
                language="vi",
                status="error",
            )
