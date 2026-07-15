"""Breeze-ASR-25 transcription with optional CAM++ diarization."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ownscribe.config import resolve_model_path
from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult
from ownscribe.transcription.utils import build_speaker_timeline, load_audio_mono, quiet_model_output

_SAMPLE_RATE = 16000


class BreezeTranscriber(Transcriber):
    """Transcribe Taiwanese Mandarin and code-switching audio with Breeze-ASR-25."""

    def __init__(
        self,
        progress: NullProgress | None = None,
        use_mps: bool = True,
        diarization_enabled: bool = False,
        models_dir: str | None = None,
        speaker_threshold: float = 0.7,
    ) -> None:
        self._progress = progress or NullProgress()
        self._use_mps = use_mps
        self._diarization_enabled = diarization_enabled
        self._models_dir = models_dir
        self._speaker_threshold = speaker_threshold
        self._model = None
        self._processor = None
        self._device = None
        self._vad_model = None
        self._spk_model = None

    def _set_detail(self, key: str, text: str | None) -> None:
        setter = getattr(self._progress, "set_detail", None)
        if callable(setter):
            setter(key, text)

    def _load_spk_models(self) -> None:
        from funasr import AutoModel

        vad = resolve_model_path("fsmn-vad", self._models_dir)
        speaker = resolve_model_path("campplus", self._models_dir)
        with quiet_model_output():
            self._vad_model = AutoModel(model=str(vad), disable_update=True)
            self._spk_model = AutoModel(model=str(speaker), disable_update=True)

    def _load_asr_model(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model = str(resolve_model_path("breeze-asr-25", self._models_dir))
        self._device = "mps" if self._use_mps and torch.backends.mps.is_available() else "cpu"
        self._processor = WhisperProcessor.from_pretrained(model)
        self._model = WhisperForConditionalGeneration.from_pretrained(model).to(self._device).eval()

    def prepare_models(self, language: str | None = None) -> None:
        progress = self._progress
        progress.begin("preparing_models")
        try:
            if self._diarization_enabled:
                self._set_detail("preparing_models", "Loading CAM++ speaker model...")
                self._load_spk_models()
            self._set_detail("preparing_models", "Loading Breeze-ASR-25...")
            self._load_asr_model()
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        os.environ.setdefault("OMP_NUM_THREADS", "12")
        audio, sample_rate = load_audio_mono(audio_path, _SAMPLE_RATE)
        progress = self._progress

        if self._model is None:
            self.prepare_models()

        timeline = []
        if self._diarization_enabled:
            progress.begin("diarizing")
            self._set_detail("diarizing", "Identifying speakers (CAM++)...")
            timeline = self._run_diarization(audio, sample_rate)
            progress.complete("diarizing")

        progress.begin("transcribing")
        device_label = str(self._device).upper()
        self._set_detail("transcribing", f"Transcribing (Breeze-ASR-25, {device_label})...")
        segments = self._transcribe_chunked(audio, sample_rate)
        progress.complete("transcribing")
        self._align_speakers(segments, timeline)
        return TranscriptResult(segments=segments, language="zh-TW", duration=len(audio) / sample_rate)

    def _transcribe_chunked(self, audio: np.ndarray, sample_rate: int) -> list[Segment]:
        import torch

        chunk_samples = 30 * sample_rate
        segments: list[Segment] = []
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            if len(chunk) < 3200:
                continue
            features = self._processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features
            with torch.inference_mode():
                predicted = self._model.generate(
                    features.to(self._device), return_timestamps=True, language="zh", task="transcribe"
                )
            decoded = self._processor.batch_decode(predicted, skip_special_tokens=False, decode_with_timestamps=True)[0]
            offset_seconds = offset / sample_rate
            for match in re.finditer(r"<\|([\d.]+)\|>(.*?)<\|([\d.]+)\|>", decoded):
                text = match.group(2).strip()
                if text and not text.startswith("<|"):
                    segments.append(
                        Segment(
                            text=text,
                            start=float(match.group(1)) + offset_seconds,
                            end=float(match.group(3)) + offset_seconds,
                            words=[],
                        )
                    )
        return segments

    def _run_diarization(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        with quiet_model_output():
            result = self._vad_model.generate(input=audio)
        vad_segments = result[0]["value"]
        embeddings: list[np.ndarray | None] = []
        for start_ms, end_ms in vad_segments:
            chunk = audio[int(start_ms / 1000 * sample_rate) : int(end_ms / 1000 * sample_rate)]
            if len(chunk) < 1600:
                embeddings.append(None)
                continue
            with quiet_model_output():
                result = self._spk_model.generate(input=chunk, input_len=np.array([len(chunk)]))
            embeddings.append(np.asarray(result[0]["spk_embedding"]).reshape(-1))
        return build_speaker_timeline(vad_segments, embeddings, self._speaker_threshold)

    def _align_speakers(self, segments: list[Segment], timeline: list[dict]) -> None:
        for segment in segments:
            segment.speaker = self._find_speaker(timeline, segment.start * 1000, segment.end * 1000)

    @staticmethod
    def _find_speaker(timeline: list[dict], start_ms: float, end_ms: float) -> str | None:
        labeled = [item for item in timeline if item["speaker"] is not None]
        if not labeled:
            return None
        overlaps = [max(0, min(end_ms, item["end_ms"]) - max(start_ms, item["start_ms"])) for item in labeled]
        if max(overlaps) > 0:
            return labeled[int(np.argmax(overlaps))]["speaker"]
        midpoint = (start_ms + end_ms) / 2
        return min(labeled, key=lambda item: abs(midpoint - (item["start_ms"] + item["end_ms"]) / 2))["speaker"]
