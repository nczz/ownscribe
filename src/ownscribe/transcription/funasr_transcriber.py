"""FunASR transcription using its native VAD, punctuation, and CAM++ pipeline."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ownscribe.config import resolve_model_path
from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult
from ownscribe.transcription.utils import quiet_model_output

if TYPE_CHECKING:
    from ownscribe.config import FunASRConfig

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "sensevoice": ("iic/SenseVoiceSmall", "sensevoice"),
    "paraformer": ("paraformer-zh", "paraformer-zh"),
    "paraformer-en": ("paraformer-en", "paraformer-en"),
}


class FunASRTranscriber(Transcriber):
    """Transcribe with FunASR and optionally enable its native CAM++ integration."""

    def __init__(self, config: FunASRConfig, progress: NullProgress | None = None) -> None:
        self._config = config
        self._progress = progress or NullProgress()
        self._model = None

    def _set_detail(self, key: str, text: str | None) -> None:
        setter = getattr(self._progress, "set_detail", None)
        if callable(setter):
            setter(key, text)

    def _resolve_model_name(self) -> str:
        model = self._config.model
        if model not in _MODEL_MAP:
            return model
        remote, local_name = _MODEL_MAP[model]
        resolved = resolve_model_path(local_name, self._config.models_dir)
        return str(resolved) if isinstance(resolved, Path) else remote

    def _resolve_component(self, local_name: str, remote: str) -> str:
        resolved = resolve_model_path(local_name, self._config.models_dir)
        return str(resolved) if isinstance(resolved, Path) else remote

    def _load_model(self) -> None:
        import warnings

        kwargs = {
            "model": self._resolve_model_name(),
            "vad_model": self._resolve_component("fsmn-vad", "fsmn-vad"),
            "vad_kwargs": {"max_single_segment_time": 30000},
            "punc_model": self._resolve_component("ct-punc", "ct-punc"),
            "device": self._config.device,
            "disable_update": True,
        }
        if self._config.spk_enabled:
            kwargs["spk_model"] = self._resolve_component("campplus", "cam++")
            kwargs["output_timestamp"] = True

        with warnings.catch_warnings(), quiet_model_output():
            warnings.simplefilter("ignore")
            from funasr import AutoModel

            self._model = AutoModel(**kwargs)

    def prepare_models(self, language: str | None = None) -> None:
        progress = self._progress
        progress.begin("preparing_models")
        try:
            self._set_detail("preparing_models", f"Loading FunASR model ({self._config.model})")
            self._load_model()
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        threads = str(max(1, min((os.cpu_count() or 8) - 2, 12)))
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        if self._model is None:
            self.prepare_models()

        kwargs = {
            "input": str(audio_path),
            "batch_size_s": self._config.batch_size_s,
            "use_itn": True,
            "merge_vad": True,
            "merge_length_s": 15,
        }
        if "sensevoice" in self._resolve_model_name().lower():
            kwargs["language"] = self._config.language or "auto"

        self._progress.begin("transcribing")
        self._set_detail("transcribing", "Transcribing audio...")
        try:
            with quiet_model_output():
                result = self._model.generate(**kwargs)
            self._progress.complete("transcribing")
        except Exception:
            self._progress.fail("transcribing")
            raise
        return self._convert_result(result, audio_path)

    def _to_traditional(self, text: str) -> str:
        if not self._config.traditional_chinese or not text:
            return text
        try:
            from opencc import OpenCC

            if not hasattr(self, "_cc"):
                self._cc = OpenCC("s2twp")
            return self._cc.convert(text)
        except ImportError:
            logger.warning("opencc is not installed; preserving original Chinese output")
            return text

    def _convert_result(self, result_list: list, audio_path: Path) -> TranscriptResult:
        import soundfile as sf
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        duration = sf.info(str(audio_path)).duration
        if not result_list or not result_list[0]:
            return TranscriptResult(segments=[], language="", duration=duration)
        result = result_list[0]
        segments: list[Segment] = []
        for sentence in result.get("sentence_info", []):
            text = sentence.get("sentence", sentence.get("text", "")).strip()
            text = self._to_traditional(rich_transcription_postprocess(text))
            if text:
                speaker_id = sentence.get("spk") if self._config.spk_enabled else None
                segments.append(
                    Segment(
                        text=text,
                        start=(sentence.get("start") or 0) / 1000,
                        end=(sentence.get("end") or 0) / 1000,
                        speaker=f"SPEAKER_{speaker_id}" if speaker_id is not None else None,
                        words=[],
                    )
                )
        if not segments:
            text = self._to_traditional(rich_transcription_postprocess(result.get("text", "")))
            timestamps = result.get("timestamp", [])
            if timestamps:
                segments = self._regroup_by_punctuation(text, timestamps)
            elif text:
                segments = [Segment(text=text.strip(), start=0, end=duration, words=[])]
        language = self._config.language or result.get("language", "zh")
        return TranscriptResult(segments=segments, language=language, duration=duration)

    @staticmethod
    def _regroup_by_punctuation(text: str, timestamps: list[list[int]]) -> list[Segment]:
        sentence_end = re.compile(r"[。？！；.?!;]")
        segments: list[Segment] = []
        current = ""
        start: float | None = None
        timestamp_index = 0
        for character in text.replace(" ", ""):
            if timestamp_index >= len(timestamps):
                current += character
                continue
            timestamp = timestamps[timestamp_index]
            start = timestamp[0] / 1000 if start is None else start
            current += character
            timestamp_index += 1
            if sentence_end.search(character) and current.strip():
                segments.append(Segment(text=current.strip(), start=start, end=timestamp[1] / 1000, words=[]))
                current, start = "", None
        if current.strip() and start is not None:
            segments.append(Segment(text=current.strip(), start=start, end=timestamps[-1][1] / 1000, words=[]))
        return segments
