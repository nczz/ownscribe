"""FunASR-based transcription with built-in speaker diarization (CAM++).

This backend replaces WhisperX + pyannote with FunASR's unified pipeline,
providing significantly better Chinese recognition (CER ~7.81% vs ~20%)
and built-in speaker diarization without requiring a HuggingFace token.

Supported models:
  - "sensevoice" (default): SenseVoice-Small — fastest, multilingual, emotion detection
  - "paraformer": Paraformer-zh — Chinese-only, character-level timestamps, hotword support
  - "paraformer-en": Paraformer-en — English-only
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult, Word

logger = logging.getLogger(__name__)

# Model name mapping: user-friendly name → (ModelScope ID, local dir name)
_MODEL_MAP = {
    "sensevoice": ("iic/SenseVoiceSmall", "sensevoice"),
    "paraformer": ("paraformer-zh", "paraformer-zh"),
    "paraformer-en": ("paraformer-en", "paraformer-en"),
}

_VAD_MAP = ("fsmn-vad", "fsmn-vad")
_PUNC_MAP = ("ct-punc", "ct-punc")
_SPK_MAP = ("cam++", "campplus")

_LOCAL_CACHE = Path("~/.cache/funasr/models").expanduser()
_SAMPLE_RATE = 16000


class FunASRTranscriber(Transcriber):
    """Transcribes audio using FunASR with optional CAM++ speaker diarization.

    Key advantages over WhisperX for Chinese:
    - CER ~7.81% (SenseVoice) vs ~20% (Whisper) on Chinese
    - Built-in speaker diarization via CAM++ (no HuggingFace token needed)
    - Non-autoregressive: ~17x realtime on CPU, ~170x on GPU
    - MIT licensed, all models freely downloadable
    """

    def __init__(
        self,
        config: "FunASRConfig",
        progress: NullProgress | None = None,
    ) -> None:
        self._config = config
        self._progress = progress or NullProgress()
        self._model = None

    def _set_detail(self, key: str, text: str | None) -> None:
        set_detail = getattr(self._progress, "set_detail", None)
        if callable(set_detail):
            set_detail(key, text)

    def _resolve_model_name(self) -> str:
        """Resolve user-friendly model name to FunASR model ID, preferring local cache."""
        from ownscribe.config import resolve_model_path

        model = self._config.model
        if model in _MODEL_MAP:
            _, local_name = _MODEL_MAP[model]
            local_path = resolve_model_path(local_name)
            if local_path.exists():
                return str(local_path)
            # Fallback to ModelScope ID
            return _MODEL_MAP[model][0]
        return model

    @staticmethod
    def _resolve_component(mapping: tuple[str, str]) -> str:
        """Resolve a component (VAD/Punc/SPK) to local path or ModelScope ID."""
        from ownscribe.config import resolve_model_path

        ms_id, local_name = mapping
        local_path = resolve_model_path(local_name)
        if local_path.exists():
            return str(local_path)
        return ms_id

    def _load_model(self) -> None:
        """Load FunASR model with optional speaker diarization."""
        import contextlib
        import io
        import logging as _logging
        import warnings

        import tqdm

        model_name = self._resolve_model_name()

        kwargs = {
            "model": model_name,
            "vad_model": self._resolve_component(_VAD_MAP),
            "vad_kwargs": {"max_single_segment_time": 30000},
            "punc_model": self._resolve_component(_PUNC_MAP),
            "device": self._config.device,
            "disable_update": True,
        }

        # Add speaker diarization if enabled
        if self._config.spk_enabled:
            kwargs["spk_model"] = self._resolve_component(_SPK_MAP)

        # Globally disable tqdm progress bars
        _orig_tqdm_init = tqdm.tqdm.__init__

        def _silent_tqdm_init(self, *args, **kwargs):
            kwargs["disable"] = True
            _orig_tqdm_init(self, *args, **kwargs)

        tqdm.tqdm.__init__ = _silent_tqdm_init

        # Suppress all noisy output during model loading
        _prev_level = _logging.root.level
        _logging.root.setLevel(_logging.ERROR)
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            from funasr import AutoModel
            self._model = AutoModel(**kwargs)
        _logging.root.setLevel(_prev_level)

    def prepare_models(self, language: str | None = None) -> None:
        """Pre-load FunASR models (downloads ~2GB on first run)."""
        progress = self._progress
        progress.begin("preparing_models")
        try:
            self._set_detail("preparing_models", f"Loading FunASR model ({self._config.model})")
            self._load_model()
            self._set_detail("preparing_models", "FunASR model ready")
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe audio file using FunASR pipeline.

        Returns TranscriptResult with segments containing speaker labels
        (when spk_enabled=True) and timestamps.
        """
        # Set thread count for Apple Silicon optimization
        ncpu = os.cpu_count() or 8
        optimal_threads = str(min(ncpu - 2, 12))  # Leave headroom
        os.environ.setdefault("OMP_NUM_THREADS", optimal_threads)
        os.environ.setdefault("MKL_NUM_THREADS", optimal_threads)

        progress = self._progress
        progress.begin("transcribing")

        if self._model is None:
            self._set_detail("transcribing", f"Loading FunASR model ({self._config.model})")
            self._load_model()

        self._set_detail("transcribing", "Transcribing audio...")

        # Run FunASR pipeline
        language = self._config.language if self._config.language else "auto"
        generate_kwargs = {
            "input": str(audio_path),
            "batch_size_s": self._config.batch_size_s,
            "use_itn": True,  # Inverse text normalization (numbers, dates → written form)
        }

        # SenseVoice supports language parameter
        if "sensevoice" in self._resolve_model_name().lower() or "SenseVoice" in self._resolve_model_name():
            generate_kwargs["language"] = language

        # Suppress noisy warnings and progress bars during inference
        import contextlib
        import io
        import logging as _logging

        _prev_level = _logging.root.level
        _logging.root.setLevel(_logging.ERROR)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                res = self._model.generate(**generate_kwargs)
        finally:
            _logging.root.setLevel(_prev_level)

        progress.complete("transcribing")

        # If diarization was requested, mark it as done (it's part of generate())
        if self._config.spk_enabled:
            progress.begin("diarizing")
            progress.complete("diarizing")

        # Convert FunASR output to OwnScribe data models
        return self._convert_result(res, audio_path)

    def _to_traditional(self, text: str) -> str:
        """Convert Simplified Chinese to Traditional Chinese (Taiwan variant)."""
        if not self._config.traditional_chinese or not text:
            return text
        try:
            from opencc import OpenCC
            if not hasattr(self, "_cc"):
                self._cc = OpenCC("s2twp")  # 簡體 → 台灣正體（含台灣慣用詞）
            return self._cc.convert(text)
        except ImportError:
            logger.warning("opencc not installed, skipping Traditional Chinese conversion")
            return text

    def _convert_result(self, res: list, audio_path: Path) -> TranscriptResult:
        """Convert FunASR output to OwnScribe TranscriptResult."""
        import soundfile as sf
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        # Get audio duration
        info = sf.info(str(audio_path))
        duration = info.duration

        if not res or not res[0]:
            return TranscriptResult(segments=[], language="", duration=duration)

        result = res[0]
        segments: list[Segment] = []

        # FunASR with speaker diarization returns sentence_info
        sentence_info = result.get("sentence_info", [])

        if sentence_info:
            # Has speaker diarization data
            for sent in sentence_info:
                speaker = f"SPEAKER_{sent['spk']}" if "spk" in sent else None
                start_sec = sent["start"] / 1000.0
                end_sec = sent["end"] / 1000.0
                text = sent.get("sentence", sent.get("text", "")).strip()
                # SenseVoice returns rich tags like <|zh|><|NEUTRAL|>..., strip them
                text = rich_transcription_postprocess(text)
                text = self._to_traditional(text)

                if not text:
                    continue

                segments.append(
                    Segment(
                        text=text,
                        start=start_sec,
                        end=end_sec,
                        speaker=speaker,
                        words=[],  # FunASR sentence_info doesn't provide word-level
                    )
                )
        else:
            # No speaker info — fallback to plain text with timestamps
            text = result.get("text", "")
            text = rich_transcription_postprocess(text)
            text = self._to_traditional(text)
            timestamp = result.get("timestamp", [])

            if timestamp:
                # Has timestamps but no speaker
                for i, ts in enumerate(timestamp):
                    start_sec = ts[0] / 1000.0
                    end_sec = ts[1] / 1000.0
                    # For Paraformer, timestamp aligns with characters
                    # We group into sentences using punctuation from the text
                    segments.append(
                        Segment(
                            text=text if i == 0 else "",  # simplified
                            start=start_sec,
                            end=end_sec,
                            speaker=None,
                            words=[],
                        )
                    )
                # If we got character-level timestamps, re-group into sentences
                if len(timestamp) > 1:
                    segments = self._regroup_by_punctuation(text, timestamp)
            elif text:
                # Plain text only
                segments = [
                    Segment(
                        text=text.strip(),
                        start=0.0,
                        end=duration,
                        speaker=None,
                        words=[],
                    )
                ]

        # Detect language from result or config
        language = self._config.language or result.get("language", "zh")

        return TranscriptResult(
            segments=segments,
            language=language,
            duration=duration,
        )

    @staticmethod
    def _regroup_by_punctuation(text: str, timestamp: list[list[int]]) -> list[Segment]:
        """Regroup character-level timestamps into sentence-level segments."""
        import re

        # Chinese and English sentence-ending punctuation
        sentence_ends = re.compile(r"[。？！；.?!;]")

        segments: list[Segment] = []
        current_text = ""
        current_start: float | None = None

        chars = list(text.replace(" ", ""))
        ts_idx = 0

        for char in chars:
            if ts_idx >= len(timestamp):
                current_text += char
                continue

            ts = timestamp[ts_idx]
            if current_start is None:
                current_start = ts[0] / 1000.0

            current_text += char
            ts_idx += 1

            if sentence_ends.search(char) and current_text.strip():
                segments.append(
                    Segment(
                        text=current_text.strip(),
                        start=current_start,
                        end=ts[1] / 1000.0,
                        speaker=None,
                        words=[],
                    )
                )
                current_text = ""
                current_start = None

        # Remaining text
        if current_text.strip() and current_start is not None:
            end_time = timestamp[-1][1] / 1000.0 if timestamp else 0.0
            segments.append(
                Segment(
                    text=current_text.strip(),
                    start=current_start,
                    end=end_time,
                    speaker=None,
                    words=[],
                )
            )

        return segments
