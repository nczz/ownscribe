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
        """Load FunASR model (ASR only, no spk_model to avoid distribute_spk bug)."""
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

        # NOTE: We intentionally do NOT pass spk_model here.
        # FunASR's built-in distribute_spk has a bug (TypeError when SenseVoice
        # timestamps contain None). Speaker diarization is done independently
        # via _run_diarization() using CAM++ directly.

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

        # Load independent CAM++ for speaker diarization (if enabled)
        if self._config.spk_enabled:
            self._load_spk_models()

    def _load_spk_models(self) -> None:
        """Load independent VAD + CAM++ for speaker diarization."""
        import contextlib
        import io

        from ownscribe.config import resolve_model_path

        vad_path = resolve_model_path("fsmn-vad")
        spk_path = resolve_model_path("campplus")

        if not vad_path.exists() or not spk_path.exists():
            missing = []
            if not vad_path.exists():
                missing.append("fsmn-vad")
            if not spk_path.exists():
                missing.append("campplus")
            raise RuntimeError(
                f"Speaker diarization models not found: {', '.join(missing)}\n"
                f"Download them first:\n"
                f"  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('funasr/fsmn-vad', local_dir='models/fsmn-vad'); "
                f"snapshot_download('funasr/campplus', local_dir='models/campplus')\""
            )

        from funasr import AutoModel

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self._vad_model = AutoModel(model=str(vad_path), disable_update=True)
            self._spk_model = AutoModel(model=str(spk_path), disable_update=True)

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

        # Speaker diarization (independent CAM++, avoids FunASR distribute_spk bug)
        spk_timeline = []
        if self._config.spk_enabled and hasattr(self, "_vad_model"):
            progress.begin("diarizing")
            import soundfile as sf
            audio_data, sr = sf.read(str(audio_path))
            spk_timeline = self._run_diarization(audio_data, sr)
            progress.complete("diarizing")

        # Convert FunASR output to OwnScribe data models
        return self._convert_result(res, audio_path, spk_timeline)

    def _run_diarization(self, audio, sr: int) -> list[dict]:
        """Run independent VAD + CAM++ speaker diarization."""
        import contextlib
        import io

        import numpy as np
        from numpy.linalg import norm

        # VAD
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            vad_res = self._vad_model.generate(input=audio)
        vad_segments = vad_res[0]["value"]

        if not vad_segments:
            return []

        # Extract speaker embeddings
        embeddings = []
        for seg in vad_segments:
            chunk = audio[int(seg[0] / 1000 * sr):int(seg[1] / 1000 * sr)]
            if len(chunk) < 1600:
                embeddings.append(None)
                continue
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                emb_res = self._spk_model.generate(input=chunk, input_len=np.array([len(chunk)]))
            embeddings.append(np.array(emb_res[0]["spk_embedding"]).flatten())

        # Cluster by cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))

        speakers = [-1] * len(embeddings)
        next_spk = 0
        for i in range(len(embeddings)):
            if embeddings[i] is None or speakers[i] >= 0:
                continue
            speakers[i] = next_spk
            for j in range(i + 1, len(embeddings)):
                if embeddings[j] is None or speakers[j] >= 0:
                    continue
                if cosine_sim(embeddings[i], embeddings[j]) > 0.7:
                    speakers[j] = next_spk
            next_spk += 1

        timeline = []
        for i, seg in enumerate(vad_segments):
            spk = speakers[i] if speakers[i] >= 0 else next_spk
            timeline.append({
                "start_ms": seg[0],
                "end_ms": seg[1],
                "speaker": f"SPEAKER_{spk}",
            })
        return timeline

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

    def _convert_result(self, res: list, audio_path: Path, spk_timeline: list[dict] | None = None) -> TranscriptResult:
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
            # Has sentence-level data from FunASR
            for sent in sentence_info:
                start_sec = sent["start"] / 1000.0 if sent.get("start") is not None else 0.0
                end_sec = sent["end"] / 1000.0 if sent.get("end") is not None else 0.0
                text = sent.get("sentence", sent.get("text", "")).strip()
                # SenseVoice returns rich tags like <|zh|><|NEUTRAL|>..., strip them
                text = rich_transcription_postprocess(text)
                text = self._to_traditional(text)

                if not text:
                    continue

                # Get speaker from independent CAM++ timeline (not from FunASR's buggy spk field)
                speaker = self._find_speaker(spk_timeline, start_sec * 1000, end_sec * 1000) if spk_timeline else None

                segments.append(
                    Segment(
                        text=text,
                        start=start_sec,
                        end=end_sec,
                        speaker=speaker,
                        words=[],
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

    @staticmethod
    def _find_speaker(timeline: list[dict], start_ms: float, end_ms: float) -> str:
        """Find best matching speaker for a time range using overlap."""
        if not timeline:
            return "SPEAKER_0"

        best_speaker = None
        best_overlap = 0
        for s in timeline:
            overlap = max(0, min(end_ms, s["end_ms"]) - max(start_ms, s["start_ms"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s["speaker"]

        if best_speaker:
            return best_speaker

        # Fallback: nearest segment
        mid = (start_ms + end_ms) / 2
        min_dist = float("inf")
        nearest = "SPEAKER_0"
        for s in timeline:
            seg_mid = (s["start_ms"] + s["end_ms"]) / 2
            dist = abs(mid - seg_mid)
            if dist < min_dist:
                min_dist = dist
                nearest = s["speaker"]
        return nearest
