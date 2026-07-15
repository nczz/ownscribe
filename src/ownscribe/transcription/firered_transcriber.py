"""FireRedASR2-AED transcription with optional CAM++ diarization."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ownscribe.config import resolve_model_path
from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult
from ownscribe.transcription.utils import audio_duration, build_speaker_timeline, iter_audio_chunks, quiet_model_output

_SAMPLE_RATE = 16000


class FireRedTranscriber(Transcriber):
    """Use a configured FireRedASR2S checkout for high-accuracy transcription."""

    def __init__(
        self,
        progress: NullProgress | None = None,
        use_mps: bool = False,
        diarization_enabled: bool = False,
        models_dir: str | None = None,
        speaker_threshold: float = 0.7,
        firered_repo: str = "",
        chunk_seconds: int = 60,
    ) -> None:
        self._progress = progress or NullProgress()
        # FireRedASR2S officially supports CPU/CUDA. MPS monkey-patching is intentionally unsupported.
        self._use_gpu = False
        self._diarization_enabled = diarization_enabled
        self._models_dir = models_dir
        self._speaker_threshold = speaker_threshold
        self._firered_repo = Path(firered_repo).expanduser() if firered_repo else None
        self._chunk_seconds = chunk_seconds
        self._asr_system = None
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

    def _load_asr_system(self) -> None:
        if self._firered_repo is None:
            raise RuntimeError(
                "FireRed requires transcription.firered_repo to point to a compatible FireRedASR2S checkout"
            )
        package = self._firered_repo / "fireredasr2s"
        if not package.is_dir():
            raise RuntimeError(f"Invalid FireRedASR2S checkout: {self._firered_repo}")
        repo = str(self._firered_repo.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
        from fireredasr2s.fireredasr2 import FireRedAsr2Config
        from fireredasr2s.fireredlid import FireRedLidConfig
        from fireredasr2s.fireredpunc import FireRedPuncConfig
        from fireredasr2s.fireredvad import FireRedVadConfig

        asr = FireRedAsr2Config(
            use_gpu=False,
            use_half=False,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        vad_model = self._resolve_local_model("firered-vad")
        lid_model = self._resolve_local_model("firered-lid")
        asr_model = self._resolve_local_model("firered-asr2-aed")
        punctuation_model = self._resolve_local_model("firered-punc")
        config = FireRedAsr2SystemConfig(
            str(vad_model / "VAD"),
            str(lid_model),
            "aed",
            str(asr_model),
            str(punctuation_model),
            FireRedVadConfig(use_gpu=False),
            FireRedLidConfig(use_gpu=False, use_half=False),
            asr,
            FireRedPuncConfig(use_gpu=False),
            enable_vad=1,
            enable_lid=1,
            enable_punc=1,
        )
        with quiet_model_output():
            self._asr_system = FireRedAsr2System(config)

    def _resolve_local_model(self, name: str) -> Path:
        """Resolve a FireRed model to a local directory, downloading canonical IDs when needed."""
        resolved = resolve_model_path(name, self._models_dir)
        if isinstance(resolved, Path):
            return resolved
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=resolved))

    def prepare_models(self, language: str | None = None) -> None:
        progress = self._progress
        progress.begin("preparing_models")
        try:
            if self._diarization_enabled:
                self._set_detail("preparing_models", "Loading CAM++ speaker model...")
                self._load_spk_models()
            self._set_detail("preparing_models", "Loading FireRedASR2-AED...")
            self._load_asr_system()
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        import soundfile as sf

        os.environ.setdefault("OMP_NUM_THREADS", "12")
        if self._asr_system is None:
            self.prepare_models()

        all_sentences: list[dict] = []
        vad_segments: list[list[float]] = []
        embeddings: list[np.ndarray | None] = []
        if self._diarization_enabled:
            self._progress.begin("diarizing")
        self._progress.begin("transcribing")
        for offset, audio in iter_audio_chunks(audio_path, _SAMPLE_RATE, self._chunk_seconds):
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    temporary = Path(handle.name)
                sf.write(str(temporary), audio, _SAMPLE_RATE, subtype="PCM_16")
                with quiet_model_output():
                    result = self._asr_system.process(str(temporary))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            for sentence in result.get("sentences", []):
                sentence = dict(sentence)
                sentence["start_ms"] += offset * 1000
                sentence["end_ms"] += offset * 1000
                all_sentences.append(sentence)
            if self._diarization_enabled:
                chunk_vad, chunk_embeddings = self._extract_speaker_embeddings(audio, _SAMPLE_RATE, offset)
                vad_segments.extend(chunk_vad)
                embeddings.extend(chunk_embeddings)
        self._progress.complete("transcribing")
        timeline = build_speaker_timeline(vad_segments, embeddings, self._speaker_threshold)
        if self._diarization_enabled:
            self._progress.complete("diarizing")
        segments = self._align_speakers(all_sentences, timeline)
        try:
            from opencc import OpenCC

            converter = OpenCC("s2twp")
            for segment in segments:
                segment.text = converter.convert(segment.text)
        except ImportError:
            pass
        language = all_sentences[0].get("lang", "zh") if all_sentences else "zh"
        return TranscriptResult(segments=segments, language=language, duration=audio_duration(audio_path))

    def _extract_speaker_embeddings(
        self, audio: np.ndarray, sample_rate: int, offset_seconds: float
    ) -> tuple[list[list[float]], list[np.ndarray | None]]:
        with quiet_model_output():
            result = self._vad_model.generate(input=audio)
        local_segments = result[0]["value"]
        vad_segments = [[start + offset_seconds * 1000, end + offset_seconds * 1000] for start, end in local_segments]
        embeddings: list[np.ndarray | None] = []
        for start_ms, end_ms in local_segments:
            chunk = audio[int(start_ms / 1000 * sample_rate) : int(end_ms / 1000 * sample_rate)]
            if len(chunk) < 1600:
                embeddings.append(None)
                continue
            with quiet_model_output():
                result = self._spk_model.generate(input=chunk, input_len=np.array([len(chunk)]))
            embeddings.append(np.asarray(result[0]["spk_embedding"]).reshape(-1))
        return vad_segments, embeddings

    def _align_speakers(self, sentences: list[dict], timeline: list[dict]) -> list[Segment]:
        segments = []
        for sentence in sentences:
            text = sentence["text"].strip()
            if text:
                start = sentence["start_ms"]
                end = sentence["end_ms"]
                segments.append(
                    Segment(
                        text=text,
                        start=start / 1000,
                        end=end / 1000,
                        speaker=self._find_speaker(timeline, start, end),
                        words=[],
                    )
                )
        return segments

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
