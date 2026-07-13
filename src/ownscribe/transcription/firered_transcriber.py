"""FireRedASR2-AED + CAM++ speaker diarization backend.

Combines the highest-accuracy Chinese ASR model (CER 3.05%) with
FunASR's CAM++ speaker embeddings for speaker-attributed transcription.

Pipeline:
1. CAM++ (via FunASR): VAD + speaker embedding + clustering → speaker timeline
2. FireRedASR2-AED: precision transcription with word-level timestamps
3. Time alignment: assign speaker labels to each sentence

Requires:
- FireRedASR2S repo cloned at ~/Projects/FireRedASR2S (or configured path)
- FunASR models: fsmn-vad, campplus (already cached at ~/.cache/funasr/models)
- Apple Silicon: uses MPS GPU acceleration via monkey-patch
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from ownscribe.config import resolve_model_path
from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult

logger = logging.getLogger(__name__)

_FIRERED_BASE = Path("~/Projects/FireRedASR2S").expanduser()
_SAMPLE_RATE = 16000
_SPK_THRESHOLD = 0.7  # Cosine similarity threshold for same-speaker


class FireRedTranscriber(Transcriber):
    """High-accuracy Chinese ASR using FireRedASR2-AED + CAM++ speaker diarization.

    Best for: post-meeting precision transcription when quality matters most.
    Speed: ~2.3x realtime on Apple M5 Pro (MPS GPU).
    """

    def __init__(
        self,
        progress: NullProgress | None = None,
        use_mps: bool = True,
    ) -> None:
        self._progress = progress or NullProgress()
        self._use_mps = use_mps
        self._asr_system = None
        self._vad_model = None
        self._spk_model = None

    def _set_detail(self, key: str, text: str | None) -> None:
        set_detail = getattr(self._progress, "set_detail", None)
        if callable(set_detail):
            set_detail(key, text)

    def _apply_mps_patch(self) -> None:
        """Monkey-patch .cuda() → .to('mps') for Apple Silicon."""
        if not self._use_mps:
            return
        import torch
        if not torch.backends.mps.is_available():
            return
        torch.Tensor.cuda = lambda self, *a, **k: self.to("mps")
        torch.nn.Module.cuda = lambda self, *a, **k: self.to("mps")

    def _load_spk_models(self) -> None:
        """Load FunASR VAD + CAM++ for speaker diarization."""
        from funasr import AutoModel

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self._vad_model = AutoModel(
                model=str(resolve_model_path("fsmn-vad")), disable_update=True
            )
            self._spk_model = AutoModel(
                model=str(resolve_model_path("campplus")), disable_update=True
            )

    def _load_asr_system(self) -> None:
        """Load FireRedASR2 system."""
        # Add FireRedASR2S to path
        firered_path = str(_FIRERED_BASE)
        if firered_path not in sys.path:
            sys.path.insert(0, firered_path)

        self._apply_mps_patch()

        from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
        from fireredasr2s.fireredasr2 import FireRedAsr2Config
        from fireredasr2s.fireredlid import FireRedLidConfig
        from fireredasr2s.fireredpunc import FireRedPuncConfig
        from fireredasr2s.fireredvad import FireRedVadConfig

        use_gpu = self._use_mps
        asr_config = FireRedAsr2Config(
            use_gpu=use_gpu, use_half=False,
            beam_size=3, nbest=1, decode_max_len=0,
            softmax_smoothing=1.25, aed_length_penalty=0.6,
            eos_penalty=1.0, return_timestamp=True,
        )
        config = FireRedAsr2SystemConfig(
            str(resolve_model_path("firered-vad") / "VAD"),
            str(resolve_model_path("firered-lid")),
            "aed", str(resolve_model_path("firered-asr2-aed")),
            str(resolve_model_path("firered-punc")),
            FireRedVadConfig(use_gpu=False),
            FireRedLidConfig(use_gpu=use_gpu, use_half=False),
            asr_config,
            FireRedPuncConfig(use_gpu=use_gpu),
            enable_vad=1, enable_lid=1, enable_punc=1,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            self._asr_system = FireRedAsr2System(config)

    def prepare_models(self, language: str | None = None) -> None:
        """Pre-load all models."""
        progress = self._progress
        progress.begin("preparing_models")
        try:
            self._set_detail("preparing_models", "Loading CAM++ speaker model...")
            self._load_spk_models()
            self._set_detail("preparing_models", "Loading FireRedASR2-AED...")
            self._load_asr_system()
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe with FireRedASR2 + CAM++ speaker diarization."""
        import soundfile as sf

        os.environ.setdefault("OMP_NUM_THREADS", "12")

        progress = self._progress

        # Ensure 16kHz mono
        audio_path = self._ensure_16k(audio_path)
        audio, sr = sf.read(str(audio_path))
        duration = len(audio) / sr

        # Load models if needed
        if self._vad_model is None:
            progress.begin("preparing_models")
            self._set_detail("preparing_models", "Loading models...")
            self._load_spk_models()
            self._load_asr_system()
            progress.complete("preparing_models")

        # STEP 1: Speaker diarization (CAM++)
        progress.begin("diarizing")
        self._set_detail("diarizing", "Identifying speakers (CAM++)...")
        spk_timeline = self._run_diarization(audio, sr)
        progress.complete("diarizing")

        # STEP 2: ASR transcription (FireRedASR2)
        progress.begin("transcribing")
        self._set_detail("transcribing", "Transcribing (FireRedASR2-AED)...")
        with contextlib.redirect_stderr(io.StringIO()):
            result = self._asr_system.process(str(audio_path))
        progress.complete("transcribing")

        # STEP 3: Align speakers to sentences
        sentences = result.get("sentences", [])
        segments = self._align_speakers(sentences, spk_timeline)

        # Apply Traditional Chinese conversion
        try:
            from opencc import OpenCC
            cc = OpenCC("s2twp")
            for seg in segments:
                seg.text = cc.convert(seg.text)
        except ImportError:
            pass

        return TranscriptResult(
            segments=segments,
            language=result.get("sentences", [{}])[0].get("lang", "zh") if sentences else "zh",
            duration=duration,
        )

    def _run_diarization(self, audio: np.ndarray, sr: int) -> list[dict]:
        """Run VAD + CAM++ to produce speaker timeline."""
        # VAD
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            vad_res = self._vad_model.generate(input=audio)
        vad_segments = vad_res[0]["value"]

        if not vad_segments:
            return []

        # Extract speaker embeddings per segment
        embeddings = []
        for seg in vad_segments:
            chunk = audio[int(seg[0] / 1000 * sr):int(seg[1] / 1000 * sr)]
            if len(chunk) < 1600:  # Skip tiny segments (<0.1s)
                embeddings.append(None)
                continue
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                emb_res = self._spk_model.generate(input=chunk, input_len=np.array([len(chunk)]))
            embeddings.append(np.array(emb_res[0]["spk_embedding"]).flatten())

        # Cluster by cosine similarity
        from numpy.linalg import norm

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
                if cosine_sim(embeddings[i], embeddings[j]) > _SPK_THRESHOLD:
                    speakers[j] = next_spk
            next_spk += 1

        # Build timeline
        timeline = []
        for i, seg in enumerate(vad_segments):
            spk = speakers[i] if speakers[i] >= 0 else next_spk
            timeline.append({
                "start_ms": seg[0],
                "end_ms": seg[1],
                "speaker": f"SPEAKER_{spk}",
            })

        return timeline

    def _align_speakers(self, sentences: list[dict], spk_timeline: list[dict]) -> list[Segment]:
        """Align FireRedASR2 sentences with CAM++ speaker timeline."""
        segments = []
        for sent in sentences:
            start_ms = sent["start_ms"]
            end_ms = sent["end_ms"]
            text = sent["text"].strip()
            if not text:
                continue

            speaker = self._find_speaker(spk_timeline, start_ms, end_ms)
            segments.append(Segment(
                text=text,
                start=start_ms / 1000.0,
                end=end_ms / 1000.0,
                speaker=speaker,
                words=[],
            ))

        return segments

    @staticmethod
    def _find_speaker(timeline: list[dict], start_ms: float, end_ms: float) -> str:
        """Find the best matching speaker for a time range."""
        if not timeline:
            return "SPEAKER_0"

        # Strategy: maximum overlap
        best_speaker = None
        best_overlap = 0
        for s in timeline:
            overlap = max(0, min(end_ms, s["end_ms"]) - max(start_ms, s["start_ms"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = s["speaker"]

        if best_speaker:
            return best_speaker

        # Fallback: nearest speaker segment
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

    @staticmethod
    def _ensure_16k(audio_path: Path) -> Path:
        """Convert audio to 16kHz mono if needed."""
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate == 16000 and info.channels == 1:
            return audio_path

        # Convert via ffmpeg
        import subprocess
        out_path = audio_path.parent / f"{audio_path.stem}_16k.wav"
        if out_path.exists():
            return out_path

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", str(out_path)],
            capture_output=True,
        )
        return out_path
