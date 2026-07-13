"""Breeze-ASR-25 (MediaTek Research) + CAM++ speaker diarization backend.

Optimized for Taiwanese Mandarin + code-switching (中英混合).
Outputs Traditional Chinese natively — no OpenCC conversion needed.

Pipeline:
1. CAM++ (FunASR): VAD + speaker embedding + clustering → speaker timeline
2. Breeze-ASR-25: Whisper-large-v2 fine-tuned for Taiwan → transcription with timestamps
3. Time alignment: assign speaker labels to each segment
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from pathlib import Path

import numpy as np

from ownscribe.config import resolve_model_path
from ownscribe.progress import NullProgress
from ownscribe.transcription.base import Transcriber
from ownscribe.transcription.models import Segment, TranscriptResult, Word

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SPK_THRESHOLD = 0.7


class BreezeTranscriber(Transcriber):
    """Taiwanese Mandarin ASR using Breeze-ASR-25 + CAM++ speaker diarization.

    Best for: meetings with Taiwanese Mandarin + English code-switching.
    Speed: ~5.2x realtime on Apple M5 Pro (MPS GPU).
    Output: Native Traditional Chinese (繁體中文).
    """

    def __init__(
        self,
        progress: NullProgress | None = None,
        use_mps: bool = True,
    ) -> None:
        self._progress = progress or NullProgress()
        self._use_mps = use_mps
        self._model = None
        self._processor = None
        self._device = None
        self._vad_model = None
        self._spk_model = None

    def _set_detail(self, key: str, text: str | None) -> None:
        set_detail = getattr(self._progress, "set_detail", None)
        if callable(set_detail):
            set_detail(key, text)

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

    def _load_asr_model(self) -> None:
        """Load Breeze-ASR-25 model."""
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_path = str(resolve_model_path("breeze-asr-25"))
        self._device = "mps" if (self._use_mps and torch.backends.mps.is_available()) else "cpu"
        self._processor = WhisperProcessor.from_pretrained(model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_path
        ).to(self._device).eval()

    def prepare_models(self, language: str | None = None) -> None:
        """Pre-load all models."""
        progress = self._progress
        progress.begin("preparing_models")
        try:
            self._set_detail("preparing_models", "Loading CAM++ speaker model...")
            self._load_spk_models()
            self._set_detail("preparing_models", "Loading Breeze-ASR-25...")
            self._load_asr_model()
            progress.complete("preparing_models")
        except Exception:
            progress.fail("preparing_models")
            raise

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Transcribe with Breeze-ASR-25 + CAM++ speaker diarization."""
        import soundfile as sf
        import torch

        os.environ.setdefault("OMP_NUM_THREADS", "12")

        progress = self._progress

        # Load audio
        audio, sr = sf.read(str(audio_path))
        if sr != _SAMPLE_RATE:
            import torchaudio
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, _SAMPLE_RATE)
            audio = resampler(waveform).squeeze().numpy()
            sr = _SAMPLE_RATE
        duration = len(audio) / sr

        # Load models if needed
        if self._model is None:
            progress.begin("preparing_models")
            self._set_detail("preparing_models", "Loading models...")
            self._load_spk_models()
            self._load_asr_model()
            progress.complete("preparing_models")

        # STEP 1: Speaker diarization (CAM++)
        progress.begin("diarizing")
        self._set_detail("diarizing", "Identifying speakers (CAM++)...")
        spk_timeline = self._run_diarization(audio, sr)
        progress.complete("diarizing")

        # STEP 2: ASR transcription (Breeze-ASR-25)
        progress.begin("transcribing")
        self._set_detail("transcribing", "Transcribing (Breeze-ASR-25, MPS)...")

        # Process audio in 30s chunks for long audio
        all_segments = self._transcribe_chunked(audio, sr)

        progress.complete("transcribing")

        # STEP 3: Align speakers to segments
        segments = self._align_speakers(all_segments, spk_timeline)

        return TranscriptResult(
            segments=segments,
            language="zh-TW",
            duration=duration,
        )

    def _transcribe_chunked(self, audio: np.ndarray, sr: int) -> list[Segment]:
        """Transcribe audio, handling long recordings by chunking at 30s."""
        import re
        import torch

        chunk_duration = 30  # Whisper's max context is 30s
        chunk_samples = chunk_duration * sr
        total_samples = len(audio)
        segments = []

        offset = 0
        while offset < total_samples:
            chunk = audio[offset:offset + chunk_samples]

            # Pad if too short
            if len(chunk) < 3200:  # < 0.2s
                offset += chunk_samples
                continue

            # Process through Whisper
            input_features = self._processor(
                chunk, sampling_rate=sr, return_tensors="pt"
            ).input_features.to(self._device)

            with torch.no_grad():
                predicted_ids = self._model.generate(
                    input_features,
                    return_timestamps=True,
                    language="zh",
                    task="transcribe",
                )

            # Decode with timestamps
            decoded = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=False, decode_with_timestamps=True
            )[0]

            # Parse timestamp tokens: <|0.00|>text<|2.50|>
            offset_sec = offset / sr
            pattern = r"<\|([\d.]+)\|>(.*?)<\|([\d.]+)\|>"
            for match in re.finditer(pattern, decoded):
                start = float(match.group(1)) + offset_sec
                text = match.group(2).strip()
                end = float(match.group(3)) + offset_sec

                if text and not text.startswith("<|"):
                    segments.append(Segment(
                        text=text,
                        start=start,
                        end=end,
                        speaker=None,
                        words=[],
                    ))

            offset += chunk_samples

        return segments

    def _run_diarization(self, audio: np.ndarray, sr: int) -> list[dict]:
        """Run VAD + CAM++ to produce speaker timeline."""
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

        # Cluster
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

        timeline = []
        for i, seg in enumerate(vad_segments):
            spk = speakers[i] if speakers[i] >= 0 else next_spk
            timeline.append({
                "start_ms": seg[0],
                "end_ms": seg[1],
                "speaker": f"SPEAKER_{spk}",
            })

        return timeline

    def _align_speakers(self, segments: list[Segment], spk_timeline: list[dict]) -> list[Segment]:
        """Assign speaker labels to transcribed segments."""
        for seg in segments:
            seg.speaker = self._find_speaker(spk_timeline, seg.start * 1000, seg.end * 1000)
        return segments

    @staticmethod
    def _find_speaker(timeline: list[dict], start_ms: float, end_ms: float) -> str:
        """Find best matching speaker for a time range."""
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

        # Fallback: nearest
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
