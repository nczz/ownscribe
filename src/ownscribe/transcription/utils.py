"""Shared audio and speaker-diarization helpers."""

from __future__ import annotations

import contextlib
import io
import logging
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def audio_duration(path: Path) -> float:
    """Return duration without decoding the audio payload."""
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def iter_audio_chunks(
    path: Path, target_rate: int = 16000, chunk_seconds: int = 60
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield bounded mono chunks as ``(offset_seconds, samples)``."""
    import soundfile as sf

    if chunk_seconds < 30:
        raise ValueError("transcription chunk_seconds must be at least 30")
    converted: Path | None = None
    try:
        source = sf.SoundFile(str(path))
    except sf.LibsndfileError:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            converted = Path(handle.name)
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ar", str(target_rate), "-ac", "1", str(converted)],
                check=True,
                capture_output=True,
            )
            source = sf.SoundFile(str(converted))
        except Exception:
            converted.unlink(missing_ok=True)
            raise
    try:
        source_rate = source.samplerate
        source_frames = chunk_seconds * source_rate
        offset_frames = 0
        yielded = False
        while True:
            block = source.read(source_frames, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            yielded = True
            mono = np.ascontiguousarray(block.mean(axis=1), dtype=np.float32)
            if not np.isfinite(mono).all():
                raise ValueError(f"Audio contains non-finite samples: {path}")
            if source_rate != target_rate:
                import torch
                import torchaudio

                waveform = torch.from_numpy(mono).unsqueeze(0)
                mono = torchaudio.functional.resample(waveform, source_rate, target_rate).squeeze(0).cpu().numpy()
            yield offset_frames / source_rate, np.ascontiguousarray(mono, dtype=np.float32)
            offset_frames += len(block)
        if not yielded:
            raise ValueError(f"Audio file is empty: {path}")
    finally:
        source.close()
        if converted is not None:
            converted.unlink(missing_ok=True)


def cluster_speaker_embeddings(embeddings: list[np.ndarray | None], threshold: float) -> list[int | None]:
    """Cluster normalized embeddings using incrementally updated centroids."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("speaker threshold must be between 0 and 1")

    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[int | None] = []
    for embedding in embeddings:
        if embedding is None:
            labels.append(None)
            continue
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        magnitude = float(np.linalg.norm(vector))
        if not np.isfinite(magnitude) or magnitude <= 1e-12:
            labels.append(None)
            continue
        vector /= magnitude
        similarities = [float(np.dot(vector, centroid)) for centroid in centroids]
        if similarities and max(similarities) >= threshold:
            label = int(np.argmax(similarities))
            updated = centroids[label] * counts[label] + vector
            updated_norm = float(np.linalg.norm(updated))
            centroids[label] = updated / updated_norm
            counts[label] += 1
        else:
            label = len(centroids)
            centroids.append(vector)
            counts.append(1)
        labels.append(label)
    return labels


def build_speaker_timeline(vad_segments: list, embeddings: list[np.ndarray | None], threshold: float) -> list[dict]:
    """Build a timeline, leaving unclassifiable short regions unlabeled."""
    labels = cluster_speaker_embeddings(embeddings, threshold)
    return [
        {
            "start_ms": segment[0],
            "end_ms": segment[1],
            "speaker": f"SPEAKER_{label}" if label is not None else None,
        }
        for segment, label in zip(vad_segments, labels, strict=True)
    ]


@contextlib.contextmanager
def quiet_model_output() -> Iterator[None]:
    """Temporarily suppress noisy model output without mutating tqdm globally."""
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        root.setLevel(previous_level)
