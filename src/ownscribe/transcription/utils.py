"""Shared audio and speaker-diarization helpers."""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def load_audio_mono(path: Path, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load finite float32 mono audio and optionally resample it."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")

    if target_rate is not None and sample_rate != target_rate:
        import torch
        import torchaudio

        waveform = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(waveform, sample_rate, target_rate).squeeze(0).cpu().numpy()
        sample_rate = target_rate
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


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
