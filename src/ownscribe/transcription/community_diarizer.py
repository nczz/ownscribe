"""Bounded-memory pyannote Community-1 speaker diarization."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ownscribe.config import DiarizationConfig
from ownscribe.transcription.models import TranscriptResult
from ownscribe.transcription.utils import iter_audio_windows

_SAMPLE_RATE = 16_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class CommunityDiarizer:
    """Run Community-1 on overlapping windows and reconcile speakers globally."""

    def __init__(self, config: DiarizationConfig, progress=None) -> None:
        self._config = config
        self._progress = progress
        self._pipeline = None

    def prepare_models(self) -> None:
        if not self._config.hf_token:
            raise ValueError("Community-1 diarization requires HF_TOKEN or diarization.hf_token")
        if self._pipeline is not None:
            return

        os.environ["PYANNOTE_METRICS_ENABLED"] = "1" if self._config.telemetry else "0"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
            from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self._config.hf_token,
        )
        pipeline.segmentation_batch_size = max(1, self._config.segmentation_batch_size)
        pipeline.embedding_batch_size = max(1, self._config.embedding_batch_size)
        device = "cpu" if self._config.device == "auto" else self._config.device
        if device != "cpu":
            import torch

            pipeline.to(torch.device(device))
        self._pipeline = pipeline

    def close(self) -> None:
        self._pipeline = None
        _release_torch_memory()

    def diarize(self, audio_path: Path, transcript: TranscriptResult) -> TranscriptResult:
        self._validate_config()
        self.prepare_models()
        if self._progress is not None:
            self._progress.begin("diarizing")

        centroids: list[np.ndarray | None] = []
        centroid_counts: list[int] = []
        turns: list[SpeakerTurn] = []
        try:
            windows = iter_audio_windows(
                audio_path,
                _SAMPLE_RATE,
                self._config.window_seconds,
                self._config.window_overlap_seconds,
            )
            for window_index, (offset, audio, is_last) in enumerate(windows):
                local_turns, local_embeddings = self._diarize_window(
                    audio, is_only_window=window_index == 0 and is_last
                )
                mapping = self._match_speakers(
                    local_turns,
                    local_embeddings,
                    turns,
                    centroids,
                    centroid_counts,
                    offset,
                )
                keep_start = offset if window_index == 0 else offset + self._config.window_overlap_seconds / 2
                window_end = offset + len(audio) / _SAMPLE_RATE
                keep_end = window_end if is_last else window_end - self._config.window_overlap_seconds / 2
                for turn in local_turns:
                    start = max(offset + turn.start, keep_start)
                    end = min(offset + turn.end, keep_end)
                    if end > start:
                        turns.append(SpeakerTurn(start, end, mapping[turn.speaker]))
        except Exception:
            if self._progress is not None:
                self._progress.fail("diarizing")
            raise

        if self._progress is not None:
            self._progress.complete("diarizing")
        if self._config.min_speakers and len(centroids) < self._config.min_speakers:
            logger.warning(
                "Community-1 detected %d speakers, below configured minimum %d",
                len(centroids),
                self._config.min_speakers,
            )
        return assign_speakers(transcript, turns)

    def _diarize_window(
        self, audio: np.ndarray, *, is_only_window: bool
    ) -> tuple[list[SpeakerTurn], dict[str, np.ndarray]]:
        import torch

        kwargs = {}
        # A global minimum must not force phantom speakers into every long-file window.
        if is_only_window and self._config.min_speakers > 0:
            kwargs["min_speakers"] = self._config.min_speakers
        if self._config.max_speakers > 0:
            kwargs["max_speakers"] = self._config.max_speakers
        if self._progress is not None:
            kwargs["hook"] = self._progress.diarization_hook
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
            output = self._pipeline({"waveform": waveform, "sample_rate": _SAMPLE_RATE}, **kwargs)
        annotation = output.exclusive_speaker_diarization
        turns = [SpeakerTurn(segment.start, segment.end, speaker) for segment, speaker in annotation]
        turn_speakers = {turn.speaker for turn in turns}
        labels = output.speaker_diarization.labels()
        embeddings = _collect_valid_embeddings(labels, output.speaker_embeddings, turn_speakers)
        return turns, embeddings

    def _match_speakers(
        self,
        local_turns: list[SpeakerTurn],
        local_embeddings: dict[str, np.ndarray],
        global_turns: list[SpeakerTurn],
        centroids: list[np.ndarray | None],
        centroid_counts: list[int],
        offset: float,
    ) -> dict[str, str]:
        local_labels = sorted({turn.speaker for turn in local_turns} | set(local_embeddings))
        mapping: dict[str, str] = {}
        used_global: set[int] = set()

        overlap_end = offset + self._config.window_overlap_seconds
        overlap_scores: list[tuple[float, str, int]] = []
        for local_label in local_labels:
            for global_index in range(len(centroids)):
                score = _timeline_overlap(
                    local_turns,
                    local_label,
                    global_turns,
                    _speaker_label(global_index),
                    offset,
                    overlap_end,
                )
                if score >= 0.5:
                    overlap_scores.append((score, local_label, global_index))
        for _, local_label, global_index in sorted(overlap_scores, reverse=True):
            if local_label not in mapping and global_index not in used_global:
                mapping[local_label] = _speaker_label(global_index)
                used_global.add(global_index)

        similarities: list[tuple[float, str, int]] = []
        for local_label in local_labels:
            if local_label in mapping:
                continue
            vector = local_embeddings.get(local_label)
            if vector is None:
                continue
            for global_index, centroid in enumerate(centroids):
                if global_index not in used_global and centroid is not None:
                    similarities.append((float(np.dot(vector, centroid)), local_label, global_index))
        for similarity, local_label, global_index in sorted(similarities, reverse=True):
            if similarity < self._config.community_speaker_threshold:
                break
            if local_label not in mapping and global_index not in used_global:
                mapping[local_label] = _speaker_label(global_index)
                used_global.add(global_index)

        for local_label in local_labels:
            if local_label not in mapping:
                local_embedding = local_embeddings.get(local_label)
                available = [index for index in range(len(centroids)) if index not in used_global]
                if self._config.max_speakers and len(centroids) >= self._config.max_speakers and available:
                    if local_embedding is None:
                        global_index = available[0]
                    else:
                        global_index = max(
                            available,
                            key=lambda index: (
                                float(np.dot(local_embedding, centroids[index]))
                                if centroids[index] is not None
                                else float("-inf")
                            ),
                        )
                    mapping[local_label] = _speaker_label(global_index)
                    used_global.add(global_index)
                else:
                    mapping[local_label] = _speaker_label(len(centroids))
                    centroids.append(local_embedding)
                    centroid_counts.append(0)
            local_embedding = local_embeddings.get(local_label)
            if local_embedding is None:
                continue
            global_index = int(mapping[local_label].removeprefix("SPEAKER_"))
            count = centroid_counts[global_index]
            centroid = centroids[global_index]
            if centroid is None:
                centroids[global_index] = local_embedding
                centroid_counts[global_index] = 1
                continue
            updated = centroid * count + local_embedding
            centroids[global_index] = _normalize_embedding(updated)
            centroid_counts[global_index] = count + 1
        return mapping

    def _validate_config(self) -> None:
        if self._config.window_seconds < 30:
            raise ValueError("diarization.window_seconds must be at least 30")
        if not 0 <= self._config.window_overlap_seconds * 2 < self._config.window_seconds:
            raise ValueError("diarization.window_overlap_seconds must be less than half the window")
        if not 0 < self._config.community_speaker_threshold < 1:
            raise ValueError("diarization.community_speaker_threshold must be between 0 and 1")
        if self._config.min_speakers < 0 or self._config.max_speakers < 0:
            raise ValueError("speaker bounds cannot be negative")
        if self._config.max_speakers and self._config.min_speakers > self._config.max_speakers:
            raise ValueError("diarization.min_speakers cannot exceed max_speakers")
        if self._config.segmentation_batch_size < 1 or self._config.embedding_batch_size < 1:
            raise ValueError("Community-1 batch sizes must be positive")


def assign_speakers(transcript: TranscriptResult, turns: list[SpeakerTurn]) -> TranscriptResult:
    """Assign exclusive diarization turns to words and transcript segments."""
    for segment in transcript.segments:
        for word in segment.words:
            word.speaker = _speaker_for_interval(turns, word.start, word.end)
        word_speakers = [word.speaker for word in segment.words if word.speaker]
        segment.speaker = _speaker_for_interval(turns, segment.start, segment.end) or _majority_label(word_speakers)
    return transcript


def _speaker_for_interval(turns: list[SpeakerTurn], start: float, end: float) -> str | None:
    scores: dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap:
            scores[turn.speaker] = scores.get(turn.speaker, 0.0) + overlap
    if scores:
        return max(scores, key=scores.get)
    midpoint = (start + end) / 2
    nearby = [(min(abs(midpoint - turn.start), abs(midpoint - turn.end)), turn.speaker) for turn in turns]
    if nearby and min(nearby)[0] <= 1.0:
        return min(nearby)[1]
    return None


def _majority_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    counts = {label: labels.count(label) for label in dict.fromkeys(labels)}
    return max(counts, key=counts.get)


def _timeline_overlap(
    local_turns: list[SpeakerTurn],
    local_label: str,
    global_turns: list[SpeakerTurn],
    global_label: str,
    offset: float,
    overlap_end: float,
) -> float:
    score = 0.0
    for local in local_turns:
        if local.speaker != local_label:
            continue
        local_start, local_end = offset + local.start, offset + local.end
        for global_turn in global_turns:
            if global_turn.speaker == global_label:
                score += max(
                    0.0,
                    min(local_end, global_turn.end, overlap_end) - max(local_start, global_turn.start, offset),
                )
    return score


def _collect_valid_embeddings(
    labels: list[str],
    speaker_embeddings,
    allowed_labels: set[str] | None = None,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for index, label in enumerate(labels):
        if allowed_labels is not None and label not in allowed_labels:
            continue
        try:
            embeddings[label] = _normalize_embedding(speaker_embeddings[index])
        except (IndexError, TypeError, ValueError):
            logger.warning(
                "Community-1 returned no usable embedding for %s; speaker stitching will use overlap only",
                label,
            )
    return embeddings


def _normalize_embedding(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        raise ValueError("Community-1 returned an invalid speaker embedding")
    return vector / magnitude


def _speaker_label(index: int) -> str:
    return f"SPEAKER_{index:02d}"


def _release_torch_memory() -> None:
    gc.collect()
    with contextlib.suppress(ImportError):
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
