"""Tests for bounded Community-1 diarization and global speaker stitching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ownscribe.config import DiarizationConfig
from ownscribe.transcription.community_diarizer import CommunityDiarizer, SpeakerTurn, assign_speakers
from ownscribe.transcription.models import Segment, TranscriptResult, Word
from ownscribe.transcription.utils import iter_audio_windows


def test_audio_windows_are_bounded_overlapping_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(75 * 100, dtype=np.float32), 100)

    windows = list(iter_audio_windows(path, 100, window_seconds=60, overlap_seconds=10))

    assert [(offset, len(audio), final) for offset, audio, final in windows] == [
        (0.0, 6000, False),
        (50.0, 2500, True),
    ]


def test_audio_windows_exact_boundary_does_not_emit_overlap_only_tail(tmp_path: Path) -> None:
    path = tmp_path / "exact.wav"
    sf.write(path, np.zeros(60 * 100, dtype=np.float32), 100)

    windows = list(iter_audio_windows(path, 100, window_seconds=60, overlap_seconds=10))

    assert [(offset, len(audio), final) for offset, audio, final in windows] == [(0.0, 6000, True)]


def test_overlap_timeline_reconciles_local_speaker_ids() -> None:
    diarizer = CommunityDiarizer(DiarizationConfig(window_seconds=60, window_overlap_seconds=10))
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    first_embeddings = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    first_mapping = diarizer._match_speakers([], first_embeddings, [], centroids, counts, 0.0)
    assert first_mapping == {"A": "SPEAKER_00", "B": "SPEAKER_01"}

    previous = [SpeakerTurn(50.0, 55.0, "SPEAKER_00"), SpeakerTurn(55.0, 60.0, "SPEAKER_01")]
    local = [SpeakerTurn(0.0, 5.0, "X"), SpeakerTurn(5.0, 10.0, "Y")]
    second_mapping = diarizer._match_speakers(
        local,
        {"X": np.array([0.0, 1.0]), "Y": np.array([1.0, 0.0])},
        previous,
        centroids,
        counts,
        50.0,
    )

    assert second_mapping == {"X": "SPEAKER_00", "Y": "SPEAKER_01"}


def test_embedding_fallback_matches_speaker_absent_from_overlap() -> None:
    config = DiarizationConfig(window_seconds=60, window_overlap_seconds=10, community_speaker_threshold=0.8)
    diarizer = CommunityDiarizer(config)
    centroids = [np.array([1.0, 0.0], dtype=np.float32)]
    counts = [1]

    mapping = diarizer._match_speakers(
        [],
        {"local": np.array([0.99, 0.01])},
        [],
        centroids,
        counts,
        50.0,
    )

    assert mapping == {"local": "SPEAKER_00"}


def test_global_speaker_cap_prevents_cross_window_explosion() -> None:
    config = DiarizationConfig(
        max_speakers=2,
        window_seconds=60,
        window_overlap_seconds=10,
        community_speaker_threshold=0.99,
    )
    diarizer = CommunityDiarizer(config)
    centroids = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    counts = [1, 1]

    mapping = diarizer._match_speakers(
        [],
        {"new": np.array([-1.0, 0.0])},
        [],
        centroids,
        counts,
        50.0,
    )

    assert mapping["new"] in {"SPEAKER_00", "SPEAKER_01"}
    assert len(centroids) == 2


def test_assign_speakers_uses_words_then_segment_overlap() -> None:
    transcript = TranscriptResult(
        segments=[
            Segment(
                text="hello there",
                start=0.0,
                end=2.0,
                words=[Word("hello", 0.0, 0.8), Word("there", 1.1, 2.0)],
            )
        ]
    )
    turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00"), SpeakerTurn(1.0, 2.0, "SPEAKER_01")]

    result = assign_speakers(transcript, turns)

    assert [word.speaker for word in result.segments[0].words] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0].speaker in {"SPEAKER_00", "SPEAKER_01"}
