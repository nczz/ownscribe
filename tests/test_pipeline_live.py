"""Behavioral tests for bounded live-preview buffering."""

from __future__ import annotations

import queue

import numpy as np

from ownscribe.pipeline_live import _put_latest_preview


def test_preview_queue_drops_oldest_chunk_when_inference_lags():
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
    first = np.array([1.0], dtype=np.float32)
    second = np.array([2.0], dtype=np.float32)
    latest = np.array([3.0], dtype=np.float32)
    audio_queue.put_nowait(first)
    audio_queue.put_nowait(second)

    assert _put_latest_preview(audio_queue, latest) is True
    np.testing.assert_array_equal(audio_queue.get_nowait(), second)
    np.testing.assert_array_equal(audio_queue.get_nowait(), latest)


def test_preview_queue_reports_no_drop_when_capacity_is_available():
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    chunk = np.array([1.0], dtype=np.float32)

    assert _put_latest_preview(audio_queue, chunk) is False
    np.testing.assert_array_equal(audio_queue.get_nowait(), chunk)
