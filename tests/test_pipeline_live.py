"""Behavioral tests for bounded live-preview buffering."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np
import pytest

from ownscribe.pipeline_live import (
    _next_preview_chunk,
    _PreviewCapture,
    _put_latest_preview,
    _stop_preview_capture,
    _stop_preview_process,
)


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


def test_preview_shutdown_terminates_worker_that_cannot_stop_cleanly():
    context = mp.get_context("spawn")
    stop_event = context.Event()
    process = context.Process(target=time.sleep, args=(60,))
    process.start()

    started = time.monotonic()
    try:
        assert _stop_preview_process(process, stop_event, timeout=0.05) is True
        assert time.monotonic() - started < 3
        assert process.exitcode is not None
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def test_preview_read_fails_when_worker_has_exited():
    context = mp.get_context("spawn")
    process = context.Process(target=time.sleep, args=(0,))
    audio_queue = context.Queue(maxsize=1)
    capture = _PreviewCapture(process, audio_queue, context.Event(), context.Value("Q", 0, lock=False))
    process.start()
    process.join()

    try:
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            _next_preview_chunk(capture, timeout=0.01)
    finally:
        _stop_preview_capture(capture)
