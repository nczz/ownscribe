"""Behavioral tests for the Swift CoreAudio helper wrapper."""

from __future__ import annotations

import time
from unittest import mock

import pytest

from ownscribe.audio import coreaudio
from ownscribe.audio.coreaudio import CoreAudioRecorder


def test_large_helper_stderr_cannot_block_shutdown(tmp_path, capsys):
    helper = tmp_path / "fake-ownscribe-audio"
    helper.write_text(
        """#!/usr/bin/env python3
import signal
import sys
import time
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
ready = Path(str(output) + ".ready")

def stop(_signum, _frame):
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
for _ in range(100_000):
    sys.stderr.write("Recording filler line that must not fill a parent pipe.\\n")
sys.stderr.write("[SILENCE_WARNING] synthetic helper warning\\n")
sys.stderr.write("[SILENCE_TIMEOUT]\\n")
sys.stderr.flush()
ready.write_text("ready")
while True:
    time.sleep(0.1)
"""
    )
    helper.chmod(0o755)

    with mock.patch("ownscribe.audio.coreaudio._find_binary", return_value=helper):
        recorder = CoreAudioRecorder()

    output = tmp_path / "recording.wav"
    ready = tmp_path / "recording.wav.ready"
    recorder.start(output)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and recorder.is_recording and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "helper blocked while writing stderr"
    finally:
        recorder.stop()

    captured = capsys.readouterr()
    assert recorder.silence_warning is True
    assert recorder.silence_timed_out is True
    assert recorder.is_recording is False
    assert "synthetic helper warning" in captured.err
    assert "filler line" not in captured.err

    # Repeated cleanup is safe and does not re-read or re-emit the temporary log.
    recorder.stop()
    assert capsys.readouterr().err == ""


def test_find_binary_rejects_stale_helper_protocol(tmp_path):
    stale = tmp_path / "stale-helper"
    current = tmp_path / "current-helper"
    stale.write_text("#!/bin/sh\nprintf '1\\n'\n")
    current.write_text("#!/bin/sh\nprintf '2\\n'\n")
    stale.chmod(0o755)
    current.chmod(0o755)

    with (
        mock.patch.object(coreaudio, "_BINARY_CANDIDATES", [stale]),
        mock.patch.object(coreaudio, "_CACHE_DIR", tmp_path / "cache"),
        mock.patch("ownscribe.audio.coreaudio.shutil.which", return_value=str(current)),
        mock.patch("ownscribe.audio.coreaudio._download_binary") as download,
    ):
        assert coreaudio._find_binary() == current

    download.assert_not_called()


def test_nonzero_helper_exit_is_not_treated_as_success(tmp_path, capsys):
    helper = tmp_path / "failing-helper"
    helper.write_text(
        """#!/bin/sh
if [ "$1" = "version" ]; then
    printf '2\\n'
    exit 0
fi
printf 'capture failed decisively\\n' >&2
exit 7
"""
    )
    helper.chmod(0o755)

    with mock.patch("ownscribe.audio.coreaudio._find_binary", return_value=helper):
        recorder = CoreAudioRecorder()

    recorder.start(tmp_path / "recording.wav")
    deadline = time.monotonic() + 5
    while recorder.is_recording and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="status 7"):
        recorder.stop()

    assert "capture failed decisively" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capture_mode": "unknown"}, "capture_mode"),
        ({"silence_timeout": -1}, "silence_timeout"),
    ],
)
def test_invalid_capture_configuration_fails_before_start(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CoreAudioRecorder(**kwargs)
