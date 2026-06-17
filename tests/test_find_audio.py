"""Tests for media-file discovery in _find_audio (audio + video containers)."""

from __future__ import annotations

from ownscribe.pipeline import _find_audio


def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"")


class TestFindAudioVideo:
    def test_finds_mp4_when_no_recording_wav(self, tmp_path):
        _touch(tmp_path, "meeting.mp4")
        assert _find_audio(tmp_path) == tmp_path / "meeting.mp4"

    def test_finds_mov(self, tmp_path):
        _touch(tmp_path, "meeting.mov")
        assert _find_audio(tmp_path) == tmp_path / "meeting.mov"

    def test_finds_mkv(self, tmp_path):
        _touch(tmp_path, "meeting.mkv")
        assert _find_audio(tmp_path) == tmp_path / "meeting.mkv"

    def test_prefers_recording_wav_over_video(self, tmp_path):
        _touch(tmp_path, "meeting.mp4", "recording.wav")
        assert _find_audio(tmp_path) == tmp_path / "recording.wav"

    def test_case_insensitive_extension(self, tmp_path):
        _touch(tmp_path, "Meeting.MP4")
        assert _find_audio(tmp_path) == tmp_path / "Meeting.MP4"

    def test_ignores_unrelated_files(self, tmp_path):
        _touch(tmp_path, "notes.txt", "slides.pdf")
        assert _find_audio(tmp_path) is None
