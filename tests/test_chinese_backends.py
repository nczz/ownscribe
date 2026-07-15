"""Contract tests for the optional Chinese transcription backends."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from ownscribe.config import Config, FunASRConfig, resolve_model_path
from ownscribe.pipeline import _create_transcriber
from ownscribe.pipeline_live import _atomic_write, _post_transcribe
from ownscribe.transcription.breeze_transcriber import BreezeTranscriber
from ownscribe.transcription.firered_transcriber import FireRedTranscriber
from ownscribe.transcription.funasr_transcriber import FunASRTranscriber
from ownscribe.transcription.models import Segment, TranscriptResult
from ownscribe.transcription.utils import cluster_speaker_embeddings, load_audio_mono, quiet_model_output


def test_load_audio_mono_downmixes_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = np.ones(1600, dtype=np.float32)
    right = np.zeros(1600, dtype=np.float32)
    sf.write(path, np.column_stack([left, right]), 16000)

    audio, sample_rate = load_audio_mono(path, 16000)

    assert audio.shape == (1600,)
    assert sample_rate == 16000
    assert np.mean(audio) == pytest.approx(0.5, abs=1e-3)


def test_load_audio_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sf.write(path, np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="empty"):
        load_audio_mono(path)


def test_centroid_clustering_handles_invalid_and_updates_centroid() -> None:
    labels = cluster_speaker_embeddings(
        [np.array([1.0, 0.0]), np.array([0.8, 0.6]), np.array([0.0, 0.0]), None],
        0.7,
    )
    assert labels == [0, 0, None, None]


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1])
def test_centroid_clustering_validates_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        cluster_speaker_embeddings([], threshold)


def test_unknown_backend_fails_fast() -> None:
    config = Config()
    config.transcription.asr_backend = "breeeze"
    with pytest.raises(ValueError, match="Unknown transcription backend"):
        _create_transcriber(config)


def test_factory_propagates_diarization_contract() -> None:
    config = Config()
    config.transcription.asr_backend = "breeze"
    config.transcription.models_dir = "/models"
    config.diarization.enabled = False
    config.diarization.speaker_threshold = 0.75

    transcriber = _create_transcriber(config)

    assert transcriber._diarization_enabled is False
    assert transcriber._models_dir == "/models"
    assert transcriber._speaker_threshold == 0.75


def test_breeze_skips_diarization_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(4000, dtype=np.float32), 16000)
    transcriber = BreezeTranscriber(diarization_enabled=False)
    transcriber._model = object()
    transcriber._device = "cpu"
    transcriber._transcribe_chunked = MagicMock(return_value=[])
    transcriber._run_diarization = MagicMock(side_effect=AssertionError("must not run"))

    result = transcriber.transcribe(path)

    assert result.duration == pytest.approx(0.25)
    transcriber._run_diarization.assert_not_called()


def test_funasr_uses_native_speaker_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=FakeAutoModel))
    transcriber = FunASRTranscriber(FunASRConfig(spk_enabled=True))
    transcriber._load_model()

    assert captured["spk_model"]
    assert captured["output_timestamp"] is True


def test_funasr_does_not_configure_speaker_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=FakeAutoModel))
    FunASRTranscriber(FunASRConfig(spk_enabled=False))._load_model()
    assert "spk_model" not in captured


def test_quiet_model_output_restores_logging_on_failure() -> None:
    root = logging.getLogger()
    original = root.level
    with pytest.raises(RuntimeError), quiet_model_output():
        raise RuntimeError("boom")
    assert root.level == original


def test_firered_requires_explicit_checkout() -> None:
    with pytest.raises(RuntimeError, match="firered_repo"):
        FireRedTranscriber()._load_asr_system()


def test_firered_rejects_invalid_checkout(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Invalid"):
        FireRedTranscriber(firered_repo=str(tmp_path))._load_asr_system()


def test_firered_downloads_canonical_remote_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    transcriber = FireRedTranscriber(models_dir=str(tmp_path))
    monkeypatch.setattr("ownscribe.transcription.firered_transcriber.resolve_model_path", lambda *_: "org/model")
    downloader = MagicMock(return_value=str(tmp_path / "downloaded"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", downloader)

    assert transcriber._resolve_local_model("firered-vad") == tmp_path / "downloaded"
    downloader.assert_called_once_with(repo_id="org/model")


def test_model_alias_resolves_to_canonical_remote_id(tmp_path: Path) -> None:
    with patch("ownscribe.config.Path.exists", return_value=False):
        assert resolve_model_path("breeze-asr-25", str(tmp_path)) == "MediaTek-Research/Breeze-ASR-25"


def test_atomic_write_replaces_utf8_content(tmp_path: Path) -> None:
    destination = tmp_path / "transcript.md"
    destination.write_text("old", encoding="utf-8")
    _atomic_write(destination, "新的內容")
    assert destination.read_text(encoding="utf-8") == "新的內容"
    assert list(tmp_path.glob(".transcript.md.*")) == []


def test_post_transcribe_does_not_mutate_diarization_setting(tmp_path: Path) -> None:
    config = Config()
    config.diarization.enabled = False
    config.summarization.enabled = False
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    result = TranscriptResult(segments=[Segment(text="hello", start=0, end=1, words=[])], language="en", duration=1)
    transcriber = MagicMock()
    transcriber.transcribe.return_value = result

    with (
        patch("ownscribe.pipeline._create_transcriber", return_value=transcriber),
        patch("ownscribe.pipeline._format_output", return_value=("transcript", None)),
    ):
        _post_transcribe(config, audio, tmp_path)

    assert config.diarization.enabled is False
    assert (tmp_path / "transcript.md").read_text(encoding="utf-8") == "transcript"
