"""Live meeting pipeline: real-time subtitles + recording + post-meeting transcription.

All-in-one flow:
1. Start recording (Core Audio system audio + mic)
2. Simultaneously stream audio to paraformer-zh-streaming for live subtitles
3. On Ctrl+C: stop recording, run SenseVoice + CAM++ for accurate transcript with speaker labels
"""

from __future__ import annotations

import os
import queue
import signal
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np

from ownscribe.config import Config
from ownscribe.transcription.utils import quiet_model_output


def _load_streaming_model(config: Config):
    """Load the configured live model without mutating global logging or tqdm state."""
    from funasr import AutoModel

    from ownscribe.config import resolve_model_path

    model = resolve_model_path("paraformer-zh-streaming", config.transcription.models_dir)
    with quiet_model_output():
        return AutoModel(model=str(model), disable_update=True)


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text artifact in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_live_pipeline(
    config: Config,
    *,
    mic_device: int | None = None,
    language: str = "zh",
    record: bool = True,
) -> None:
    """Run the live meeting pipeline: real-time preview + recording + post-transcription."""

    # Apple Silicon thread optimization
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

    # --- Setup output directory ---
    out_dir: Path | None = None
    audio_path: Path | None = None
    if record:
        base = config.output.resolved_dir
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = base / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "recording.wav"

    # --- Load streaming model ---
    click.echo("⏳ 載入即時辨識模型...")

    streaming_model = _load_streaming_model(config)

    click.echo("✅ 即時辨識模型就緒")

    # --- Setup Traditional Chinese converter ---
    _cc = None
    try:
        from opencc import OpenCC

        _cc = OpenCC("s2twp")
    except ImportError:
        pass

    def _to_trad(text: str) -> str:
        return _cc.convert(text) if _cc and text else text

    # --- Setup recorder ---
    recorder = None
    if record:
        recorder = _create_live_recorder(config)
        click.echo(f"📁 錄音將存到: {audio_path}")

    click.echo()
    click.echo("=" * 60)
    click.echo("🎙️  即時會議模式 (Ctrl+C 結束)")
    if record:
        click.echo("   ✅ 即時字幕  ✅ 錄音中  ✅ 會後轉錄")
    else:
        click.echo("   ✅ 即時字幕  ❌ 不錄音")
    click.echo("=" * 60)
    click.echo()

    # --- Streaming parameters ---
    sample_rate = 16000
    chunk_size = [0, 10, 5]  # 600ms chunks
    chunk_stride = chunk_size[1] * 960  # 9600 samples @ 16kHz
    cache = {}

    # --- Audio capture via sounddevice (for streaming preview) ---
    import sounddevice as sd

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    # --- Start recording (background) ---
    if recorder and audio_path:
        recorder.start(audio_path)

    # --- Main loop: live streaming ---
    start_time = time.time()
    stop_event = threading.Event()
    line_count = 0
    buffer = np.array([], dtype=np.float32)

    def on_interrupt(sig, frame):
        stop_event.set()

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_interrupt)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_stride,
            device=mic_device,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                buffer = np.concatenate([buffer, chunk])

                while len(buffer) >= chunk_stride:
                    audio_chunk = buffer[:chunk_stride]
                    buffer = buffer[chunk_stride:]

                    with quiet_model_output():
                        res = streaming_model.generate(
                            input=audio_chunk,
                            cache=cache,
                            is_final=False,
                            chunk_size=chunk_size,
                            encoder_chunk_look_back=4,
                            decoder_chunk_look_back=1,
                        )

                    if res and res[0] and res[0].get("text"):
                        text = res[0]["text"].strip()
                        if text:
                            text = _to_trad(text)
                            elapsed = time.time() - start_time
                            ts = str(timedelta(seconds=int(elapsed)))
                            click.echo(f"  [{ts}] {text}")
                            line_count += 1

    finally:
        signal.signal(signal.SIGINT, original_handler)
        try:
            # Flush final chunk.
            if len(buffer) > 0:
                with quiet_model_output():
                    res = streaming_model.generate(
                        input=buffer,
                        cache=cache,
                        is_final=True,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=4,
                        decoder_chunk_look_back=1,
                    )
                if res and res[0] and res[0].get("text"):
                    text = res[0]["text"].strip()
                    if text:
                        text = _to_trad(text)
                        elapsed = time.time() - start_time
                        ts = str(timedelta(seconds=int(elapsed)))
                        click.echo(f"  [{ts}] {text}")
                        line_count += 1
        finally:
            if recorder:
                recorder.stop()

    # --- Stop recording ---
    duration = time.time() - start_time
    click.echo()
    click.echo("=" * 60)
    click.echo(f"⏱️  會議時長: {timedelta(seconds=int(duration))}")
    click.echo(f"📝 即時辨識句數: {line_count}")

    if recorder:
        click.echo(f"💾 錄音已保存: {audio_path}")
        click.echo("=" * 60)
        click.echo()

        # --- Post-meeting: accurate transcription with speaker diarization ---
        if audio_path and audio_path.exists() and audio_path.stat().st_size > 44:
            mode = "with speaker diarization" if config.diarization.enabled else "without speaker diarization"
            click.echo(f"Starting accurate transcription ({config.transcription.asr_backend}, {mode})...")
            click.echo()
            _post_transcribe(config, audio_path, out_dir)
        else:
            click.echo("⚠️  錄音檔案為空，跳過轉錄。", err=True)
    else:
        click.echo("=" * 60)


def _create_live_recorder(config: Config):
    """Create recorder for the live pipeline."""
    if config.audio.backend == "coreaudio":
        from ownscribe.audio.coreaudio import CoreAudioRecorder

        recorder = CoreAudioRecorder(
            mic=config.audio.mic,
            mic_device=config.audio.mic_device,
            capture_mode=config.audio.capture_mode,
            silence_timeout=0,  # No auto-stop in live mode
        )
        if recorder.is_available():
            return recorder

    from ownscribe.audio.sounddevice_recorder import SoundDeviceRecorder

    return SoundDeviceRecorder(device=None, silence_timeout=0)


def _post_transcribe(config: Config, audio_path: Path, out_dir: Path) -> None:
    """Run post-meeting accurate transcription using the configured backend."""
    from ownscribe.pipeline import _format_output, _transcribe_audio

    result = _transcribe_audio(config, audio_path)

    # Save transcript
    transcript_str, _ = _format_output(config, result)
    ext = "json" if config.output.format == "json" else "md"
    transcript_path = out_dir / f"transcript.{ext}"
    _atomic_write(transcript_path, transcript_str)

    click.echo("✅ 轉錄完成！")
    click.echo(f"📄 逐字稿: {transcript_path}")

    # Show stats
    speakers = set(seg.speaker for seg in result.segments if seg.speaker)
    click.echo(f"   說話者: {len(speakers)} 人")
    click.echo(f"   總句數: {len(result.segments)}")
    click.echo(f"   語言: {result.language}")
    click.echo()

    # Print preview of the transcript
    click.echo("--- 轉錄預覽（前 10 句）---")
    for seg in result.segments[:10]:
        start = str(timedelta(seconds=int(seg.start)))
        speaker = seg.speaker or "?"
        click.echo(f"  [{start}] {speaker}: {seg.text}")
    if len(result.segments) > 10:
        click.echo(f"  ... 共 {len(result.segments)} 句")

    # Optional: run summarization
    if config.summarization.enabled:
        try:
            from ownscribe.output.markdown import format_summary
            from ownscribe.summarization import create_summarizer

            summarizer = create_summarizer(config)
            try:
                if summarizer.is_available():
                    click.echo()
                    click.echo("🤖 生成摘要中...")
                    summary = summarizer.summarize(result.full_text)
                    summary_path = out_dir / f"summary.{ext}"
                    summary_str = format_summary(summary) if ext == "md" else summary
                    _atomic_write(summary_path, summary_str)
                    click.echo(f"📋 摘要: {summary_path}")
            finally:
                summarizer.close()
        except Exception as exc:
            click.echo(f"⚠️  摘要失敗: {exc}", err=True)
