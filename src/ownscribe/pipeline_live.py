"""Live meeting pipeline: real-time subtitles + recording + post-meeting transcription.

All-in-one flow:
1. Start recording (Core Audio system audio + mic)
2. Simultaneously stream audio to paraformer-zh-streaming for live subtitles
3. On Ctrl+C: stop recording, run SenseVoice + CAM++ for accurate transcript with speaker labels
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click
import numpy as np

from ownscribe.config import Config
from ownscribe.transcription.utils import quiet_model_output

_PREVIEW_START_TIMEOUT = 10.0
_PREVIEW_STOP_TIMEOUT = 2.0
_PREVIEW_TERMINATE_TIMEOUT = 2.0


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


def _put_latest_preview(audio_queue: queue.Queue[np.ndarray], chunk: np.ndarray) -> bool:
    """Queue a preview chunk without allowing inference lag to grow unbounded.

    Returns True when an old or new preview chunk had to be dropped.
    """
    try:
        audio_queue.put_nowait(chunk)
        return False
    except queue.Full:
        pass

    dropped = False
    try:
        audio_queue.get_nowait()
        dropped = True
    except queue.Empty:
        pass

    try:
        audio_queue.put_nowait(chunk)
    except queue.Full:
        return True
    return dropped


@dataclass
class _PreviewCapture:
    process: Any
    audio_queue: Any
    stop_event: Any
    dropped_chunks: Any


def _capture_preview_audio(
    audio_queue,
    status_connection,
    stop_event,
    dropped_chunks,
    sample_rate: int,
    chunk_stride: int,
    mic_device: int | None,
) -> None:
    """Capture microphone chunks in a process that can be killed if CoreAudio deadlocks."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    audio_queue.cancel_join_thread()

    def audio_callback(indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        if _put_latest_preview(audio_queue, chunk):
            dropped_chunks.value += 1

    try:
        import sounddevice as sd

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_stride,
            device=mic_device,
            callback=audio_callback,
        ):
            status_connection.send(None)
            stop_event.wait()
    except Exception as exc:
        with contextlib.suppress(BrokenPipeError, OSError):
            status_connection.send(f"{type(exc).__name__}: {exc}")
    finally:
        status_connection.close()


def _stop_preview_process(process, stop_event, timeout: float = _PREVIEW_STOP_TIMEOUT) -> bool:
    """Stop a preview worker within a bounded time, even if PortAudio is deadlocked."""
    stop_event.set()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(_PREVIEW_TERMINATE_TIMEOUT)
    if process.is_alive():
        process.kill()
        process.join(_PREVIEW_TERMINATE_TIMEOUT)
    return not process.is_alive()


def _close_preview_queue(audio_queue) -> None:
    with contextlib.suppress(OSError, ValueError):
        audio_queue.cancel_join_thread()
        audio_queue.close()


def _start_preview_capture(sample_rate: int, chunk_stride: int, mic_device: int | None) -> _PreviewCapture:
    context = mp.get_context("spawn")
    audio_queue = context.Queue(maxsize=8)
    stop_event = context.Event()
    dropped_chunks = context.Value("Q", 0, lock=False)
    status_receiver, status_sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_capture_preview_audio,
        args=(
            audio_queue,
            status_sender,
            stop_event,
            dropped_chunks,
            sample_rate,
            chunk_stride,
            mic_device,
        ),
        name="ownscribe-live-preview",
        daemon=True,
    )

    try:
        process.start()
    except Exception:
        status_receiver.close()
        status_sender.close()
        _close_preview_queue(audio_queue)
        raise
    status_sender.close()

    error: str | None
    try:
        if not status_receiver.poll(_PREVIEW_START_TIMEOUT):
            error = "timed out while opening the microphone input"
        else:
            try:
                error = status_receiver.recv()
            except EOFError:
                error = "preview worker exited before opening the microphone input"
    except BaseException:
        status_receiver.close()
        _stop_preview_process(process, stop_event)
        _close_preview_queue(audio_queue)
        if not process.is_alive():
            process.close()
        raise
    else:
        status_receiver.close()

    if error is not None:
        _stop_preview_process(process, stop_event)
        _close_preview_queue(audio_queue)
        if not process.is_alive():
            process.close()
        raise RuntimeError(f"Unable to start live preview: {error}")

    return _PreviewCapture(process, audio_queue, stop_event, dropped_chunks)


def _stop_preview_capture(capture: _PreviewCapture) -> None:
    stopped = False
    error: Exception | None = None
    try:
        stopped = _stop_preview_process(capture.process, capture.stop_event)
    except Exception as exc:
        error = exc
    finally:
        _close_preview_queue(capture.audio_queue)

    if stopped:
        with contextlib.suppress(OSError, ValueError):
            capture.process.close()
        return

    detail = f" ({error})" if error else ""
    click.echo(f"⚠️  無法回收即時音訊預覽程序{detail}；錄音封存仍會繼續。", err=True)


def _next_preview_chunk(capture: _PreviewCapture, timeout: float = 0.5) -> np.ndarray | None:
    """Return the next chunk, or fail instead of waiting forever for a dead worker."""
    try:
        return capture.audio_queue.get(timeout=timeout)
    except queue.Empty:
        if not capture.process.is_alive():
            raise RuntimeError("Live preview audio capture exited unexpectedly") from None
        return None


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

    # PortAudio can deadlock in AudioOutputUnitStop after macOS sleep/wake. Keep it
    # outside the main process so a stuck preview cannot block recording finalization.
    preview = _start_preview_capture(sample_rate, chunk_stride, mic_device)

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
        if recorder and audio_path:
            recorder.start(audio_path)

        while not stop_event.is_set():
            chunk = _next_preview_chunk(preview)
            if chunk is None:
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
            click.echo("\n⏳ 正在停止即時音訊預覽...")
            _stop_preview_capture(preview)
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
                click.echo("⏳ 正在完成錄音檔...")
                recorder.stop()

    # --- Stop recording ---
    dropped_live_chunks = preview.dropped_chunks.value
    duration = time.time() - start_time
    click.echo()
    click.echo("=" * 60)
    click.echo(f"⏱️  會議時長: {timedelta(seconds=int(duration))}")
    click.echo(f"📝 即時辨識句數: {line_count}")
    if dropped_live_chunks:
        click.echo(
            f"⚠️  即時字幕處理落後，已略過 {dropped_live_chunks} 個預覽音訊區塊；完整錄音不受影響。",
            err=True,
        )

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
