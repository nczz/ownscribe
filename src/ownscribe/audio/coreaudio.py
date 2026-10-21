"""Core Audio Taps recorder — wraps the Swift ownscribe-audio helper."""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

import click

from ownscribe.audio.base import AudioRecorder

# Timeouts for graceful shutdown of the Swift helper.
# SIGINT triggers track merging (system + mic) which can take a while for long recordings.
# wait() returns immediately on exit, so generous values cost nothing in the normal case.
_STOP_TIMEOUT = 300  # seconds to wait after SIGINT (allows long-track merging)
_KILL_TIMEOUT = 10  # seconds to wait after SIGTERM before SIGKILL
_HELPER_PROTOCOL_VERSION = "2"

_NOISE_PREFIXES = ("Recording ", "Saved ", "Merged audio saved")
_NOISE_LINES = ("[MIC_MUTED]", "[MIC_UNMUTED]", "[SILENCE_TIMEOUT]")

# Look for binary relative to package, then in PATH
_BINARY_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "bin" / "ownscribe-audio",  # dev: repo root
    Path(sys.prefix) / "bin" / "ownscribe-audio",
]

_CACHE_DIR = Path.home() / ".local" / "share" / "ownscribe" / "bin"
_DOWNLOAD_URL = "https://github.com/nczz/ownscribe/releases/latest/download/ownscribe-audio-{arch}"


def _helper_is_compatible(path: Path) -> bool:
    """Return whether an executable implements the wrapper's required protocol."""
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        result = subprocess.run(
            [str(path), "version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == _HELPER_PROTOCOL_VERSION


def _download_binary() -> Path | None:
    """Atomically download and validate the prebuilt helper."""
    if sys.platform != "darwin":
        return None

    arch = platform.machine()
    if arch not in ("arm64", "x86_64"):
        return None

    url = _DOWNLOAD_URL.format(arch=arch)
    dest = _CACHE_DIR / "ownscribe-audio"
    temp_path: Path | None = None

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        click.echo(f"Downloading ownscribe-audio ({arch}) from GitHub Releases...")
        with tempfile.NamedTemporaryFile(
            dir=_CACHE_DIR,
            prefix=".ownscribe-audio-",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        urllib.request.urlretrieve(url, temp_path)
        temp_path.chmod(0o755)
        if not _helper_is_compatible(temp_path):
            raise RuntimeError("downloaded helper is incompatible")
        temp_path.replace(dest)
        temp_path = None
        click.echo(f"Saved to {dest}")
        return dest
    except Exception as error:
        click.echo(f"Download failed: {error}", err=True)
        return None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _find_binary() -> Path | None:
    for candidate in _BINARY_CANDIDATES:
        if _helper_is_compatible(candidate):
            return candidate

    cached = _CACHE_DIR / "ownscribe-audio"
    if _helper_is_compatible(cached):
        return cached

    found = shutil.which("ownscribe-audio")
    if found and _helper_is_compatible(Path(found)):
        return Path(found)

    return _download_binary()


class CoreAudioRecorder(AudioRecorder):
    """Records system audio using the ownscribe-audio Swift helper."""

    def __init__(
        self,
        mic: bool = False,
        mic_device: str = "",
        capture_mode: str = "picker",
        silence_timeout: int = 0,
    ) -> None:
        if capture_mode not in {"picker", "all"}:
            raise ValueError("capture_mode must be 'picker' or 'all'")
        if silence_timeout < 0:
            raise ValueError("silence_timeout must be zero or positive")
        self._mic = mic
        self._mic_device = mic_device
        self._capture_mode = capture_mode
        self._silence_timeout = silence_timeout
        self._process: subprocess.Popen | None = None
        self._binary = _find_binary()
        self._silence_warning: bool = False
        self._silence_timed_out: bool = False
        self._muted: bool = False
        self._stderr_file: BinaryIO | None = None

    def is_available(self) -> bool:
        return self._binary is not None

    def start(self, output_path: Path) -> None:
        if self._process is not None:
            raise RuntimeError("Recorder has already been started; call stop() before restarting it")

        self._silence_warning = False
        self._silence_timed_out = False
        self._muted = False

        if not self._binary:
            raise RuntimeError("ownscribe-audio binary not found. Run: bash swift/build.sh")

        cmd = [str(self._binary), "capture", "--output", str(output_path)]
        if self._capture_mode == "all":
            cmd.append("--capture-mode-all")
        if self._mic or self._mic_device:
            cmd.append("--mic")
        if self._mic_device:
            cmd.extend(["--mic-device", self._mic_device])
        if self._silence_timeout > 0:
            cmd.extend(["--silence-timeout", str(self._silence_timeout)])

        # A pipe can fill while the parent is waiting for a long recording to finish,
        # deadlocking both processes. A temporary file keeps stderr drained by the OS
        # without an unbounded in-memory buffer or a reader thread.
        stderr_file = tempfile.TemporaryFile()  # noqa: SIM115 - owned until stop()
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                process_group=0,
            )
        except Exception:
            stderr_file.close()
            raise
        self._stderr_file = stderr_file

    @property
    def silence_warning(self) -> bool:
        """True if the Swift helper reported a silence warning."""
        return self._silence_warning

    def toggle_mute(self) -> None:
        """Send SIGUSR1 to the Swift helper to toggle mic mute."""
        if self._mic and self._process and self._process.poll() is None:
            try:
                self._process.send_signal(signal.SIGUSR1)
            except ProcessLookupError:
                return
            self._muted = not self._muted

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def silence_timed_out(self) -> bool:
        return self._silence_timed_out

    def stop(self) -> None:
        process = self._process
        try:
            if process and process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=_STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=_KILL_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            self._process = None
            self._muted = False
            self._consume_stderr()

        if process is not None and process.returncode not in (None, 0):
            raise RuntimeError(f"ownscribe-audio exited with status {process.returncode}")

    def _consume_stderr(self) -> None:
        """Parse helper status markers and relay actionable diagnostics."""
        stderr_file = self._stderr_file
        self._stderr_file = None
        if stderr_file is None:
            return

        try:
            stderr_file.flush()
            stderr_file.seek(0)
            for raw_line in stderr_file:
                line = raw_line.decode(errors="replace").rstrip("\r\n")
                if not line:
                    continue
                if "[SILENCE_WARNING]" in line:
                    self._silence_warning = True
                if "[SILENCE_TIMEOUT]" in line:
                    self._silence_timed_out = True
                if line not in _NOISE_LINES and not line.startswith(_NOISE_PREFIXES):
                    click.echo(line, err=True)
        finally:
            stderr_file.close()

    def list_devices(self) -> str:
        """List available audio devices using the Swift helper."""
        if not self._binary:
            return "ownscribe-audio binary not found. Run: bash swift/build.sh"
        result = subprocess.run(
            [str(self._binary), "list-devices"],
            capture_output=True,
            text=True,
        )
        return result.stdout

    def list_apps(self) -> str:
        if not self._binary:
            return "ownscribe-audio binary not found. Run: bash swift/build.sh"
        result = subprocess.run(
            [str(self._binary), "list-apps"],
            capture_output=True,
            text=True,
        )
        return result.stdout
