"""Configuration management with TOML loading and defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path("~/.config/ownscribe").expanduser()
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG_TOML = """\
[audio]
backend = "coreaudio"     # "coreaudio" (default) or "sounddevice"
device = ""               # empty = system audio; or device name/index for sounddevice
mic = false               # also capture microphone input
mic_device = ""           # specific mic device name (empty = default)
capture_mode = "picker"   # "picker" = show source picker; "all" = capture all system audio directly
silence_timeout = 300     # seconds of silence before auto-stop; 0 = disabled

[transcription]
model = "base"            # whisper model: tiny, base, small, medium, large-v3
language = ""             # empty = auto-detect
# initial_prompt = ""     # prime Whisper with context: domain vocab, speaker names, expected phrases
# hotwords = ""           # comma-separated words to boost recognition (softer hint than initial_prompt)
# asr_backend = "whisperx"  # "whisperx" (default) or "funasr" (better Chinese, built-in diarization)
# funasr_model = "sensevoice"  # "sensevoice" (fastest, multilingual), "paraformer" (Chinese + timestamps)
# models_dir = "~/.cache/ownscribe/models"
# firered_repo = ""          # required for FireRed; path to a FireRedASR2S checkout
# chunk_seconds = 60          # bounded-memory decode window; must be >= 30

[diarization]
enabled = false           # speaker diarization is opt-in
hf_token = ""             # HuggingFace token for pyannote models
backend = "auto"          # Community-1 with token; native backend diarization otherwise
min_speakers = 0          # 0 = auto-detect
max_speakers = 8          # global safety cap; set 0 for no upper bound
telemetry = false         # set to true to allow HuggingFace Hub + pyannote metrics telemetry
device = "mps"            # Apple Silicon default; use "cpu" for compatibility fallback
speaker_threshold = 0.7    # CAM++ cosine threshold for Breeze/FireRed
window_seconds = 600       # bounded Community-1 window
window_overlap_seconds = 30
community_speaker_threshold = 0.55
segmentation_batch_size = 4
embedding_batch_size = 8

[summarization]
enabled = true
backend = "local"         # "local" (built-in, no server needed), "ollama", or "openai"
model = "phi-4-mini"      # local: "phi-4-mini", path to GGUF, or hf:owner/repo/file.gguf; ollama/openai: model name
# host = "http://localhost:11434"  # only for ollama/openai backends
# api_key = ""            # only for openai backend; required by servers like oMLX (or set OPENAI_API_KEY)
# template = "meeting"    # built-in: "meeting", "lecture", or "brief"
# context_size = 0        # 0 = auto-detect from model; set manually for OpenAI-compatible backends

# Custom templates (optional):
# [templates.my-notes]
# system_prompt = "You are a helpful assistant."
# prompt = "Summarize:\\n{transcript}"

[output]
dir = "~/ownscribe"       # base output directory
format = "markdown"       # "markdown" or "json"
keep_recording = true     # keep WAV files after transcription; false = auto-delete
"""


@dataclass
class AudioConfig:
    backend: str = "coreaudio"
    device: str = ""
    mic: bool = False
    mic_device: str = ""
    capture_mode: str = "picker"  # "picker" = show source picker; "all" = all system audio
    silence_timeout: int = 300  # seconds of silence before auto-stop; 0 = disabled


@dataclass
class TranscriptionConfig:
    model: str = "base"
    language: str = ""
    initial_prompt: str = ""
    hotwords: str = ""
    asr_backend: str = "whisperx"  # "whisperx" (default) or "funasr" or "firered" or "breeze"
    funasr_model: str = "sensevoice"  # "sensevoice", "paraformer", "paraformer-en"
    models_dir: str = "~/.cache/ownscribe/models"  # unified model storage directory
    firered_repo: str = ""  # path to a compatible FireRedASR2S checkout
    chunk_seconds: int = 60


@dataclass
class DiarizationConfig:
    enabled: bool = False
    hf_token: str = ""
    backend: str = "auto"
    min_speakers: int = 0
    max_speakers: int = 8
    telemetry: bool = False
    device: str = "mps"
    speaker_threshold: float = 0.7
    window_seconds: int = 600
    window_overlap_seconds: int = 30
    community_speaker_threshold: float = 0.55
    segmentation_batch_size: int = 4
    embedding_batch_size: int = 8


@dataclass
class SummarizationConfig:
    enabled: bool = True
    backend: str = "local"
    model: str = "phi-4-mini"
    host: str = "http://localhost:11434"
    api_key: str = ""
    template: str = ""
    context_size: int = 0


@dataclass
class TemplateConfig:
    system_prompt: str = ""
    prompt: str = ""


@dataclass
class OutputConfig:
    dir: str = "~/ownscribe"
    format: str = "markdown"
    keep_recording: bool = True

    @property
    def resolved_dir(self) -> Path:
        return Path(self.dir).expanduser()


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    templates: dict[str, TemplateConfig] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Config:
        """Load config from TOML file, falling back to defaults."""
        config = cls()

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
            config = _merge_toml(config, data)

        # Env var overrides
        if hf_token := os.environ.get("HF_TOKEN"):
            config.diarization.hf_token = hf_token
        if ollama_host := os.environ.get("OLLAMA_HOST"):
            config.summarization.host = ollama_host
        if api_key := os.environ.get("OPENAI_API_KEY"):
            config.summarization.api_key = api_key

        return config


def _merge_toml(config: Config, data: dict) -> Config:
    """Merge TOML data into config dataclass."""
    if "audio" in data:
        for k, v in data["audio"].items():
            if hasattr(config.audio, k):
                setattr(config.audio, k, v)

    if "transcription" in data:
        for k, v in data["transcription"].items():
            if hasattr(config.transcription, k):
                setattr(config.transcription, k, v)

    if "diarization" in data:
        for k, v in data["diarization"].items():
            if hasattr(config.diarization, k):
                setattr(config.diarization, k, v)

    if "summarization" in data:
        for k, v in data["summarization"].items():
            if hasattr(config.summarization, k):
                setattr(config.summarization, k, v)

    if "output" in data:
        for k, v in data["output"].items():
            if hasattr(config.output, k):
                setattr(config.output, k, v)

    if "templates" in data:
        for name, t_data in data["templates"].items():
            config.templates[name] = TemplateConfig(
                system_prompt=t_data.get("system_prompt", ""),
                prompt=t_data.get("prompt", ""),
            )

    return config


@dataclass
class FunASRConfig:
    """Configuration for the FunASR transcription backend.

    This is an alternative to TranscriptionConfig + DiarizationConfig,
    providing a unified pipeline with better Chinese recognition.
    """

    model: str = "sensevoice"  # "sensevoice", "paraformer", "paraformer-en", or full model ID
    language: str = ""  # "" = auto-detect; "zh", "en", "ja", "ko", "yue"
    device: str = "cpu"  # "cpu" or "cuda"
    spk_enabled: bool = True  # Enable CAM++ speaker diarization (no HF token needed)
    batch_size_s: int = 300  # Seconds per batch (higher = more memory, faster)
    hotwords: str = ""  # Comma-separated hotwords (Paraformer SeACo only)
    traditional_chinese: bool = True  # Convert output to Traditional Chinese (Taiwan)
    models_dir: str = "~/.cache/ownscribe/models"
    speaker_threshold: float = 0.7
    chunk_seconds: int = 60


_MODEL_IDS = {
    "breeze-asr-25": "MediaTek-Research/Breeze-ASR-25",
    "sensevoice": "iic/SenseVoiceSmall",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
    "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "firered-vad": "FireRedTeam/FireRedVAD",
    "firered-lid": "FireRedTeam/FireRedLID",
    "firered-asr2-aed": "FireRedTeam/FireRedASR2-AED",
    "firered-punc": "FireRedTeam/FireRedPunc",
}


def resolve_model_path(name: str, models_dir: str | None = None) -> Path | str:
    """Resolve a model name to its local path.

    Looks in the project's models/ directory first, then the configured models_dir,
    then falls back to the name itself (for ModelScope/HuggingFace auto-download IDs).

    Usage:
        path = resolve_model_path("sensevoice")
        path = resolve_model_path("breeze-asr-25")
        path = resolve_model_path("firered-asr2-aed")
    """
    # 1. Project models/ directory (follows symlinks)
    project_models = Path(__file__).parent.parent.parent / "models"
    local = project_models / name
    if local.exists():
        return local.resolve()  # resolve symlinks to actual path

    # 2. Configured models_dir
    if models_dir:
        configured = Path(models_dir).expanduser() / name
        if configured.exists():
            return configured.resolve()

    # 3. Maybe it's already an absolute path
    p = Path(name).expanduser()
    if p.exists():
        return p

    # 4. Return the canonical remote model ID.
    return _MODEL_IDS.get(name, name)


def ensure_config_file() -> Path:
    """Create default config file if it doesn't exist. Returns the path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML)
    return CONFIG_PATH
