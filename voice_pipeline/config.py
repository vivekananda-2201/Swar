"""
Configuration System for Voice Pipeline.

Provides strongly-typed dataclasses for all pipeline subsystems:
- General / Audio Hardware / Logging settings
- VAD (Silero Voice Activity Detection)
- STT (NVIDIA Parakeet TDT 0.6B)
- TTS (Kokoro-82M, decoupled workers, queue depths)
- Chunking (Progressive & Adaptive sentence batching)
- Agent (LLM, tags, screen routing)

Can be loaded from or exported to YAML or JSON files, or customized in code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from voice_pipeline.speak_out_parser import ChunkingConfig
from voice_pipeline.wake_word import WakeTriggerConfig, WakeWordConfig


@dataclass
class GeneralConfig:
    """General runtime and hardware configuration."""
    device: str = "cpu"                  # Compute device: "cpu" or "cuda"
    cpu_threads: int = 4                 # PyTorch intra-op CPU threads (tuned for physical P-cores, avoiding E-core stalls)
    verbose: bool = False                # If False, pipeline is completely silent (no terminal prints)
    allow_barge_in: bool = True          # If True, user speech interrupts playback. If False, assistant finishes speaking.
    input_device: Optional[int | str] = None   # Audio input device ID or name (None = default)
    output_device: Optional[int | str] = None  # Audio output device ID or name (None = default)
    wake_mode: bool = False                    # If True, requires wake word to activate


@dataclass
class VADConfig:
    """Silero Voice Activity Detection configuration."""
    threshold: float = 0.6               # Speech probability threshold (0.0 to 1.0)
    min_silence_ms: int = 800            # Silence duration before closing user turn
    min_speech_ms: int = 384             # Minimum speech duration to register as speech
    min_speech_continuation_ms: int = 192 # Continuity window during brief pauses
    speech_pad_ms: int = 500             # Pre-speech buffer in ms (retains first words)
    progressive_interval: float = 0.5    # Interval in seconds for live interim transcripts


@dataclass
class STTConfig:
    """Speech-to-Text configuration (NVIDIA Parakeet TDT)."""
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3" # Model identifier / HF hub path
    language: str = "en"                            # Language code
    enable_live_transcription: bool = True          # Emit progressive transcripts while speaking


@dataclass
class TTSConfig:
    """Kokoro Text-to-Speech and worker thread configuration."""
    voice: str = "af_bella"              # Voice name (e.g., af_bella, af_sarah, am_adam, bf_emma)
    speed: float = 1.0                   # Speech speed multiplier (0.5 to 2.0)
    lang_code: str = "a"                 # 'a' for American English, 'b' for British English
    sample_rate: int = 24000             # Audio sample rate in Hz
    max_text_queue_size: int = 100       # Max pending text chunks in generation queue
    max_audio_queue_size: int = 50       # Max pending audio items in playback queue


@dataclass
class PipelineConfig:
    """Unified master configuration for the entire voice pipeline."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Builds a PipelineConfig from a dictionary."""
        general_data = data.get("general", {})
        vad_data = data.get("vad", {})
        stt_data = data.get("stt", {})
        tts_data = data.get("tts", {})
        chunking_data = data.get("chunking", {})
        wake_word_data = data.get("wake_word", {})

        # If wake_mode is set in general, sync it to wake_word.enabled
        if "wake_mode" in general_data and "enabled" not in wake_word_data:
            wake_word_data["enabled"] = bool(general_data.get("wake_mode"))
        elif "wake_mode" in data and "enabled" not in wake_word_data:
            wake_word_data["enabled"] = bool(data.get("wake_mode"))

        # Handle progressive_stages tuples if loaded as lists from YAML/JSON
        if "progressive_stages" in chunking_data:
            chunking_data["progressive_stages"] = [
                tuple(stage) for stage in chunking_data["progressive_stages"]
            ]

        return cls(
            general=GeneralConfig(**general_data),
            vad=VADConfig(**vad_data),
            stt=STTConfig(**stt_data),
            tts=TTSConfig(**tts_data),
            chunking=ChunkingConfig(**chunking_data),
            wake_word=WakeWordConfig.from_dict(wake_word_data),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Loads configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Loads configuration from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def load_default_or_file(cls, path: Optional[Union[str, Path]] = None) -> "PipelineConfig":
        """
        Attempts to load from path or 'config.yaml' in current directory.
        If no file exists, returns default PipelineConfig().
        """
        target_path = Path(path) if path else Path("config.yaml")
        if target_path.is_file():
            return cls.from_yaml(target_path)
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a dictionary."""
        return asdict(self)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Converts configuration to YAML string, optionally writing to path."""
        data = self.to_dict()
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str
