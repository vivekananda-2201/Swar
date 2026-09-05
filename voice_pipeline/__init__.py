"""
Voice Pipeline: Silero VAD + Parakeet TDT STT + Kokoro TTS (CPU Native)
Adapted from Hugging Face speech-to-speech architecture.
Includes:
- Decoupled TTS Generation and Playback
- Incremental <speak-out> Parser
- Progressive and Adaptive Chunking
"""
import logging
import warnings

# Suppress harmless PyTorch/Kokoro and upstream deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("speech_to_speech.VAD.vad_handler").setLevel(logging.ERROR)

# Apply upstream compatibility fixes in memory (works automatically in any fresh .venv)
from voice_pipeline.patches import apply_patches
apply_patches()

from voice_pipeline.vad import SileroVAD, VADChunk
from voice_pipeline.stt import ParakeetSTT, SmartProgressiveStreaming, PartialTranscription
from voice_pipeline.tts import KokoroTTS, TTSPipeline, AudioItem
from voice_pipeline.speak_out_parser import (
    IncrementalSpeakOutParser,
    RobustSentenceChunker,
    ChunkingConfig,
    TextSegment,
)
from voice_pipeline.config import (
    GeneralConfig,
    PipelineConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
)
from voice_pipeline.wake_word import (
    WakeWordConfig,
    WakeTriggerConfig,
    WakeWordEngine,
    WakeState,
)
from voice_pipeline.pipeline import VoicePipeline

__all__ = [
    "SileroVAD",
    "VADChunk",
    "ParakeetSTT",
    "SmartProgressiveStreaming",
    "PartialTranscription",
    "KokoroTTS",
    "TTSPipeline",
    "AudioItem",
    "IncrementalSpeakOutParser",
    "RobustSentenceChunker",
    "ChunkingConfig",
    "TextSegment",
    "VoicePipeline",
    "PipelineConfig",
    "GeneralConfig",
    "VADConfig",
    "STTConfig",
    "TTSConfig",
    "WakeWordConfig",
    "WakeTriggerConfig",
    "WakeWordEngine",
    "WakeState",
]
