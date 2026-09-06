"""
Swar (स्वर) — Local Conversational Audio Runtime.
100% Offline, CPU-First Conversational Audio Orchestration Layer.

Provides clean top-level imports:
    from swar import VoicePipeline, PipelineConfig, VoiceprintManager
"""

from voice_pipeline import (
    SileroVAD,
    VADChunk,
    ParakeetSTT,
    SmartProgressiveStreaming,
    PartialTranscription,
    KokoroTTS,
    TTSPipeline,
    AudioItem,
    IncrementalSpeakOutParser,
    RobustSentenceChunker,
    ChunkingConfig,
    TextSegment,
    VoicePipeline,
    PipelineConfig,
    GeneralConfig,
    VADConfig,
    STTConfig,
    TTSConfig,
    WakeWordConfig,
    WakeTriggerConfig,
    WakeWordEngine,
    WakeState,
    BenchmarkLogger,
    BargeInGuardConfig,
    PipeWireAECConfig,
    VoiceprintManager,
    SpeakerEmbeddingExtractor,
    PipeWireAECManager,
)

__all__ = [
    "VoicePipeline",
    "PipelineConfig",
    "GeneralConfig",
    "VADConfig",
    "STTConfig",
    "TTSConfig",
    "ChunkingConfig",
    "WakeWordConfig",
    "WakeTriggerConfig",
    "WakeWordEngine",
    "WakeState",
    "BargeInGuardConfig",
    "PipeWireAECConfig",
    "VoiceprintManager",
    "SpeakerEmbeddingExtractor",
    "PipeWireAECManager",
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
    "TextSegment",
    "BenchmarkLogger",
]
