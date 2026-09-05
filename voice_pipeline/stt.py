"""
Parakeet TDT Speech-to-Text Handler with Smart Progressive Streaming.
Adapted from Hugging Face speech-to-speech STT implementation (nano-parakeet + smart_progressive_streaming).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PartialTranscription:
    """Represents an interim progressive transcription update."""

    fixed_text: str  # Stabilized sentences from earlier in the utterance
    active_text: str  # Current actively updated words
    timestamp: float  # Audio position in seconds
    is_final: bool  # True if final transcription for the turn

    @property
    def text(self) -> str:
        """Returns the full combined text."""
        if self.fixed_text and self.active_text:
            return f"{self.fixed_text} {self.active_text}".strip()
        return (self.fixed_text or self.active_text).strip()


class SmartProgressiveStreaming:
    """
    Smart progressive streaming with sentence-aware window management.
    Prevents runaway CPU latency by keeping finalized sentences fixed
    and only re-transcribing the active tail when speech exceeds 15 seconds.
    """

    def __init__(
        self,
        model: Any,
        sample_rate: int = 16000,
        emission_interval: float = 0.5,
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.emission_interval = emission_interval
        self.max_window_size = max_window_size
        self.sentence_buffer = sentence_buffer

        self.reset()

    def reset(self) -> None:
        """Reset state for new speech turn."""
        self.fixed_sentences: list[str] = []
        self.fixed_end_time: float = 0.0
        self.last_transcribed_length: int = 0

    def _decode_window(self, audio_window: np.ndarray) -> SimpleNamespace:
        """Decodes an audio window using Parakeet model with timestamps."""
        try:
            result = self.model.transcribe(audio_window, timestamps=True)
            text = getattr(result, "text", "")
            timestamp_dict = getattr(result, "timestamp", {}) or {}
            segments = timestamp_dict.get("segment", []) if isinstance(timestamp_dict, dict) else []
            sentences = [
                SimpleNamespace(
                    text=seg.get("segment", "").strip(),
                    end=seg.get("end", 0.0),
                )
                for seg in segments
                if isinstance(seg, dict)
            ]
            return SimpleNamespace(text=text, sentences=sentences)
        except Exception as e:
            logger.debug(f"Progressive window decode exception (fallback): {e}")
            text = self.model.transcribe(audio_window, timestamps=False)
            return SimpleNamespace(text=str(text), sentences=[])

    def transcribe_incremental(self, audio: np.ndarray) -> PartialTranscription:
        """
        Transcribe audio incrementally as it grows while user speaks.
        Returns a PartialTranscription with fixed_text and active_text.
        """
        current_length = len(audio)

        # Need at least emission_interval of audio
        if current_length < int(self.sample_rate * self.emission_interval):
            return PartialTranscription(
                fixed_text=" ".join(self.fixed_sentences),
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=False,
            )

        if current_length == self.last_transcribed_length:
            return PartialTranscription(
                fixed_text=" ".join(self.fixed_sentences),
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=False,
            )

        self.last_transcribed_length = current_length

        # Extract window starting from last fixed sentence
        window_start_samples = int(self.fixed_end_time * self.sample_rate)
        audio_window = audio[window_start_samples:]

        # Decode active window
        result = self._decode_window(audio_window)

        # Check if window exceeds max_window_size (slide window if too long)
        window_duration = len(audio_window) / self.sample_rate
        if window_duration >= self.max_window_size and len(result.sentences) > 1:
            cutoff_time = window_duration - self.sentence_buffer
            new_fixed_sentences = []
            new_fixed_end_time = self.fixed_end_time

            for sentence in result.sentences:
                sentence_abs_time = self.fixed_end_time + sentence.end
                if sentence.end < cutoff_time:
                    new_fixed_sentences.append(sentence.text.strip())
                    new_fixed_end_time = sentence_abs_time
                else:
                    break

            if new_fixed_sentences:
                self.fixed_sentences.extend(new_fixed_sentences)
                self.fixed_end_time = new_fixed_end_time

                # Re-transcribe active remainder
                window_start_samples = int(self.fixed_end_time * self.sample_rate)
                audio_window = audio[window_start_samples:]
                result = self._decode_window(audio_window)

        fixed_text = " ".join(self.fixed_sentences)
        active_text = result.text.strip()
        timestamp = current_length / self.sample_rate

        return PartialTranscription(
            fixed_text=fixed_text,
            active_text=active_text,
            timestamp=timestamp,
            is_final=False,
        )


class ParakeetSTT:
    """
    NVIDIA Parakeet TDT Speech-to-Text running on CPU via nano-parakeet.
    Provides instant real-time partial streaming while user speaks
    and high-accuracy final transcription when speech concludes.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cpu",
        language: str = "en",
        enable_progressive: bool = True,
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.enable_progressive = enable_progressive
        self.sample_rate = sample_rate
        self.lock = threading.Lock()

        logger.info(f"Loading Parakeet TDT model '{model_name}' on {device}...")
        import nano_parakeet

        self.model = nano_parakeet.from_pretrained(model_name=model_name, device=device)
        logger.info("Parakeet TDT model loaded successfully.")

        self.streaming_handler = SmartProgressiveStreaming(
            model=self.model,
            sample_rate=sample_rate,
            emission_interval=0.5,
            max_window_size=15.0,
            sentence_buffer=2.0,
        )

        self._warmup()

    def _warmup(self) -> None:
        """Warm up model with dummy silent frame."""
        logger.info("Warming up Parakeet STT...")
        dummy = np.zeros(self.sample_rate, dtype=np.float32)
        try:
            with self.lock:
                _ = self.model.transcribe(dummy, timestamps=False)
            logger.info("Parakeet STT warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")

    def reset_turn(self) -> None:
        """Reset progressive streaming state for a new turn."""
        self.streaming_handler.reset()

    def transcribe_progressive(self, audio: np.ndarray) -> PartialTranscription:
        """
        Transcribe progressive audio buffer while user is actively speaking.
        """
        with self.lock:
            return self.streaming_handler.transcribe_incremental(audio)

    def transcribe_final(self, audio: np.ndarray) -> str:
        """
        Transcribe full finalized audio buffer when user stops speaking.
        """
        with self.lock:
            # Transcribe full final audio for maximum accuracy
            result = self.model.transcribe(audio, timestamps=False)
            text = str(result).strip()
            self.reset_turn()
            return text
