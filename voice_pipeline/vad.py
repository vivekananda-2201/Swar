"""
Silero VAD Handler with Progressive Speech Streaming.
Adapted from Hugging Face speech-to-speech vad_handler and vad_iterator.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class VADChunk:
    """Represents a speech segment emitted by VAD."""

    audio: np.ndarray  # 16kHz float32 audio
    is_final: bool  # True if user stopped speaking, False if interim progressive update
    duration: float  # Audio duration in seconds
    turn_id: int  # Conversation turn counter


class SileroVAD:
    """
    Silero Voice Activity Detector with real-time progressive streaming.

    While the user speaks, it emits progressive audio slices every `progressive_interval` seconds.
    When the user stops speaking (silence > `min_silence_ms`), it emits the finalized audio segment.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_ms: int = 800,
        min_speech_ms: int = 250,
        speech_pad_ms: int = 30,
        progressive_interval: float = 0.5,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.speech_pad_ms = speech_pad_ms
        self.progressive_interval = progressive_interval
        self.device = device

        # Silero VAD requires exact 512 samples per step at 16kHz (32ms)
        self.window_size_samples = 512 if sample_rate == 16000 else 256
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)

        # Load Silero VAD model
        logger.info(f"Loading Silero VAD model on {device}...")
        self.model = self._load_model()
        self.model.eval()

        # State tracking
        self.turn_counter = 0
        self.triggered = False
        self.temp_end_samples = 0
        self.speech_buffer: list[np.ndarray] = []
        self.pre_speech_buffer: collections.deque[np.ndarray] = collections.deque()
        self.pre_speech_samples = 0
        self.leftover_samples = np.array([], dtype=np.float32)

        self.last_progressive_time: float = 0.0
        self.speech_started_emitted = False

    def _load_model(self) -> torch.nn.Module:
        local_cache_path = Path.home() / ".cache/torch/hub/snakers4_silero-vad_master"
        if local_cache_path.exists():
            try:
                model, _ = torch.hub.load(
                    str(local_cache_path),
                    "silero_vad",
                    source="local",
                    trust_repo=True,
                    verbose=False,
                )
                return model.to(self.device)
            except Exception as e:
                logger.debug(f"Loading from local hub cache failed: {e}")

        try:
            model, _ = torch.hub.load(
                "snakers4/silero-vad:master",
                "silero_vad",
                trust_repo=True,
                verbose=False,
            )
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {e}")

    def reset(self) -> None:
        """Reset internal state for a new session."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        self.triggered = False
        self.temp_end_samples = 0
        self.speech_buffer.clear()
        self.pre_speech_buffer.clear()
        self.pre_speech_samples = 0
        self.leftover_samples = np.array([], dtype=np.float32)
        self.last_progressive_time = 0.0
        self.speech_started_emitted = False

    def _trim_pre_speech(self) -> None:
        while self.pre_speech_buffer and self.pre_speech_samples > self.speech_pad_samples:
            first = self.pre_speech_buffer[0]
            excess = self.pre_speech_samples - self.speech_pad_samples
            if len(first) <= excess:
                self.pre_speech_buffer.popleft()
                self.pre_speech_samples -= len(first)
            else:
                self.pre_speech_buffer[0] = first[excess:]
                self.pre_speech_samples -= excess

    def process_chunk(self, audio_data: bytes | np.ndarray) -> Iterator[VADChunk]:
        """
        Feed an incoming audio chunk (raw 16-bit PCM bytes or float32 np.ndarray).
        Yields progressive VADChunks while speaking and final VADChunks on speech stop.
        """
        # Convert to float32 1D numpy array
        if isinstance(audio_data, bytes):
            int16_arr = np.frombuffer(audio_data, dtype=np.int16)
            audio = int16_arr.astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.int16:
                audio = audio_data.astype(np.float32) / 32768.0
            else:
                audio = audio_data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_data)}")

        if len(self.leftover_samples) > 0:
            audio = np.concatenate([self.leftover_samples, audio])
            self.leftover_samples = np.array([], dtype=np.float32)

        idx = 0
        total_len = len(audio)

        while idx + self.window_size_samples <= total_len:
            frame = audio[idx : idx + self.window_size_samples]
            idx += self.window_size_samples

            frame_tensor = torch.from_numpy(frame).to(self.device)
            with torch.no_grad():
                speech_prob = self.model(frame_tensor, self.sample_rate).item()

            current_time = time.time()

            # Speech detected
            if speech_prob >= self.threshold:
                if self.temp_end_samples != 0:
                    self.temp_end_samples = 0

                if not self.triggered:
                    self.triggered = True
                    self.turn_counter += 1
                    self.speech_started_emitted = True
                    self.last_progressive_time = current_time

                    # Prepend pre-speech pad
                    if self.pre_speech_buffer:
                        self.speech_buffer.extend(list(self.pre_speech_buffer))
                        self.pre_speech_buffer.clear()
                        self.pre_speech_samples = 0

                self.speech_buffer.append(frame)

                # Check if we should yield a progressive update while speaking
                current_speech_samples = sum(len(f) for f in self.speech_buffer)
                if (
                    current_speech_samples >= self.min_speech_samples
                    and (current_time - self.last_progressive_time) >= self.progressive_interval
                ):
                    accumulated = np.concatenate(self.speech_buffer)
                    duration = len(accumulated) / self.sample_rate
                    self.last_progressive_time = current_time
                    yield VADChunk(
                        audio=accumulated,
                        is_final=False,
                        duration=duration,
                        turn_id=self.turn_counter,
                    )

            # Silence detected
            else:
                if not self.triggered:
                    # Buffer pre-speech audio
                    self.pre_speech_buffer.append(frame)
                    self.pre_speech_samples += len(frame)
                    self._trim_pre_speech()
                else:
                    self.speech_buffer.append(frame)
                    self.temp_end_samples += self.window_size_samples

                    # Check if silence duration exceeded min_silence
                    if self.temp_end_samples >= self.min_silence_samples:
                        # Finalize speech segment
                        accumulated = np.concatenate(self.speech_buffer)
                        # Trim trailing silence
                        trim_samples = max(0, self.temp_end_samples - self.speech_pad_samples)
                        if trim_samples > 0 and len(accumulated) > trim_samples:
                            final_audio = accumulated[:-trim_samples]
                        else:
                            final_audio = accumulated

                        duration = len(final_audio) / self.sample_rate

                        if len(final_audio) >= self.min_speech_samples:
                            yield VADChunk(
                                audio=final_audio,
                                is_final=True,
                                duration=duration,
                                turn_id=self.turn_counter,
                            )

                        # Reset state for next turn
                        self.triggered = False
                        self.temp_end_samples = 0
                        self.speech_buffer.clear()
                        self.pre_speech_buffer.clear()
                        self.pre_speech_samples = 0
                        self.speech_started_emitted = False
                        self.last_progressive_time = 0.0
                        if hasattr(self.model, "reset_states"):
                            self.model.reset_states()

        # Save remaining samples for next chunk
        if idx < total_len:
            self.leftover_samples = audio[idx:]
