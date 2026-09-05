"""
Real-time Audio Input and Output Stream Manager using sounddevice.
Supports microphone capture, smooth speaker playback, and instant barge-in interruption.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Manages live microphone capture (16kHz mono) and speaker playback (24kHz mono).
    Includes cancellation / barge-in support to immediately cut off speech when user speaks.
    """

    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        input_chunk_size: int = 512,
        input_device: Optional[int | str] = None,
        output_device: Optional[int | str] = None,
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.input_chunk_size = input_chunk_size
        self.input_device = input_device
        self.output_device = output_device

        self.input_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.output_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()

        self.is_recording = False
        self.is_playing = False
        self.stop_event = threading.Event()
        self.interrupt_playback_event = threading.Event()

        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._playback_thread: Optional[threading.Thread] = None

    def start_recording(self, on_chunk: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """Start recording audio from microphone."""
        if self.is_recording:
            return

        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio input status: {status}")
            chunk = indata[:, 0].copy()  # extract mono channel
            if on_chunk:
                on_chunk(chunk)
            else:
                self.input_queue.put(chunk)

        try:
            self._input_stream = sd.InputStream(
                samplerate=self.input_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.input_chunk_size,
                device=self.input_device,
                callback=_audio_callback,
            )
            self._input_stream.start()
            self.is_recording = True
            logger.info("Microphone stream started.")
        except Exception as e:
            logger.error(f"Failed to start microphone stream: {e}")
            raise

    def stop_recording(self) -> None:
        """Stop microphone recording."""
        if not self.is_recording:
            return
        self.is_recording = False
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        logger.info("Microphone stream stopped.")

    def play_chunk(self, audio: np.ndarray) -> None:
        """Queue an audio chunk (float32 at output_sample_rate) for playback."""
        self.output_queue.put(audio)
        if not self.is_playing:
            self.start_playback_worker()

    def start_playback_worker(self) -> None:
        """Starts background worker thread to stream audio to speaker."""
        if self.is_playing:
            return

        self.is_playing = True
        self.interrupt_playback_event.clear()

        def _worker():
            try:
                with sd.OutputStream(
                    samplerate=self.output_sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.output_device,
                ) as stream:
                    while not self.stop_event.is_set():
                        try:
                            chunk = self.output_queue.get(timeout=0.1)
                        except queue.Empty:
                            if not self.is_playing:
                                break
                            continue

                        if chunk is None or self.interrupt_playback_event.is_set():
                            # Clear remaining queue on interrupt
                            while not self.output_queue.empty():
                                try:
                                    self.output_queue.get_nowait()
                                except queue.Empty:
                                    break
                            break

                        stream.write(chunk)

            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                self.is_playing = False

        self._playback_thread = threading.Thread(target=_worker, daemon=True)
        self._playback_thread.start()

    def stop_playback(self) -> None:
        """Immediately interrupts playback (barge-in)."""
        self.interrupt_playback_event.set()
        self.is_playing = False
        # Drain output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("Playback interrupted and cleared.")

    def close(self) -> None:
        """Clean shutdown of all streams."""
        self.stop_event.set()
        self.stop_recording()
        self.stop_playback()
