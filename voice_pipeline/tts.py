"""
Decoupled TTS Generation and Playback Pipeline with Kokoro TTS on CPU.
Directly implements Change 2, Change 3, and Change 4 from tts-changes-prompt.txt.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional

import numpy as np
import sounddevice as sd

from voice_pipeline.speak_out_parser import (
    ChunkingConfig,
    IncrementalSpeakOutParser,
    RobustSentenceChunker,
    TextSegment,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioItem:
    """A synthesized audio chunk waiting in the ready audio queue."""
    chunk_id: int
    text: str
    audio: np.ndarray  # float32 24kHz audio
    duration_s: float  # Audio duration in seconds
    turn_id: int = 0


class KokoroTTS:
    """
    Kokoro-82M Text-to-Speech synthesis engine running natively on CPU.
    """

    def __init__(
        self,
        voice: str = "af_bella",
        lang_code: str = "a",
        speed: float = 1.0,
        device: str = "cpu",
    ):
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.device = device
        self.sample_rate = 24000

        logger.info(f"Loading Kokoro TTS engine (voice={voice}, lang={lang_code}) on {device}...")
        import kokoro

        self.pipeline = kokoro.KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=device,
        )
        logger.info("Kokoro TTS engine loaded.")

    def synthesize_chunk(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[np.ndarray]:
        """Synthesizes a single chunk of text into a float32 numpy array."""
        if not text or not text.strip():
            return None

        v = voice or self.voice
        s = speed or self.speed
        audio_parts: List[np.ndarray] = []

        try:
            for gs, ps, audio in self.pipeline(text.strip(), voice=v, speed=s):
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("Kokoro synthesis cancelled.")
                    return None

                if audio is None:
                    continue

                if hasattr(audio, "cpu"):
                    audio_np = audio.cpu().numpy()
                elif isinstance(audio, np.ndarray):
                    audio_np = audio
                else:
                    audio_np = np.array(audio, dtype=np.float32)

                audio_parts.append(audio_np.astype(np.float32))

            if audio_parts:
                return np.concatenate(audio_parts)
            return None
        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")
            return None


class TTSPipeline:
    """
    Decoupled Producer-Consumer TTS Pipeline.
    
    Architecture:
      <speak-out> Text Chunks
             │
             ▼
      Text Queue
             │
             ▼
      TTS Generation Worker (Independent Thread - runs ahead as fast as CPU permits)
             │
             ▼
      Ready Audio Queue (Buffered audio ready for playback)
             │
             ▼
      Audio Playback Worker (Independent Thread - writes sequentially to speaker)
             │
             ▼
      Speaker Hardware
    """

    def __init__(
        self,
        tts_engine: Optional[KokoroTTS] = None,
        voice: str = "af_bella",
        chunking_config: Optional[ChunkingConfig] = None,
        output_device: Optional[int | str] = None,
        sample_rate: int = 24000,
        max_text_queue_size: int = 100,
        max_audio_queue_size: int = 50,
    ):
        self.tts = tts_engine or KokoroTTS(voice=voice, device="cpu")
        self.chunking_config = chunking_config or ChunkingConfig()
        self.output_device = output_device
        self.sample_rate = sample_rate

        # Sentence Chunker
        self.sentence_chunker = RobustSentenceChunker(self.chunking_config)

        # Queues
        self.text_queue: queue.Queue[Optional[tuple[int, int, str]]] = queue.Queue(maxsize=max_text_queue_size)
        self.ready_audio_queue: queue.Queue[Optional[AudioItem]] = queue.Queue(maxsize=max_audio_queue_size)

        # Threading & Control
        self.is_running = False
        self.stop_event = threading.Event()
        self.cancel_current_turn_event = threading.Event()

        self._generation_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None

        self._chunk_counter = 0
        self._current_turn_id = 0
        self._active_stream: Optional[sd.OutputStream] = None
        self._active_playing_item: Optional[AudioItem] = None
        self._lock = threading.RLock()

    @property
    def buffered_audio_seconds(self) -> float:
        """
        Measures the actual audio duration currently queued ahead in ready_audio_queue.
        Used for adaptive chunking decisions.
        """
        with self._lock:
            # Inspect items in queue without removing
            total_duration = 0.0
            for item in list(self.ready_audio_queue.queue):
                if item is not None and isinstance(item, AudioItem):
                    total_duration += item.duration_s
            return total_duration

    @property
    def is_speaking(self) -> bool:
        """True if audio is currently playing or ready in queue."""
        return self._active_playing_item is not None or not self.ready_audio_queue.empty()

    def start(self) -> None:
        """Starts both generation and playback background workers."""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.cancel_current_turn_event.clear()

        # Start independent workers
        self._generation_thread = threading.Thread(target=self._generation_worker, daemon=True, name="TTS-Generation")
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True, name="TTS-Playback")

        self._generation_thread.start()
        self._playback_thread.start()
        logger.info("Independent TTS Generation and Playback workers started.")

    def stop(self) -> None:
        """Stops all workers and clears queues."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()
        self.interrupt()

        self.text_queue.put(None)
        self.ready_audio_queue.put(None)

        if self._generation_thread and self._generation_thread.is_alive():
            self._generation_thread.join(timeout=1.0)
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        logger.info("TTS Pipeline stopped.")

    def interrupt(self) -> None:
        """
        Immediately interrupts any playing audio and ongoing synthesis (Barge-in).
        Flushes all pending text and ready audio queues instantly.
        """
        with self._lock:
            self._current_turn_id += 1
            self.cancel_current_turn_event.set()

            # Drain text queue
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except queue.Empty:
                    break

            # Drain ready audio queue
            while not self.ready_audio_queue.empty():
                try:
                    self.ready_audio_queue.get_nowait()
                except queue.Empty:
                    break

            self._active_playing_item = None
            self.sentence_chunker.reset()
        logger.debug(f"TTS Pipeline interrupted. Advanced to turn epoch #{self._current_turn_id}.")

    def feed_text(self, text: str, turn_id: Optional[int] = None) -> None:
        """
        Feeds any arbitrary text directly to the TTS pipeline.
        Intelligently splits using progressive/adaptive chunking and enqueues for generation.
        """
        if not self.is_running or not text or not text.strip():
            return

        with self._lock:
            # If an explicit turn_id is supplied and is older than the current epoch, drop it
            if turn_id is not None and turn_id < self._current_turn_id:
                logger.debug(f"TTS dropping stale chunk from turn {turn_id} (active is {self._current_turn_id})")
                return

            # If no turn_id specified, use the current epoch
            active_turn = turn_id if turn_id is not None else self._current_turn_id
            self._current_turn_id = max(self._current_turn_id, active_turn)

            # Active new turn text clears cancellation
            self.cancel_current_turn_event.clear()
            current_buffered_s = self.buffered_audio_seconds

            for chunk in self.sentence_chunker.feed_text(
                text.strip(),
                is_block_end=True,
                current_buffered_audio_s=current_buffered_s,
            ):
                if chunk.strip():
                    self._chunk_counter += 1
                    self.text_queue.put((active_turn, self._chunk_counter, chunk.strip()))

    def feed_stream(self, token_or_text_stream: Iterator[str], turn_id: Optional[int] = None) -> None:
        """
        Feeds any token/text stream directly to the TTS pipeline.
        Synthesizes chunks as soon as sentence boundaries are reached.
        """
        if not self.is_running:
            return

        active_turn = turn_id if turn_id is not None else self._current_turn_id
        with self._lock:
            if turn_id is not None and turn_id < self._current_turn_id:
                logger.debug(f"TTS stream dropping stale turn {turn_id} (active is {self._current_turn_id})")
                return
            self._current_turn_id = max(self._current_turn_id, active_turn)
            self.cancel_current_turn_event.clear()

        for token in token_or_text_stream:
            with self._lock:
                if active_turn < self._current_turn_id or self.cancel_current_turn_event.is_set():
                    logger.debug("TTS stream feed aborted due to interruption.")
                    break
            current_buffered_s = self.buffered_audio_seconds
            for chunk in self.sentence_chunker.feed_text(
                token,
                is_block_end=False,
                current_buffered_audio_s=current_buffered_s,
            ):
                if chunk.strip():
                    self._chunk_counter += 1
                    self.text_queue.put((active_turn, self._chunk_counter, chunk.strip()))

        # Flush any trailing text at the end of the stream if not cancelled
        with self._lock:
            is_cancelled = active_turn < self._current_turn_id or self.cancel_current_turn_event.is_set()

        if not is_cancelled:
            for chunk in self.sentence_chunker.feed_text(
                "",
                is_block_end=True,
                current_buffered_audio_s=self.buffered_audio_seconds,
            ):
                if chunk.strip():
                    self._chunk_counter += 1
                    self.text_queue.put((active_turn, self._chunk_counter, chunk.strip()))

    def _generation_worker(self) -> None:
        """
        Independent TTS Generation Process:
        Continuously consumes available text chunks and synthesizes audio as quickly
        as CPU hardware allows, without waiting for playback.
        """
        while not self.stop_event.is_set():
            try:
                item = self.text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            turn_id, chunk_id, text = item

            with self._lock:
                if turn_id < self._current_turn_id or self.cancel_current_turn_event.is_set():
                    continue

            logger.debug(f"TTS Generating chunk #{chunk_id} (turn #{turn_id}): {text[:50]}...")
            start_t = time.perf_counter()

            audio = self.tts.synthesize_chunk(
                text=text,
                cancel_event=self.cancel_current_turn_event,
            )

            gen_duration = time.perf_counter() - start_t

            with self._lock:
                is_stale = turn_id < self._current_turn_id or self.cancel_current_turn_event.is_set()

            if audio is not None and not is_stale:
                duration_s = len(audio) / self.sample_rate
                audio_item = AudioItem(
                    chunk_id=chunk_id,
                    text=text,
                    audio=audio,
                    duration_s=duration_s,
                    turn_id=turn_id,
                )
                logger.debug(
                    f"TTS Chunk #{chunk_id} READY: audio={duration_s:.2f}s, gen_time={gen_duration:.2f}s "
                    f"(ahead buffer: {self.buffered_audio_seconds:.2f}s)"
                )
                self.ready_audio_queue.put(audio_item)

    def _playback_worker(self) -> None:
        """
        Independent Audio Playback Process:
        Consumes ready audio chunks sequentially and streams them to the speaker.
        """
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.output_device,
            ) as stream:
                self._active_stream = stream
                while not self.stop_event.is_set():
                    try:
                        item = self.ready_audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if item is None:
                        break

                    with self._lock:
                        is_stale = item.turn_id < self._current_turn_id or self.cancel_current_turn_event.is_set()

                    if is_stale:
                        continue

                    self._active_playing_item = item
                    logger.debug(f"Audio Playback SPEAKING chunk #{item.chunk_id}: {item.text[:50]}...")

                    # Write in 512-sample blocks to allow immediate barge-in interruption (<21ms)
                    chunk_size = 512
                    audio_data = item.audio
                    for i in range(0, len(audio_data), chunk_size):
                        with self._lock:
                            interrupted = item.turn_id < self._current_turn_id or self.cancel_current_turn_event.is_set()
                        if interrupted or self.stop_event.is_set():
                            logger.debug("Playback interrupted mid-chunk.")
                            break
                        slice_chunk = audio_data[i : i + chunk_size]
                        stream.write(slice_chunk)

                    self._active_playing_item = None

        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._active_stream = None
            self._active_playing_item = None
