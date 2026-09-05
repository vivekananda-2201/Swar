"""
Unified Voice Pipeline using Official Hugging Face Speech-to-Speech Architecture.

Uses the exact VAD and STT pipeline from huggingface/speech-to-speech:
- Silero VAD (with 500ms speech_pad_ms, min_speech_continuation_ms hysteresis, 800ms min_silence)
- NVIDIA Parakeet TDT 0.6B on CPU (with Smart Progressive Streaming)
- TranscriptionNotifier (official event dispatch)
- Non-blocking 16-bit PCM RawInputStream (no audio dropped)
- Decoupled Kokoro TTS Pipeline with <speak-out> tags & progressive chunking
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import warnings
from typing import Callable, Iterator, Optional

# Suppress harmless PyTorch/Kokoro and upstream deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("speech_to_speech.VAD.vad_handler").setLevel(logging.ERROR)

import sounddevice as sd
from rich.console import Console

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.VAD.vad_handler import VADHandler

from voice_pipeline.config import PipelineConfig
from voice_pipeline.speak_out_parser import ChunkingConfig, TextSegment
from voice_pipeline.tts import KokoroTTS, TTSPipeline
from voice_pipeline.wake_word import WakeState, WakeTriggerConfig, WakeWordConfig, WakeWordEngine

logger = logging.getLogger(__name__)
console = Console()
_turn_local = threading.local()


class VoicePipeline:
    """
    Unified Voice Pipeline with official Hugging Face VAD & Parakeet STT
    and decoupled Kokoro TTS with progressive chunking.
    Fully configurable via config.yaml or PipelineConfig.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        # Overrides (if specified, these override config values):
        vad_threshold: Optional[float] = None,
        min_silence_ms: Optional[int] = None,
        min_speech_ms: Optional[int] = None,
        min_speech_continuation_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
        progressive_interval: Optional[float] = None,
        stt_model_name: Optional[str] = None,
        stt_language: Optional[str] = None,
        enable_live_transcription: Optional[bool] = None,
        tts_voice: Optional[str] = None,
        tts_speed: Optional[float] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        llm_client: Optional[LLMClient] = None,
        device: Optional[str] = None,
        input_device: Optional[int | str] = None,
        output_device: Optional[int | str] = None,
        verbose: Optional[bool] = None,
        allow_barge_in: Optional[bool] = None,
        wake_mode: Optional[bool] = None,
        wake_words: Optional[Union[List[str], Dict[str, float], List[WakeTriggerConfig]]] = None,
        wake_timeout: Optional[float] = None,
        # Callbacks
        on_speech_started: Optional[Callable[[], None]] = None,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
        on_final_transcript: Optional[Callable[[str], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
        on_wake: Optional[Callable[[str, str, float], None]] = None,
        on_sleep: Optional[Callable[[], None]] = None,
        on_chunk_synthesized: Optional[Callable[[int, int, str, float, float], None]] = None,
        on_playback_start: Optional[Callable[[int, int], None]] = None,
        on_playback_complete: Optional[Callable[[int, int, bool], None]] = None,
    ):
        # 0. Load base config from file (or defaults) and apply any explicit overrides
        self.config = config or PipelineConfig.load_default_or_file()

        if device is not None:
            self.config.general.device = device
        if verbose is not None:
            self.config.general.verbose = verbose
        if allow_barge_in is not None:
            self.config.general.allow_barge_in = allow_barge_in
        if input_device is not None:
            self.config.general.input_device = input_device
        if output_device is not None:
            self.config.general.output_device = output_device

        if vad_threshold is not None:
            self.config.vad.threshold = vad_threshold
        if min_silence_ms is not None:
            self.config.vad.min_silence_ms = min_silence_ms
        if min_speech_ms is not None:
            self.config.vad.min_speech_ms = min_speech_ms
        if min_speech_continuation_ms is not None:
            self.config.vad.min_speech_continuation_ms = min_speech_continuation_ms
        if speech_pad_ms is not None:
            self.config.vad.speech_pad_ms = speech_pad_ms
        if progressive_interval is not None:
            self.config.vad.progressive_interval = progressive_interval

        if stt_model_name is not None:
            self.config.stt.model_name = stt_model_name
        if stt_language is not None:
            self.config.stt.language = stt_language
        if enable_live_transcription is not None:
            self.config.stt.enable_live_transcription = enable_live_transcription

        if tts_voice is not None:
            self.config.tts.voice = tts_voice
        if tts_speed is not None:
            self.config.tts.speed = tts_speed
        if chunking_config is not None:
            self.config.chunking = chunking_config

        self.device = self.config.general.device
        self.verbose = self.config.general.verbose
        self.allow_barge_in = self.config.general.allow_barge_in
        self.input_device = self.config.general.input_device
        self.output_device = self.config.general.output_device
        self.enable_live_transcription = self.config.stt.enable_live_transcription

        # Callbacks
        self.on_speech_started = on_speech_started
        self.on_partial_transcript = on_partial_transcript
        self.on_final_transcript = on_final_transcript
        self.on_interrupted = on_interrupted
        self.on_wake = on_wake
        self.on_sleep = on_sleep

        # Wake Word & Wake Sentence Configuration
        if wake_mode is not None:
            self.config.wake_word.enabled = wake_mode
        elif self.config.general.wake_mode:
            self.config.wake_word.enabled = True

        if wake_timeout is not None:
            self.config.wake_word.default_timeout = wake_timeout

        if wake_words is not None:
            triggers = []
            if isinstance(wake_words, dict):
                for p, t in wake_words.items():
                    triggers.append(WakeTriggerConfig(phrase=str(p), timeout=float(t)))
            elif isinstance(wake_words, list):
                for item in wake_words:
                    if isinstance(item, WakeTriggerConfig):
                        triggers.append(item)
                    elif isinstance(item, str):
                        triggers.append(WakeTriggerConfig(phrase=item, timeout=self.config.wake_word.default_timeout))
            self.config.wake_word.triggers = triggers

        self.wake_mode = self.config.wake_word.enabled

        self.wake_engine = WakeWordEngine(
            config=self.config.wake_word,
            is_busy_callback=lambda: getattr(self, "tts_pipeline", None) is not None and self.tts_pipeline.is_speaking or self._is_generating_response,
            on_wake=self._on_engine_wake,
            on_sleep=self._on_engine_sleep,
            on_log=self._log,
        )

        # Control primitives
        self.stop_event = threading.Event()
        self.should_listen = threading.Event()
        self.should_listen.set()

        # Inter-handler queues
        self.audio_in_queue: queue.Queue = queue.Queue(maxsize=256)
        self.spoken_prompt_queue: queue.Queue = queue.Queue()
        self.stt_out_queue: queue.Queue = queue.Queue()
        self.text_prompt_queue: queue.Queue = queue.Queue()
        self.text_output_queue: queue.Queue = queue.Queue()

        self.speculative_turns = SpeculativeTurnTracker()
        self.cancel_scope = CancelScope()

        # 1. Initialize Official Hugging Face VAD Handler
        self._log(f"[cyan]Initializing Official Silero VAD (speech_pad={self.config.vad.speech_pad_ms}ms, min_silence={self.config.vad.min_silence_ms}ms)...[/cyan]")
        self.vad_handler = VADHandler(
            self.stop_event,
            queue_in=self.audio_in_queue,
            queue_out=self.spoken_prompt_queue,
            setup_args=(self.should_listen,),
            setup_kwargs={
                "speculative_turns": self.speculative_turns,
                "thresh": self.config.vad.threshold,
                "sample_rate": 16000,
                "min_silence_ms": self.config.vad.min_silence_ms,
                "min_speech_ms": self.config.vad.min_speech_ms,
                "min_speech_continuation_ms": self.config.vad.min_speech_continuation_ms,
                "speech_pad_ms": self.config.vad.speech_pad_ms,
                "enable_realtime_transcription": self.config.stt.enable_live_transcription,
                "realtime_processing_pause": self.config.vad.progressive_interval,
                "text_output_queue": self.text_output_queue,
                "smart_turn": False,
                "speculative_reopen_ms": 0,
                "unanswered_reopen_ms": 0,
            },
        )

        # 2. Initialize Official Hugging Face Parakeet TDT Handler
        self._log(f"[cyan]Initializing Official Parakeet TDT STT ({self.config.stt.model_name} on {self.device})...[/cyan]")
        self.stt_handler = ParakeetTDTSTTHandler(
            self.stop_event,
            queue_in=self.spoken_prompt_queue,
            queue_out=self.stt_out_queue,
            setup_kwargs={
                "model_name": self.config.stt.model_name,
                "device": self.device,
                "language": self.config.stt.language,
                "enable_live_transcription": self.config.stt.enable_live_transcription,
                "live_transcription_update_interval": self.config.vad.progressive_interval,
            },
        )

        # 3. Initialize Official Hugging Face Transcription Notifier
        self.notifier = TranscriptionNotifier(
            self.stop_event,
            queue_in=self.stt_out_queue,
            queue_out=self.text_prompt_queue,
            setup_kwargs={
                "text_output_queue": self.text_output_queue,
                "should_listen": self.should_listen,
            },
        )

        # 4. Initialize Decoupled Kokoro TTS Pipeline
        self._log(f"[cyan]Initializing Decoupled Kokoro TTS (voice={self.config.tts.voice}) on {self.device}...[/cyan]")
        self.chunking_config = self.config.chunking
        self.tts_engine = KokoroTTS(
            voice=self.config.tts.voice,
            lang_code=self.config.tts.lang_code,
            speed=self.config.tts.speed,
            device=self.device,
        )
        self.tts_pipeline = TTSPipeline(
            tts_engine=self.tts_engine,
            voice=self.config.tts.voice,
            chunking_config=self.chunking_config,
            output_device=self.output_device,
            sample_rate=self.config.tts.sample_rate,
            max_text_queue_size=self.config.tts.max_text_queue_size,
            max_audio_queue_size=self.config.tts.max_audio_queue_size,
        )
        self.tts_pipeline.on_chunk_synthesized = on_chunk_synthesized
        self.tts_pipeline.on_playback_start = on_playback_start
        self.tts_pipeline.on_playback_complete = on_playback_complete

        # Worker threads
        self.threads: list[threading.Thread] = []
        self.mic_stream: Optional[sd.RawInputStream] = None
        self.is_running = False
        self._is_generating_response = False
        self._active_turn_id = 0
        self._active_turn_interrupted = threading.Event()

    def _on_engine_wake(self, phrase: str, remainder: str, timeout: float) -> None:
        if self.on_wake:
            self.on_wake(phrase, remainder, timeout)

    def _on_engine_sleep(self) -> None:
        if self.on_sleep:
            self.on_sleep()

    @property
    def is_awake(self) -> bool:
        """True if the pipeline is awake and actively accepting speech."""
        return self.wake_engine.is_awake

    @property
    def wake_state(self) -> WakeState:
        """Returns WakeState.STANDBY or WakeState.ACTIVE."""
        return self.wake_engine.state

    @property
    def wake_time_remaining(self) -> float:
        """Remaining seconds before returning to wake word standby."""
        return self.wake_engine.time_remaining

    def wake(self, phrase: str = "manual", timeout: Optional[float] = None) -> None:
        """Manually wake the pipeline into active conversation mode."""
        self.wake_engine.wake(phrase=phrase, timeout=timeout)

    def sleep(self) -> None:
        """Manually put the pipeline to sleep into wake word standby."""
        self.wake_engine.sleep()

    @property
    def is_interrupted(self) -> bool:
        """True if the current conversational turn was interrupted by barge-in or superseded."""
        worker_turn = getattr(_turn_local, "turn_id", None)
        if worker_turn is not None and worker_turn != self._active_turn_id:
            return True
        return self._active_turn_interrupted.is_set()

    @property
    def active_turn_id(self) -> int:
        """Current turn epoch identifier."""
        return self._active_turn_id

    def _log(self, msg: str) -> None:
        """Outputs to terminal if verbose is True, otherwise logs via standard logging."""
        if getattr(self, "verbose", False):
            console.print(msg)
        else:
            logger.info(msg)

    def start(self) -> None:
        """Start microphone capture, official HF STT threads, and decoupled TTS."""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.should_listen.set()

        # Start decoupled TTS
        self.tts_pipeline.start()

        # Start wake engine background inactivity timer if wake_mode is enabled
        if self.wake_mode:
            self.wake_engine.start()
            trigger_info = [f"'{t.phrase}' ({t.timeout}s)" for t in self.config.wake_word.triggers]
            self._log(f"[cyan]Wake Mode active: waiting for wake phrase {trigger_info}...[/cyan]")

        # Start official HF handler threads
        self.threads = [
            threading.Thread(target=self.vad_handler.run, daemon=True, name="HF-VAD"),
            threading.Thread(target=self.stt_handler.run, daemon=True, name="HF-STT"),
            threading.Thread(target=self.notifier.run, daemon=True, name="HF-Notifier"),
            threading.Thread(target=self._event_dispatcher_loop, daemon=True, name="Event-Dispatcher"),
        ]
        for t in self.threads:
            t.start()

        # Start raw 16-bit PCM microphone capture (exact 512 samples = 1024 bytes per block)
        def _mic_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Mic status: {status}")
            if not self.should_listen.is_set():
                return
            # If barge-in is disabled, ignore mic while assistant is speaking
            # This protects against noisy room interruptions and speaker echo
            if not self.allow_barge_in and self.tts_pipeline.is_speaking:
                return
            try:
                self.audio_in_queue.put_nowait(bytes(indata))
            except queue.Full:
                logger.debug("Audio in queue full, dropping frame")

        self.mic_stream = sd.RawInputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=512,
            device=self.input_device,
            callback=_mic_callback,
        )
        self.mic_stream.start()

        self._log("[bold green]Voice Pipeline is active! Speak into your microphone...[/bold green]")

    def stop(self) -> None:
        """Stop all streams and workers cleanly."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None

        self.tts_pipeline.stop()
        self.wake_engine.stop()

        # Unblock queues
        self.audio_in_queue.put(b"")
        self.spoken_prompt_queue.put(None)
        self.stt_out_queue.put(None)
        self.text_output_queue.put(None)

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=0.5)

        self._log("[yellow]Voice Pipeline stopped.[/yellow]")

    def _handle_interruption(self) -> None:
        """Cancels ongoing TTS playback and generation on speech detection or Parakeet transcription."""
        was_interrupted = self._active_turn_interrupted.is_set()
        self._active_turn_interrupted.set()
        self._active_turn_id += 1
        self.tts_pipeline.interrupt()
        self._is_generating_response = False

        # Clear previous turn speculative prefixes only, preserving current interruption audio in VAD buffer
        if hasattr(self, "vad_handler") and self.vad_handler is not None:
            self.vad_handler._speculative_audio_prefix = None
            self.vad_handler._speculative_raw_audio_prefix = None
            self.vad_handler._pending_short_segment = None
            self.vad_handler._pending_reopen_candidate = None

        if not was_interrupted:
            if self.on_interrupted:
                self.on_interrupted()
            if self.verbose:
                console.print("\n[red][Barge-in: speech interrupted][/red]")

    def _start_user_turn(self, text: str) -> None:
        """Starts a new conversational turn and dispatches callback asynchronously."""
        self._active_turn_id += 1
        self._active_turn_interrupted.clear()
        turn_id = self._active_turn_id

        # Ensure fresh turn in VAD and STT - clear any speculative prefix
        if hasattr(self, "vad_handler") and self.vad_handler is not None:
            self.vad_handler._speculative_audio_prefix = None
            self.vad_handler._speculative_raw_audio_prefix = None
            if hasattr(self.vad_handler, "_start_new_turn"):
                self.vad_handler._start_new_turn()
        if hasattr(self, "stt_handler") and self.stt_handler is not None:
            self.stt_handler.processing_final = False
            if hasattr(self.stt_handler, "_reset_live_transcription_state"):
                self.stt_handler._reset_live_transcription_state(clear_turn=True)

        if self.verbose:
            console.print(f"[bold yellow]User: {text}[/bold yellow]")

        if self.on_final_transcript:
            threading.Thread(
                target=self._run_user_turn_callback,
                args=(text, turn_id),
                daemon=True,
                name=f"TurnWorker-{turn_id}",
            ).start()

    def _run_user_turn_callback(self, text: str, turn_id: int) -> None:
        _turn_local.turn_id = turn_id
        try:
            self._is_generating_response = True
            self.on_final_transcript(text)
        except Exception as e:
            logger.error(f"Error in on_final_transcript callback: {e}")
        finally:
            self._is_generating_response = False

    def _event_dispatcher_loop(self) -> None:
        """Processes events emitted by HF's VAD and TranscriptionNotifier."""
        while not self.stop_event.is_set():
            try:
                event = self.text_output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if event is None:
                break

            # 1. Speech Started Event (VAD detected speech)
            if isinstance(event, SpeechStartedEvent):
                if self.allow_barge_in and (self.tts_pipeline.is_speaking or self._is_generating_response):
                    self._handle_interruption()
                self.wake_engine.activity()
                if self.on_speech_started:
                    self.on_speech_started()

            # 2. Interim / Progressive Transcription Event (Parakeet started transcription while user speaks)
            elif isinstance(event, PartialTranscriptionEvent):
                if self.allow_barge_in and (self.tts_pipeline.is_speaking or self._is_generating_response):
                    self._handle_interruption()
                delta = getattr(event, "delta", "").strip()
                if delta:
                    # In wake_mode standby, suppress realtime noise spam to terminal
                    if self.wake_mode and not self.wake_engine.is_awake:
                        if self.verbose:
                            sys.stdout.write(f"\r\033[K[Standby listening]: {delta}")
                            sys.stdout.flush()
                        continue

                    if self.on_partial_transcript:
                        self.on_partial_transcript(delta)
                    elif self.verbose:
                        sys.stdout.write(f"\r\033[K[Realtime]: {delta}")
                        sys.stdout.flush()

            # 3. Final Transcription Completed Event (turn finished)
            elif isinstance(event, TranscriptionCompletedEvent):
                if self.allow_barge_in and (self.tts_pipeline.is_speaking or self._is_generating_response):
                    self._handle_interruption()

                # Commit turn in speculative tracker and clear prefix
                turn_id = getattr(event, "turn_id", None)
                turn_revision = getattr(event, "turn_revision", None)
                if turn_id is not None and turn_revision is not None:
                    self.speculative_turns.commit(turn_id, turn_revision)
                if hasattr(self, "vad_handler") and self.vad_handler is not None:
                    self.vad_handler._speculative_audio_prefix = None
                    self.vad_handler._speculative_raw_audio_prefix = None

                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    if self.verbose:
                        sys.stdout.write("\r\033[K")

                    if self.wake_mode:
                        should_forward, text_to_forward = self.wake_engine.process_transcript(transcript)
                        if should_forward and text_to_forward:
                            self._start_user_turn(text_to_forward)
                        elif not should_forward and self.wake_engine.is_awake:
                            # Woke up, but user only said the wake word without follow-up command
                            if self.verbose:
                                console.print(f"[bold green]⚡ Wake phrase detected ('{transcript}')! Listening for command...[/bold green]")
                        else:
                            # Non-wake speech in standby was discarded
                            logger.debug(f"[WakeMode] Standby ignored non-wake speech: {transcript}")
                    else:
                        self._start_user_turn(transcript)

    def speak_text(self, text: str) -> None:
        """
        Directly sends any plain text to the decoupled TTS pipeline.
        Drops text if current turn was interrupted or superseded.
        """
        worker_turn = getattr(_turn_local, "turn_id", None)
        if worker_turn is not None and worker_turn != self._active_turn_id:
            logger.debug(f"speak_text dropped: worker turn #{worker_turn} superseded by #{self._active_turn_id}")
            return

        if self._active_turn_interrupted.is_set():
            logger.debug(f"speak_text dropped: current turn #{self._active_turn_id} was interrupted.")
            return

        effective_turn = worker_turn if worker_turn is not None else self._active_turn_id
        self.wake_engine.activity()
        self.tts_pipeline.feed_text(text, turn_id=effective_turn)

    def stream_text_to_tts(self, token_or_text_stream: Iterator[str]) -> None:
        """
        Feed any arbitrary token or text stream directly to the decoupled TTS pipeline.
        Aborts stream if interrupted or superseded.
        """
        worker_turn = getattr(_turn_local, "turn_id", None)
        if worker_turn is not None and worker_turn != self._active_turn_id:
            return

        if self._active_turn_interrupted.is_set():
            return

        effective_turn = worker_turn if worker_turn is not None else self._active_turn_id
        self.wake_engine.activity()
        self.tts_pipeline.feed_stream(token_or_text_stream, turn_id=effective_turn)
