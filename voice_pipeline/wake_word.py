"""
Wake Word and Wake Sentence Detection Engine for Voice Pipeline.

Supports:
- Multi-phrase wake detection (both single words and full sentences)
- Per-phrase configurable timeout durations
- Natural two-state machine: STANDBY (sleeping) and ACTIVE (awake)
- Automatic phrase stripping (e.g., "Hey Jarvis, what time is it?" -> "what time is it?")
- Conversation exchange timeout: user can continue speaking without repeating wake words
- Inactivity countdown timer with pause/refresh during assistant speech
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class WakeState(str, Enum):
    """Wake engine operating state."""
    STANDBY = "standby"  # Sleeping / waiting for wake word
    ACTIVE = "active"    # Awake / interactive conversation mode


@dataclass
class WakeTriggerConfig:
    """Configuration for a single wake word or wake sentence."""
    phrase: str                       # e.g. "hey assistant", "jarvis", "take a note"
    timeout: float = 15.0             # Inactivity timeout in seconds after this wake phrase

    def __post_init__(self):
        self.phrase = self.phrase.strip()


@dataclass
class WakeWordConfig:
    """Master configuration for wake word & sentence detection."""
    enabled: bool = False             # Master toggle for wake_mode
    default_timeout: float = 15.0     # Default timeout if not specified per trigger
    strip_wake_phrase: bool = False   # If False (default), sends complete transcription to agent
    prefix_only: bool = False         # If False (default), triggers if sentence contains wake phrase anywhere
    triggers: List[WakeTriggerConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WakeWordConfig":
        """Builds WakeWordConfig from a dictionary supporting multiple formats."""
        enabled = data.get("enabled", False)
        default_timeout = float(data.get("default_timeout", 15.0))
        strip_wake_phrase = data.get("strip_wake_phrase", False)
        prefix_only = data.get("prefix_only", False)

        raw_triggers = data.get("triggers", [])
        triggers: List[WakeTriggerConfig] = []

        if isinstance(raw_triggers, dict):
            # Dict format: {"hey assistant": 20.0, "jarvis": 15.0}
            for phrase, timeout in raw_triggers.items():
                triggers.append(WakeTriggerConfig(phrase=str(phrase), timeout=float(timeout)))
        elif isinstance(raw_triggers, list):
            for item in raw_triggers:
                if isinstance(item, dict):
                    # List of dicts: [{"phrase": "hey assistant", "timeout": 20.0}]
                    phrase = item.get("phrase", "")
                    timeout = float(item.get("timeout", default_timeout))
                    if phrase:
                        triggers.append(WakeTriggerConfig(phrase=phrase, timeout=timeout))
                elif isinstance(item, str) and item.strip():
                    # Simple list of strings: ["hey assistant", "jarvis"]
                    triggers.append(WakeTriggerConfig(phrase=item.strip(), timeout=default_timeout))

        # If no triggers specified but enabled, provide sensible defaults
        if enabled and not triggers:
            triggers = [
                WakeTriggerConfig(phrase="hey assistant", timeout=default_timeout),
                WakeTriggerConfig(phrase="assistant", timeout=default_timeout),
            ]

        return cls(
            enabled=enabled,
            default_timeout=default_timeout,
            strip_wake_phrase=strip_wake_phrase,
            prefix_only=prefix_only,
            triggers=triggers,
        )


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


class WakeWordEngine:
    """
    Stateful Wake Word and Wake Sentence Engine.
    
    Manages transitions between STANDBY and ACTIVE states, matching triggers,
    managing conversation timeout counters, and stripping wake words.
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        is_busy_callback: Optional[Callable[[], bool]] = None,
        on_wake: Optional[Callable[[str, str, float], None]] = None,
        on_sleep: Optional[Callable[[], None]] = None,
        on_state_change: Optional[Callable[[WakeState], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or WakeWordConfig()
        self.is_busy_callback = is_busy_callback
        self.on_wake = on_wake
        self.on_sleep = on_sleep
        self.on_state_change = on_state_change
        self.on_log = on_log or (lambda msg: None)

        self._state = WakeState.STANDBY if self.config.enabled else WakeState.ACTIVE
        self._current_timeout = self.config.default_timeout
        self._last_activity_time = time.time()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None

        # Sort triggers by word count descending (e.g. "hey my assistant" before "assistant")
        self._sorted_triggers = sorted(
            self.config.triggers,
            key=lambda t: len(t.phrase.split()),
            reverse=True,
        )

    @property
    def is_enabled(self) -> bool:
        """True if wake_mode is currently enabled."""
        return self.config.enabled

    @property
    def state(self) -> WakeState:
        """Current wake state (STANDBY or ACTIVE)."""
        with self._lock:
            return self._state

    @property
    def is_awake(self) -> bool:
        """True if in ACTIVE conversation state or if wake_mode is disabled."""
        if not self.config.enabled:
            return True
        with self._lock:
            return self._state == WakeState.ACTIVE

    @property
    def current_timeout(self) -> float:
        """Current active timeout duration in seconds."""
        with self._lock:
            return self._current_timeout

    @property
    def time_remaining(self) -> float:
        """Remaining seconds before returning to STANDBY (0 if already in STANDBY)."""
        with self._lock:
            if self._state != WakeState.ACTIVE or not self.config.enabled:
                return 0.0
            elapsed = time.time() - self._last_activity_time
            return max(0.0, self._current_timeout - elapsed)

    def start(self) -> None:
        """Start the background inactivity timer thread."""
        if not self.config.enabled:
            return
        if self._timer_thread and self._timer_thread.is_alive():
            return

        self._stop_event.clear()
        self._timer_thread = threading.Thread(
            target=self._countdown_loop,
            daemon=True,
            name="Wake-Inactivity-Timer",
        )
        self._timer_thread.start()

    def stop(self) -> None:
        """Stop the background inactivity timer thread."""
        self._stop_event.set()
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)
            self._timer_thread = None

    def activity(self) -> None:
        """Refreshes the inactivity timeout countdown timer."""
        with self._lock:
            self._last_activity_time = time.time()

    def wake(self, phrase: str = "manual", timeout: Optional[float] = None) -> None:
        """Programmatically wake the engine into ACTIVE state."""
        with self._lock:
            prev_state = self._state
            self._state = WakeState.ACTIVE
            self._current_timeout = timeout or self.config.default_timeout
            self._last_activity_time = time.time()

        if prev_state != WakeState.ACTIVE:
            self.on_log(f"[bold green]⚡ Wake activated (phrase='{phrase}', timeout={self._current_timeout:.1f}s)[/bold green]")
            if self.on_wake:
                self.on_wake(phrase, "", self._current_timeout)
            if self.on_state_change:
                self.on_state_change(WakeState.ACTIVE)

    def sleep(self) -> None:
        """Programmatically put the engine into STANDBY state."""
        with self._lock:
            prev_state = self._state
            self._state = WakeState.STANDBY

        if prev_state != WakeState.STANDBY:
            self.on_log("[dim yellow]💤 Session timeout expired. Returning to wake word standby...[/dim yellow]")
            if self.on_sleep:
                self.on_sleep()
            if self.on_state_change:
                self.on_state_change(WakeState.STANDBY)

    def match_trigger(self, text: str) -> Optional[Tuple[WakeTriggerConfig, str]]:
        """
        Checks if text matches any configured wake word or sentence (case-insensitive).
        
        Returns:
            Tuple of (matched_trigger, text_to_forward) if matched, else None.
            If strip_wake_phrase is False (default), text_to_forward is the complete transcription.
        """
        clean_input = normalize_text(text)
        if not clean_input:
            return None

        input_words = clean_input.split()

        for trigger in self._sorted_triggers:
            clean_phrase = normalize_text(trigger.phrase)
            phrase_words = clean_phrase.split()
            phrase_len = len(phrase_words)

            if not phrase_words:
                continue

            matched = False
            match_start_idx = -1

            if self.config.prefix_only:
                # 1. Prefix match (speech starts with the wake phrase)
                if len(input_words) >= phrase_len and input_words[:phrase_len] == phrase_words:
                    matched = True
                    match_start_idx = 0
            else:
                # 2. Contains match: checks if sentence contains the wake phrase anywhere (case-insensitive)
                for i in range(len(input_words) - phrase_len + 1):
                    if input_words[i : i + phrase_len] == phrase_words:
                        matched = True
                        match_start_idx = i
                        break

            if matched:
                if not self.config.strip_wake_phrase:
                    # Send the complete transcription as-is, preserving original casing and punctuation
                    return trigger, text.strip()
                else:
                    # Strip wake phrase and return remainder
                    pattern = r"(?:,\s*)?\b" + r"\s+".join(map(re.escape, phrase_words)) + r"\b[\s,\.!?:]*"
                    m = re.search(pattern, text, flags=re.IGNORECASE)
                    if m:
                        pre = text[:m.start()].strip()
                        suf = text[m.end():].strip()
                        punct = m.group(0).strip()
                        if punct and not suf and not pre.endswith((".", "?", "!")):
                            for ch in reversed(punct):
                                if ch in ".?!":
                                    pre += ch
                                    break
                        remainder = (pre + " " + suf).strip()
                    else:
                        remainder = text.strip()
                    return trigger, remainder

        return None

    def process_transcript(self, transcript: str) -> Tuple[bool, str]:
        """
        Processes a finalized STT transcript according to the current wake state.
        
        Returns:
            (should_forward_to_agent: bool, processed_text: str)
        """
        if not self.config.enabled:
            return True, transcript

        raw = transcript.strip()
        if not raw:
            return False, ""

        with self._lock:
            current_state = self._state

        if current_state == WakeState.STANDBY:
            match_result = self.match_trigger(raw)
            if match_result is None:
                logger.debug(f"[WakeWord] Standby ignoring non-wake speech: {raw}")
                return False, ""

            trigger, remainder = match_result
            self.wake(phrase=trigger.phrase, timeout=trigger.timeout)

            if remainder:
                return True, remainder
            else:
                return False, ""

        else:
            self.activity()
            return True, raw

    def _countdown_loop(self) -> None:
        """Background thread monitoring conversational exchange timeout."""
        while not self._stop_event.is_set():
            time.sleep(0.25)

            with self._lock:
                if self._state != WakeState.ACTIVE or not self.config.enabled:
                    continue

            if self.is_busy_callback and self.is_busy_callback():
                self.activity()
                continue

            with self._lock:
                elapsed = time.time() - self._last_activity_time
                timed_out = elapsed >= self._current_timeout

            if timed_out:
                self.sleep()
