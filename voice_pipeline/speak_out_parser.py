"""
Incremental <speak-out> Tag Parser and Robust Sentence Chunker.
Directly implements Change 1, Change 3, and Change 4 from tts-changes-prompt.txt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional


# Known abbreviations that should never trigger a sentence break
COMMON_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "vs.", "etc.", "approx.", "dept.",
    "fig.", "inc.", "ltd.", "co.", "corp.", "st.", "ave.",
}


@dataclass
class TextSegment:
    """Represents a segment of text with speech eligibility."""
    text: str
    is_speakable: bool  # True if inside <speak-out>, False if screen-only
    is_boundary: bool = False  # True if this closes a <speak-out> block


@dataclass
class ChunkingConfig:
    """Configuration for Progressive and Adaptive chunking."""
    strategy: str = "progressive"  # "progressive", "adaptive", or "single"
    
    # Progressive strategy settings:
    # List of (chunk_count, sentences_per_chunk) stages.
    # -1 means all remaining chunks.
    # Example: [(2, 1), (-1, 3)] -> first 2 chunks have 1 sentence; subsequent chunks have 3 sentences.
    progressive_stages: List[tuple[int, int]] = field(
        default_factory=lambda: [(2, 1), (-1, 3)]
    )

    # Adaptive strategy settings (based on buffered audio duration in seconds):
    target_buffer_seconds: float = 5.0
    low_buffer_seconds: float = 1.5
    medium_buffer_seconds: float = 3.5
    
    low_buffer_sentences: int = 1
    medium_buffer_sentences: int = 2
    high_buffer_sentences: int = 3

    # Sentence boundary settings:
    min_words_for_boundary: int = 2


def is_abbreviation_or_decimal(text_before_dot: str, text_after_dot: str) -> bool:
    """
    Checks if a period is an abbreviation, decimal number, or initial.
    """
    # 1. Check for decimal numbers (e.g. 3.14, $10.50)
    if text_before_dot and text_before_dot[-1].isdigit() and text_after_dot and text_after_dot[0].isdigit():
        return True

    # 2. Check if the next word starts with a lowercase letter (sentences start with uppercase)
    stripped_after = text_after_dot.lstrip()
    if stripped_after and stripped_after[0].islower():
        return True

    # 3. Check for acronyms like U.S.A. or initials like J.K.
    words = text_before_dot.split()
    last_word = words[-1] if words else ""
    full_token = (last_word + ".").lower()

    if full_token in COMMON_ABBREVIATIONS:
        return True

    # Regex for acronyms like U.S.A. or A.B.C.
    if re.match(r'^([a-zA-Z]\.)+[a-zA-Z]?$', full_token):
        return True

    # Single letter initial (e.g. 'John F. Kennedy')
    if len(last_word) == 1 and last_word.isalpha():
        return True

    return False


class IncrementalSpeakOutParser:
    """
    Streaming parser that detects <speak-out> and </speak-out> tags incrementally.
    - Text outside tags is emitted as screen-only.
    - Text inside tags is emitted as speakable speech content.
    """

    OPEN_TAG = "<speak-out>"
    CLOSE_TAG = "</speak-out>"

    def __init__(self, fallback_to_all_speech: bool = False):
        """
        Args:
            fallback_to_all_speech: If True, and no <speak-out> tags are present in the
                                    entire response, treat all text as speakable.
        """
        self.fallback_to_all_speech = fallback_to_all_speech
        self.reset()

    def reset(self) -> None:
        self.inside_speak = False
        self.buffer = ""

    def process_token_stream(self, token_stream: Iterator[str]) -> Generator[TextSegment, None, None]:
        """
        Incrementally processes tokens and yields TextSegments.
        """
        self.reset()
        for token in token_stream:
            self.buffer += token

            while self.buffer:
                if not self.inside_speak:
                    # Looking for <speak-out>
                    open_pos = self.buffer.find(self.OPEN_TAG)
                    if open_pos != -1:
                        # Everything before open tag is screen-only
                        screen_text = self.buffer[:open_pos]
                        if screen_text:
                            yield TextSegment(text=screen_text, is_speakable=False)
                        self.inside_speak = True
                        self.buffer = self.buffer[open_pos + len(self.OPEN_TAG):]
                    else:
                        # Check if buffer ends with a prefix of "<speak-out>"
                        partial_match = False
                        for i in range(1, len(self.OPEN_TAG)):
                            if self.buffer.endswith(self.OPEN_TAG[:i]):
                                partial_match = True
                                safe_text = self.buffer[:-i]
                                if safe_text:
                                    yield TextSegment(text=safe_text, is_speakable=False)
                                self.buffer = self.buffer[-i:]
                                break
                        if not partial_match:
                            yield TextSegment(text=self.buffer, is_speakable=False)
                            self.buffer = ""
                        break

                else:
                    # Inside <speak-out>, looking for </speak-out>
                    close_pos = self.buffer.find(self.CLOSE_TAG)
                    if close_pos != -1:
                        speak_text = self.buffer[:close_pos]
                        if speak_text:
                            yield TextSegment(text=speak_text, is_speakable=True, is_boundary=True)
                        self.inside_speak = False
                        self.buffer = self.buffer[close_pos + len(self.CLOSE_TAG):]
                    else:
                        # Check if buffer ends with a prefix of "</speak-out>"
                        partial_match = False
                        for i in range(1, len(self.CLOSE_TAG)):
                            if self.buffer.endswith(self.CLOSE_TAG[:i]):
                                partial_match = True
                                safe_text = self.buffer[:-i]
                                if safe_text:
                                    yield TextSegment(text=safe_text, is_speakable=True)
                                self.buffer = self.buffer[-i:]
                                break
                        if not partial_match:
                            yield TextSegment(text=self.buffer, is_speakable=True)
                            self.buffer = ""
                        break

        # Flush any remaining buffer
        if self.buffer:
            yield TextSegment(
                text=self.buffer,
                is_speakable=self.inside_speak,
                is_boundary=self.inside_speak,
            )
            self.buffer = ""


class RobustSentenceChunker:
    """
    Intelligently splits speech text into sentences and applies
    Progressive or Adaptive chunking strategies.
    
    Adheres strictly to rules:
    - Never split abbreviations (Dr., e.g., U.S.A.) or numbers (3.14).
    - Never discard or delay short <speak-out> blocks (e.g. "Yes.", "Okay.").
    - Chunks are batched according to progressive stages or buffer depth.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.chunk_index = 0
        self.sentence_buffer: List[str] = []
        self.unparsed_text = ""

    def reset(self) -> None:
        self.chunk_index = 0
        self.sentence_buffer.clear()
        self.unparsed_text = ""

    def get_target_sentence_count(self, current_buffered_audio_s: float = 0.0) -> int:
        """Determines how many sentences to combine into the next TTS chunk."""
        if self.config.strategy == "adaptive":
            if current_buffered_audio_s < self.config.low_buffer_seconds:
                return self.config.low_buffer_sentences
            elif current_buffered_audio_s < self.config.medium_buffer_seconds:
                return self.config.medium_buffer_sentences
            else:
                return self.config.high_buffer_sentences

        elif self.config.strategy == "progressive":
            # Walk through progressive stages
            count_so_far = 0
            for max_chunks, s_per_chunk in self.config.progressive_stages:
                if max_chunks == -1 or self.chunk_index < (count_so_far + max_chunks):
                    return s_per_chunk
                count_so_far += max_chunks
            return 1
        else:
            return 1

    def feed_text(
        self,
        text: str,
        is_block_end: bool = False,
        current_buffered_audio_s: float = 0.0,
    ) -> Generator[str, None, None]:
        """
        Feeds text into the sentence chunker.
        Yields ready-to-synthesize TTS text chunks based on the chunking strategy.
        """
        self.unparsed_text += text

        # Regex finding candidate sentence terminators (. ! ? \n)
        terminator_regex = re.compile(r'([.!?]+|\n+)')
        pos = 0

        while True:
            match = terminator_regex.search(self.unparsed_text, pos)
            if not match:
                break

            term_start = match.start()
            term_end = match.end()
            term = match.group(0)

            text_before = self.unparsed_text[:term_start]
            text_after = self.unparsed_text[term_end:]

            # If terminator is a period, check for abbreviation / decimal
            if "." in term and is_abbreviation_or_decimal(text_before, text_after):
                pos = term_end
                continue

            # Need at least whitespace after punctuation if text continues
            if text_after and not text_after[0].isspace() and "\n" not in term:
                pos = term_end
                continue

            sentence = self.unparsed_text[:term_end].strip()
            self.unparsed_text = self.unparsed_text[term_end:].lstrip()
            pos = 0

            if sentence:
                self.sentence_buffer.append(sentence)

            # Check if we have enough sentences to form a chunk
            target_sentences = self.get_target_sentence_count(current_buffered_audio_s)
            if len(self.sentence_buffer) >= target_sentences:
                chunk = " ".join(self.sentence_buffer)
                self.sentence_buffer.clear()
                self.chunk_index += 1
                yield chunk

        # If this is the end of a <speak-out> block, flush any pending sentences/text
        if is_block_end:
            if self.unparsed_text.strip():
                self.sentence_buffer.append(self.unparsed_text.strip())
                self.unparsed_text = ""

            if self.sentence_buffer:
                chunk = " ".join(self.sentence_buffer)
                self.sentence_buffer.clear()
                self.chunk_index += 1
                yield chunk
