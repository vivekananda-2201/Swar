"""
LLM Client and Sentence Streamer (Helper for Example 02).
Supports any OpenAI-compatible API (OpenAI, Ollama, vLLM, LM Studio, local llama.cpp),
as well as sentence-by-sentence streaming for low-latency TTS.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Generator, Iterator, List, Optional

logger = logging.getLogger(__name__)


def strip_thinking_tokens(token_stream: Iterator[str]) -> Generator[str, None, None]:
    """
    Filters out <think>...</think> and <thought>...</thought> blocks from a streaming
    token generator in real time. Prevents thinking/reasoning models from speaking
    their inner chain of thought out loud.
    """
    in_thinking = False
    buffer = ""

    for token in token_stream:
        buffer += token
        while buffer:
            if in_thinking:
                if "</think>" in buffer:
                    _, buffer = buffer.split("</think>", 1)
                    in_thinking = False
                elif "</thought>" in buffer:
                    _, buffer = buffer.split("</thought>", 1)
                    in_thinking = False
                else:
                    if len(buffer) > 16:
                        buffer = buffer[-16:]
                    else:
                        buffer = ""
                    break
            else:
                if "<think>" in buffer:
                    before, after = buffer.split("<think>", 1)
                    if before:
                        yield before
                    buffer = after
                    in_thinking = True
                elif "<thought>" in buffer:
                    before, after = buffer.split("<thought>", 1)
                    if before:
                        yield before
                    buffer = after
                    in_thinking = True
                else:
                    match = re.search(r"<(t|th|thi|thin|think|thou|though|thought)?$", buffer)
                    if match and match.start() < len(buffer):
                        safe_text = buffer[:match.start()]
                        buffer = buffer[match.start():]
                        if safe_text:
                            yield safe_text
                        break
                    else:
                        yield buffer
                        buffer = ""
                        break

    if buffer and not in_thinking:
        yield buffer


def split_sentences_stream(token_stream: Iterator[str]) -> Generator[str, None, None]:
    """
    Takes a stream of tokens from an LLM and buffers them until a sentence boundary
    is reached (., !, ?, \n). Emits full sentences as soon as they are formed.
    """
    buffer = ""
    sentence_end_pattern = re.compile(r"([.!?]+[\s\n]+|\n+)")

    for token in token_stream:
        buffer += token
        while True:
            match = sentence_end_pattern.search(buffer)
            if match:
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                buffer = buffer[end_pos:]
                if sentence:
                    yield sentence
            else:
                break

    if buffer.strip():
        yield buffer.strip()


class LLMClient:
    """
    Connects to any LLM endpoint or custom agent logic.
    Supports automatic thinking suppression for reasoning models.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "empty",
        model: str = "default",
        system_prompt: str = "You are a friendly, fast, and concise voice AI assistant. Keep responses short and conversational. Do not output thinking tags or chain of thought.",
        custom_handler: Optional[Callable[[str], str | Iterator[str]]] = None,
        disable_thinking: bool = True,
    ):
        self.base_url = base_url
        self.api_key = api_key or "empty"
        self.model = model
        self.system_prompt = system_prompt
        self.custom_handler = custom_handler
        self.disable_thinking = disable_thinking
        self.conversation_history: List[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self._client = None
        if not custom_handler:
            try:
                from openai import OpenAI

                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except Exception as e:
                logger.warning(f"OpenAI client initialization notice: {e}")

    def add_message(self, role: str, content: str) -> None:
        """Appends a turn to conversation memory."""
        self.conversation_history.append({"role": role, "content": content})

    def reset_history(self) -> None:
        """Reset conversation memory."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def stream_response(self, user_prompt: str) -> Generator[str, None, None]:
        """
        Sends the user transcript to the LLM.
        Yields complete sentences one by one as they are generated.
        """
        self.add_message("user", user_prompt)

        # If user provided a custom agent handler
        if self.custom_handler is not None:
            result = self.custom_handler(user_prompt)
            tokens = iter([result]) if isinstance(result, str) else result
            if self.disable_thinking:
                tokens = strip_thinking_tokens(tokens)
            full_reply = ""
            for sentence in split_sentences_stream(tokens):
                full_reply += " " + sentence
                yield sentence
            self.add_message("assistant", full_reply.strip())
            return

        # Default OpenAI-compatible API
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        create_kwargs = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": True,
            "temperature": 0.7,
        }

        # If thinking is disabled, pass server-side flags for Ollama and vLLM
        if self.disable_thinking:
            create_kwargs["extra_body"] = {
                "think": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        try:
            try:
                response = self._client.chat.completions.create(**create_kwargs)
            except Exception as e:
                # If server does not support extra_body parameters (e.g. strict OpenAI API), retry without it
                if "extra_body" in create_kwargs:
                    logger.debug(f"Server rejected extra_body thinking flags ({e}); falling back to standard request")
                    create_kwargs.pop("extra_body", None)
                    response = self._client.chat.completions.create(**create_kwargs)
                else:
                    raise

            def _token_generator():
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta

            token_stream = _token_generator()
            if self.disable_thinking:
                token_stream = strip_thinking_tokens(token_stream)

            full_reply = ""
            for sentence in split_sentences_stream(token_stream):
                full_reply += " " + sentence
                yield sentence

            self.add_message("assistant", full_reply.strip())

        except Exception as e:
            logger.error(f"LLM API request failed: {e}")
            fallback = "I heard you, but my language model endpoint is currently unavailable."
            yield fallback
            self.add_message("assistant", fallback)
