"""
Standalone LLM helper re-exported from examples.llm_client.
The core voice_pipeline is completely decoupled and does not require this file.
"""

from examples.llm_client import LLMClient, split_sentences_stream

__all__ = ["LLMClient", "split_sentences_stream"]
