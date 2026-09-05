#!/usr/bin/env python3
"""
Example 3: Custom AI Agent with <speak> Tag Parsing at Agent Level.

Key Architecture:
- Silero VAD (800ms silence turn detection, 500ms speech pad)
- Parakeet TDT STT (Real-time live progressive transcription)
- Mock LLM generates responses with thoughts, logs, and <speak>...</speak> tags.
- Agent Backend PARSES the <speak> tags itself.
- Voice Pipeline receives ONLY the clean extracted text (no tags in the pipeline).
- Decoupled Kokoro TTS synthesizes and speaks in real-time.
"""

import argparse
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

# Suppress harmless PyTorch/Kokoro and upstream deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("speech_to_speech.VAD.vad_handler").setLevel(logging.ERROR)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel

from voice_pipeline.config import PipelineConfig
from voice_pipeline.pipeline import VoicePipeline

console = Console()


# ==============================================================================
# 1. MOCK LLM (Hardcoded output containing thoughts, logs & <speak> tags)
# ==============================================================================
def mock_llm_generate(user_transcript: str) -> str:
    """
    Simulates a smart AI Agent output.
    Contains:
    - Internal thoughts / chain of thought (screen only, never spoken)
    - Diagnostics / tool outputs (screen only, never spoken)
    - Speech content wrapped in <speak>...</speak> tags
    """
    return (
        f"[Thought: User said '{user_transcript}'. Analyzing intent and checking status...]\n"
        f"[Tool Call: system_check(node='local_cpu') -> Status: 100% HEALTHY]\n"
        f"<speak>Hello! I heard you say: {user_transcript}.</speak>\n"
        f"[Action: Updated local dashboard widget in 12ms]\n"
        f"<speak>All services are operational on CPU. The tag parsing is happening entirely on my side, and the voice engine is receiving pure text!</speak>\n"
        f"[Log: Agent response completed.]"
    )


# ==============================================================================
# 2. AGENT-SIDE <speak> TAG PARSER (Implemented in Agent Code, NOT pipeline)
# ==============================================================================
def parse_speak_tags(model_output: str, tag_name: str = "speak") -> list[str]:
    """
    Extracts all text inside <tag_name>...</tag_name> tags.
    Anything outside the tags is ignored for voice playback.
    """
    pattern = re.compile(rf"<{tag_name}>(.*?)</{tag_name}>", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(model_output)
    return [match.strip() for match in matches if match.strip()]


# ==============================================================================
# 3. AGENT CONTROLLER
# ==============================================================================
def my_custom_ai_agent(user_transcript: str, pipeline: VoicePipeline):
    """
    Agent workflow:
    1. Generate response from (mock) LLM containing <speak> tags.
    2. Display the full model output so you can see tags and screen thoughts.
    3. Parse out only what is inside <speak>...</speak>.
    4. Send clean plain text to pipeline.speak_text().
    """
    tag = "speak"
    console.print(f"\n[bold magenta]🎤 Agent Received Audio Query:[/bold magenta] {user_transcript}\n")

    # Step 1: Get Model Output (simulated hardcoded LLM)
    raw_model_output = mock_llm_generate(user_transcript)

    # Step 2: Show the RAW model output (with tags & thoughts visible!)
    console.print(
        Panel(
            raw_model_output,
            title=f"[bold cyan]1. Full Model Output (with <{tag}> tags & thoughts)[/bold cyan]",
            subtitle="[dim]Visible on screen only[/dim]",
            border_style="cyan",
        )
    )

    # Step 3: Agent parses what is inside the speak tags
    spoken_segments = parse_speak_tags(raw_model_output, tag_name=tag)

    # Step 4: Show what was extracted for the voice pipeline
    extracted_text = " ".join(spoken_segments)
    console.print(
        Panel(
            extracted_text if extracted_text else "[dim italic]No <speak> tags found[/dim italic]",
            title="[bold green]2. Extracted Voice Text (Sent to Voice Pipeline)[/bold green]",
            subtitle="[dim]Pure plain text sent to pipeline.speak_text()[/dim]",
            border_style="green",
        )
    )

    # Step 5: Send plain text to the decoupled Kokoro TTS pipeline
    # The pipeline knows NOTHING about tags - it just speaks the text!
    if extracted_text and not pipeline.is_interrupted:
        pipeline.speak_text(extracted_text)


def main():
    parser = argparse.ArgumentParser(description="Custom AI Agent with Agent-side <speak> Tag Parsing")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--test", action="store_true", help="Run a one-shot simulated test immediately without mic")
    parser.add_argument("--wake", action="store_true", help="Enable wake word mode (respects config.yaml wake triggers & timeouts)")
    args = parser.parse_args()

    # Load master pipeline configuration from config.yaml
    config = PipelineConfig.load_default_or_file(args.config)
    if args.wake:
        config.wake_word.enabled = True

    triggers_str = ", ".join([f"'{t.phrase}' ({t.timeout:.0f}s)" for t in config.wake_word.triggers]) or "none"
    wake_status = f"Enabled [{triggers_str}]" if config.wake_word.enabled else "Disabled (Continuous)"

    console.print(
        Panel.fit(
            f"[bold cyan]Custom AI Agent Voice Pipeline (CPU Native)[/bold cyan]\n"
            f"[dim]• Config file: {args.config}[/dim]\n"
            f"[dim]• Silero VAD (min silence: {config.vad.min_silence_ms}ms, speech pad: {config.vad.speech_pad_ms}ms)[/dim]\n"
            f"[dim]• Parakeet TDT STT ({config.stt.model_name})[/dim]\n"
            f"[dim]• Decoupled Kokoro TTS (voice: {config.tts.voice}, speed: {config.tts.speed}x)[/dim]\n"
            f"[dim]• Chunking Strategy: {config.chunking.strategy} (stages: {config.chunking.progressive_stages})[/dim]\n"
            f"[dim]• Barge-in (Interruption): {'Enabled (Full-Duplex)' if config.general.allow_barge_in else 'Disabled (Noise/Speaker Safe)'}[/dim]\n"
            f"[dim]• Wake Mode: {wake_status}[/dim]\n"
            f"[dim]• Agent-Side <speak> Tag Parsing[/dim]\n"
            f"[dim]Press Ctrl+C to stop.[/dim]"
        )
    )

    def on_partial(interim_text: str):
        # Emitted every 500ms WHILE user is actively speaking
        sys.stdout.write(f"\r\033[K[Speaking]: {interim_text}")
        sys.stdout.flush()

    def on_final(final_transcript: str):
        # Emitted when user finishes speaking (>800ms silence)
        sys.stdout.write("\r\033[K")
        console.print(f"[bold yellow]User Speech:[/bold yellow] {final_transcript}")

        # Send the transcript to your custom AI agent
        my_custom_ai_agent(final_transcript, pipeline)

    def on_interrupted():
        console.print("[red][User started speaking - interrupted playback!][/red]")

    def on_wake(phrase: str, remainder: str, timeout: float):
        console.print(f"\n[bold green]⚡ Wake phrase detected ('{phrase}')! Active exchange for {timeout:.0f}s...[/bold green]")

    def on_sleep():
        console.print(f"\n[dim yellow]💤 Exchange timed out. Returning to wake word standby...[/dim yellow]")

    # Create the complete voice pipeline from config.yaml
    pipeline = VoicePipeline(
        config=config,
        on_partial_transcript=on_partial,
        on_final_transcript=on_final,
        on_interrupted=on_interrupted,
        on_wake=on_wake,
        on_sleep=on_sleep,
    )

    pipeline.start()

    if args.test:
        console.print("[yellow]Running simulated one-shot test...[/yellow]")
        my_custom_ai_agent("Can you hear me?", pipeline)
        time.sleep(8.0)
        pipeline.stop()
        console.print("[green]Test completed.[/green]")
        return

    console.print("[green]Ready! Speak into your microphone...[/green]\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
