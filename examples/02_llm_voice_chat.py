#!/usr/bin/env python3
"""
Quick LLM Voice Chat Tester.
Standalone tester for any OpenAI-compatible LLM endpoint (llama.cpp, vLLM, Ollama, OpenAI).
Completely decoupled from the core voice pipeline.

Usage:
  python examples/02_llm_voice_chat.py
  python examples/02_llm_voice_chat.py --url http://127.0.0.1:8080/v1
  python examples/02_llm_voice_chat.py --url https://api.openai.com/v1 --api-key sk-... --model gpt-4o-mini
  python examples/02_llm_voice_chat.py --system-prompt "You are a sarcastic robot assistant."
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

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

from examples.llm_client import LLMClient
from voice_pipeline.benchmark import BenchmarkLogger
from voice_pipeline.config import PipelineConfig
from voice_pipeline.pipeline import VoicePipeline

console = Console()

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, fast, and conversational voice AI assistant. "
    "Always keep answers brief and natural, within 1 to 3 spoken sentences. "
    "Never use markdown tables, asterisks, or bullet points."
)


def main():
    parser = argparse.ArgumentParser(
        description="Quick LLM Voice Chat Tester & Live Benchmark Logger (OpenAI-compatible / llama.cpp / vLLM / Ollama)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible base URL (default: http://127.0.0.1:8080/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="not-needed",
        help="API Key if required (default: 'not-needed' for local servers)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3.5-4B",
        help="Model name (default: 'Qwen3.5-4B')",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt for the assistant",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to voice pipeline config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--wake",
        action="store_true",
        help="Enable wake word mode (respects config.yaml wake triggers & timeouts)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow thinking/reasoning models to output chain of thought (default: disabled)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="benchmarks",
        help="Folder to store session JSONL and latest_summary.json (default: benchmarks)",
    )
    parser.add_argument(
        "--expected-tok-s",
        type=float,
        default=50.0,
        help="Expected LLM generation speed in tokens/sec (default: 50.0 for Qwen3.5-4B)",
    )
    args = parser.parse_args()

    # Load voice pipeline configuration
    config = PipelineConfig.load_default_or_file(args.config)
    if args.wake:
        config.wake_word.enabled = True

    triggers_str = ", ".join([f"'{t.phrase}' ({t.timeout:.0f}s)" for t in config.wake_word.triggers]) or "none"
    wake_status = f"Enabled [{triggers_str}]" if config.wake_word.enabled else "Disabled (Continuous)"
    thinking_status = "Enabled" if args.enable_thinking else "Disabled (Fast Voice)"

    # Initialize Benchmark Logger
    benchmark = BenchmarkLogger(
        output_dir=args.benchmark_dir,
        model_name=args.model,
        expected_tok_s=args.expected_tok_s,
    )

    hw = benchmark.hardware_info
    console.print(
        Panel.fit(
            f"[bold cyan]Quick LLM Voice Chat Tester & Live Benchmark Logger[/bold cyan]\n"
            f"[dim]• Machine:[/dim] {hw.get('machine')} ({hw.get('cpu')})\n"
            f"[dim]• GPU / Driver:[/dim] {hw.get('gpu')} ({hw.get('nvidia_driver')}) | [dim]RAM:[/dim] {hw.get('ram_gb')} GB\n"
            f"[dim]• Endpoint:[/dim] {args.url} | [dim]Model:[/dim] {args.model} (~{args.expected_tok_s:.0f} tok/s)\n"
            f"[dim]• Benchmark Output:[/dim] [bold yellow]{benchmark.session_file}[/bold yellow]\n"
            f"[dim]• Voice:[/dim] {config.tts.voice} (speed: {config.tts.speed}x) | [dim]Barge-in:[/dim] {'Enabled' if config.general.allow_barge_in else 'Disabled'}\n"
            f"[dim]• Wake Mode:[/dim] {wake_status} | [dim]Thinking:[/dim] {thinking_status}\n"
            f"[dim]Press Ctrl+C to stop and display the benchmark summary table.[/dim]"
        )
    )

    # 1. Initialize Standalone LLM Client
    llm = LLMClient(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        disable_thinking=not args.enable_thinking,
    )

    # 2. Wire Voice Callbacks & Benchmark Observers
    def on_partial(live_text: str):
        sys.stdout.write(f"\r\033[K[Speaking]: {live_text}")
        sys.stdout.flush()

    def on_final(user_transcript: str):
        sys.stdout.write("\r\033[K")
        turn_id = pipeline.active_turn_id
        benchmark.start_turn(turn_id, user_transcript)

        console.print(f"\n[bold yellow]You:[/bold yellow] {user_transcript}")
        console.print("[cyan]Assistant generating...[/cyan]")
        had_sentences = False
        try:
            # Stream sentence by sentence directly to Kokoro TTS
            # Kokoro plays the 1st sentence while LLM generates the 2nd!
            for sentence in llm.stream_response(user_transcript):
                if pipeline.is_interrupted:
                    console.print("[dim yellow][Generation stopped: user interrupted][/dim yellow]")
                    break
                console.print(f"[bold green]Assistant:[/bold green] {sentence}")
                pipeline.speak_text(sentence)
                had_sentences = True
        except Exception as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
        finally:
            # Record LLM generation metrics (TTFT, tokens/sec, full reply)
            benchmark.record_llm_metrics(turn_id, llm.last_metrics)

            # Wait for Kokoro to synthesize and playback to complete (or barge-in interruption)
            if had_sentences:
                time.sleep(0.05)
                while pipeline.tts_pipeline.is_turn_busy(turn_id) and not pipeline.is_interrupted:
                    time.sleep(0.03)

            # Finalize turn metrics, persist to JSONL, and print developer card
            turn_data = benchmark.finish_turn(turn_id)
            if turn_data:
                benchmark.print_turn_card(turn_data, console)

    def on_interrupted(turn_id: Optional[int] = None):
        sys.stdout.write("\r\033[K")
        effective_turn = turn_id if turn_id is not None else pipeline.active_turn_id
        console.print(f"[red][Interrupted: Turn #{effective_turn} interrupted (~21ms cutoff)][/red]")
        benchmark.record_interruption(effective_turn)

    def on_chunk_synthesized(turn_id: int, chunk_id: int, text: str, gen_duration: float, duration_s: float):
        benchmark.record_tts_chunk(turn_id, chunk_id, text, gen_duration, duration_s)

    def on_playback_start(turn_id: int, chunk_id: int):
        benchmark.record_playback_start(turn_id, chunk_id)

    def on_wake(phrase: str, remainder: str, timeout: float):
        console.print(f"\n[bold green]⚡ Wake phrase detected ('{phrase}')! Active exchange for {timeout:.0f}s...[/bold green]")

    def on_sleep():
        console.print(f"\n[dim yellow]💤 Conversation timed out. Returning to wake word standby...[/dim yellow]")

    # 3. Create Voice Pipeline (pure voice engine, no LLM inside)
    pipeline = VoicePipeline(
        config=config,
        on_partial_transcript=on_partial,
        on_final_transcript=on_final,
        on_interrupted=on_interrupted,
        on_wake=on_wake,
        on_sleep=on_sleep,
        on_chunk_synthesized=on_chunk_synthesized,
        on_playback_start=on_playback_start,
    )

    pipeline.start()
    console.print("[green]Ready! Speak into your microphone...[/green]\n")

    def _sig_handler(sig, frame):
        console.print("\n[yellow]Stopping voice chat session...[/yellow]")
        pipeline.stop()
        benchmark.print_session_summary(console)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pipeline.stop()
        benchmark.print_session_summary(console)


if __name__ == "__main__":
    main()

