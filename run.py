#!/usr/bin/env python3
"""
Main CLI Launcher for Voice Pipeline.
Runs Silero VAD + Parakeet TDT STT + Kokoro TTS natively on CPU.

Supports:
  python run.py --mode stt_only
  python run.py --mode chat --llm_url http://127.0.0.1:8080/v1 --kokoro_voice af_bella
"""

import argparse
import signal
import sys
import time
from rich.console import Console

from voice_pipeline.config import PipelineConfig
from voice_pipeline.pipeline import VoicePipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Voice Pipeline (Silero VAD + Parakeet STT + Kokoro TTS on CPU)")
    parser.add_argument("--mode", choices=["chat", "stt_only"], default="chat", help="Mode: chat (with LLM & TTS) or stt_only (transcription only)")
    parser.add_argument("--min_silence_ms", type=int, default=800, help="VAD minimum silence in ms before closing turn (default: 800)")
    parser.add_argument("--stt_language", type=str, default="en", help="Language for transcription (default: en)")
    parser.add_argument("--kokoro_voice", type=str, default="af_bella", help="Kokoro TTS voice (default: af_bella)")
    parser.add_argument("--tts_speed", type=float, default=1.0, help="Kokoro TTS speed multiplier (default: 1.0)")
    parser.add_argument("--llm_url", type=str, default="http://127.0.0.1:8080/v1", help="OpenAI-compatible LLM endpoint")
    parser.add_argument("--llm_api_key", type=str, default="empty", help="LLM API Key")
    parser.add_argument("--llm_model", type=str, default="default", help="LLM Model Name")
    parser.add_argument("--enable_live_transcription", action="store_true", default=True, help="Enable interim progressive transcription while speaking")
    parser.add_argument("--enable_thinking", action="store_true", help="Allow thinking models to output internal thoughts (default: disabled)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to voice pipeline config.yaml (default: config.yaml)")
    parser.add_argument("--wake", action="store_true", help="Enable wake word mode (e.g. 'relic', 'hey assistant')")

    args = parser.parse_args()

    # Load base config if available
    config = PipelineConfig.load_default_or_file(args.config)
    if args.wake:
        config.wake_word.enabled = True

    llm = None
    if args.mode == "chat":
        from examples.llm_client import LLMClient
        llm = LLMClient(
            base_url=args.llm_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            disable_thinking=not args.enable_thinking,
        )

    def on_partial(delta: str):
        sys.stdout.write(f"\r\033[K[Realtime]: {delta}")
        sys.stdout.flush()

    def on_final(transcript: str):
        sys.stdout.write("\r\033[K")
        console.print(f"[bold yellow]User: {transcript}[/bold yellow]")
        if llm is not None:
            console.print("[cyan]Assistant generating...[/cyan]")
            for sentence in llm.stream_response(transcript):
                if pipeline.is_interrupted:
                    console.print("[dim yellow][Generation stopped: user interrupted][/dim yellow]")
                    break
                console.print(f"[bold green]Assistant:[/bold green] {sentence}")
                pipeline.speak_text(sentence)

    pipeline = VoicePipeline(
        config=config,
        vad_threshold=0.5,
        min_silence_ms=args.min_silence_ms,
        stt_language=args.stt_language,
        enable_live_transcription=args.enable_live_transcription,
        tts_voice=args.kokoro_voice,
        tts_speed=args.tts_speed,
        device="cpu",
        wake_mode=config.wake_word.enabled,
        on_partial_transcript=on_partial,
        on_final_transcript=on_final,
    )

    pipeline.start()

    def _sig_handler(sig, frame):
        console.print("\n[yellow]Stopping Voice Pipeline...[/yellow]")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()
