#!/usr/bin/env python3
"""
Example 1: Pure Real-Time STT (Voice Input Only)
Uses the official Hugging Face VAD + Parakeet TDT STT engine on CPU.

Features:
- Full 500ms pre-speech buffer (retains initial words without clipping)
- 800ms natural silence turn closure
- Real-time progressive interim updates while speaking
- High-accuracy final transcript with gibberish filtering
"""

import logging
import sys
import time
import warnings
from pathlib import Path

# Suppress harmless PyTorch/Kokoro and upstream deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("speech_to_speech.VAD.vad_handler").setLevel(logging.ERROR)

# Add project root to sys.path so voice_pipeline is always importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel

from voice_pipeline.pipeline import VoicePipeline

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Real-Time Speech-to-Text (Official HF VAD + Parakeet TDT on CPU)[/bold cyan]\n"
            "[dim]Speaks into your microphone. Watch the transcription update in real-time as you speak![/dim]\n"
            "[dim]Press Ctrl+C to exit.[/dim]"
        )
    )

    def on_partial(interim_text: str):
        # Real-time interim update while actively speaking
        sys.stdout.write(f"\r\033[K[Realtime]: {interim_text}")
        sys.stdout.flush()

    def on_final(final_transcript: str):
        # Finalized turn (>800ms silence)
        sys.stdout.write("\r\033[K")
        console.print(f"[bold yellow]Final Transcript: {final_transcript}[/bold yellow]\n")

    # Initialize the Voice Pipeline with official HF STT parameters
    pipeline = VoicePipeline(
        vad_threshold=0.6,
        min_silence_ms=800,
        min_speech_ms=384,
        speech_pad_ms=500,       # 500ms pre-speech buffer prevents dropping first 1-2 words
        min_speech_continuation_ms=192,  # Hysteresis prevents splitting pauses in sentences
        enable_live_transcription=True,
        stt_language="en",
        device="cpu",
        on_partial_transcript=on_partial,
        on_final_transcript=on_final,
    )

    pipeline.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping microphone...[/yellow]")
        pipeline.stop()
        console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
