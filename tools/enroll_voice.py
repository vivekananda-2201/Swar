#!/usr/bin/env python3
"""
Standalone Voiceprint Enrollment and Acoustic Calibration Studio for Swar.

Features:
1. User Enrollment: Record a clean voice sample with silence check to create a personalized voiceprint (.npy).
2. Multi-Kokoro Acoustic Calibration:
   - Lists available Kokoro voices (with language, gender, and description).
   - Prompts user to preview Kokoro sound through laptop speakers before calibration.
   - Prompts user to ensure a quiet/silent environment before recording.
   - Plays Kokoro through speakers while recording through mic to capture machine speaker resonance.
   - Enrolls multiple Kokoro voices independently.
3. Deletion Studio: Delete enrolled speaker voiceprints or Kokoro voiceprints (via CLI or interactive selector).
4. List & Live Verification: View all enrolled voiceprints and test live recognition.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_pipeline.config import PipelineConfig
from voice_pipeline.speaker_verification import VoiceprintManager, DEFAULT_VOICEPRINTS_DIR, DEFAULT_MODEL_DIR

console = Console()

# Curated popular Kokoro voices with metadata
POPULAR_KOKORO_VOICES = [
    # American English - Female
    {"id": "af_bella", "name": "Bella", "lang": "en-us", "gender": "Female", "desc": "Warm, conversational (Default)"},
    {"id": "af_heart", "name": "Heart", "lang": "en-us", "gender": "Female", "desc": "Warm, natural tone"},
    {"id": "af_sarah", "name": "Sarah", "lang": "en-us", "gender": "Female", "desc": "Crisp, professional"},
    {"id": "af_nicole", "name": "Nicole", "lang": "en-us", "gender": "Female", "desc": "Soft, whispery tone"},
    {"id": "af_sky", "name": "Sky", "lang": "en-us", "gender": "Female", "desc": "Bright, cheerful"},
    {"id": "af_alloy", "name": "Alloy", "lang": "en-us", "gender": "Female", "desc": "Balanced, modern"},
    {"id": "af_aoede", "name": "Aoede", "lang": "en-us", "gender": "Female", "desc": "Melodic, dynamic"},
    {"id": "af_jessica", "name": "Jessica", "lang": "en-us", "gender": "Female", "desc": "Clear, engaging"},
    {"id": "af_river", "name": "River", "lang": "en-us", "gender": "Female", "desc": "Smooth, relaxed"},
    # American English - Male
    {"id": "am_adam", "name": "Adam", "lang": "en-us", "gender": "Male", "desc": "Deep, authoritative"},
    {"id": "am_michael", "name": "Michael", "lang": "en-us", "gender": "Male", "desc": "Natural, conversational"},
    {"id": "am_echo", "name": "Echo", "lang": "en-us", "gender": "Male", "desc": "Resonant, calm"},
    {"id": "am_eric", "name": "Eric", "lang": "en-us", "gender": "Male", "desc": "Upbeat, friendly"},
    {"id": "am_fenrir", "name": "Fenrir", "lang": "en-us", "gender": "Male", "desc": "Deep, gritty"},
    {"id": "am_liam", "name": "Liam", "lang": "en-us", "gender": "Male", "desc": "Clear, articulate"},
    {"id": "am_onyx", "name": "Onyx", "lang": "en-us", "gender": "Male", "desc": "Rich, resonant"},
    {"id": "am_puck", "name": "Puck", "lang": "en-us", "gender": "Male", "desc": "Playful, energetic"},
    # British English - Female
    {"id": "bf_emma", "name": "Emma", "lang": "en-gb", "gender": "Female", "desc": "British RP, clear"},
    {"id": "bf_isabella", "name": "Isabella", "lang": "en-gb", "gender": "Female", "desc": "British, expressive"},
    {"id": "bf_alice", "name": "Alice", "lang": "en-gb", "gender": "Female", "desc": "British, gentle"},
    {"id": "bf_lily", "name": "Lily", "lang": "en-gb", "gender": "Female", "desc": "British, soft"},
    # British English - Male
    {"id": "bm_george", "name": "George", "lang": "en-gb", "gender": "Male", "desc": "British, classic narrator"},
    {"id": "bm_fable", "name": "Fable", "lang": "en-gb", "gender": "Male", "desc": "British, storytelling"},
    {"id": "bm_lewis", "name": "Lewis", "lang": "en-gb", "gender": "Male", "desc": "British, relaxed"},
    {"id": "bm_daniel", "name": "Daniel", "lang": "en-gb", "gender": "Male", "desc": "British, formal"},
]


def record_audio(duration_s: float = 6.0, sample_rate: int = 16000, device=None) -> np.ndarray:
    """Records audio from microphone with a progress animation."""
    console.print(f"[cyan]Recording for {duration_s:.1f} seconds... Speak now![/cyan]")
    num_frames = int(duration_s * sample_rate)
    recording = sd.rec(num_frames, samplerate=sample_rate, channels=1, dtype="float32", device=device)

    # Progress bar in terminal
    steps = 20
    for i in range(steps):
        time.sleep(duration_s / steps)
        bar = "█" * (i + 1) + "░" * (steps - i - 1)
        pct = int(((i + 1) / steps) * 100)
        sys.stdout.write(f"\r  [{bar}] {pct}%")
        sys.stdout.flush()
    sd.wait()
    sys.stdout.write("\n")
    return recording.flatten()


def enroll_user_interactive(
    vm: VoiceprintManager,
    name: Optional[str] = None,
    duration_s: float = 6.0,
    device=None,
):
    """Enrolls an authorized human speaker voiceprint with silence notice."""
    if not name:
        console.print("\n[bold yellow]🎤 Speaker Enrollment[/bold yellow]")
        name = input("Enter speaker/user name (e.g. 'vicky'): ").strip()
        if not name:
            console.print("[yellow]Enrollment cancelled (no name entered).[/yellow]\n")
            return

    console.print(f"\n[bold yellow]🎤 Voice Enrollment for User:[/bold yellow] [bold green]'{name}'[/bold green]")

    # Silent environment notice
    notice = (
        "[bold yellow]🔇 ENVIRONMENT NOTICE: QUIET ENVIRONMENT REQUIRED[/bold yellow]\n\n"
        "• Please ensure you are in a [bold]quiet or silent environment[/bold] with minimal background noise.\n"
        "• Speak naturally in your normal conversational tone once recording starts.\n"
        "• A short sentence (e.g. 'Hello, this is my voice enrollment for Swar assistant') works best."
    )
    console.print(Panel(notice, border_style="yellow"))

    input("Press [Enter] when ready to start recording...")

    audio = record_audio(duration_s=duration_s, sample_rate=16000, device=device)

    # Energy check
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        console.print("[bold red]⚠️ Warning: Audio level was very quiet! Check microphone volume or input device.[/bold red]")

    console.print("[cyan]Extracting 192-dim speaker voiceprint...[/cyan]")
    emb = vm.enroll_user(name, audio, sample_rate=16000)
    vm.load_all_user_voiceprints()

    save_path = vm.voiceprints_dir / f"{name.lower().replace(' ', '_')}.npy"
    console.print(f"[bold green]✓ Successfully enrolled voiceprint for '{name}'![/bold green]")
    console.print(f"[dim]Saved to: {save_path} (L2-norm: {np.linalg.norm(emb):.4f})[/dim]\n")


def select_kokoro_voice_interactive() -> str:
    """Displays popular Kokoro voices in a table and asks user for their choice."""
    console.print("\n[bold cyan]Kokoro Available Voices:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=4)
    table.add_column("Voice ID", width=14)
    table.add_column("Name", width=12)
    table.add_column("Accent / Lang", width=14)
    table.add_column("Gender", width=8)
    table.add_column("Description")

    for idx, v in enumerate(POPULAR_KOKORO_VOICES, 1):
        is_def = " [dim](default)[/dim]" if v["id"] == "af_bella" else ""
        table.add_row(
            str(idx),
            f"[bold green]{v['id']}[/bold green]{is_def}",
            v["name"],
            v["lang"],
            v["gender"],
            v["desc"],
        )

    console.print(table)
    console.print("[dim]Select a number [1-25] from the table or type any Kokoro voice ID (e.g. af_bella, am_adam).[/dim]")
    choice = input("\nEnter voice number or ID (default: af_bella): ").strip()
    if not choice:
        return "af_bella"
    if choice.isdigit() and 1 <= int(choice) <= len(POPULAR_KOKORO_VOICES):
        return POPULAR_KOKORO_VOICES[int(choice) - 1]["id"]
    return choice.lower().strip()


def calibrate_kokoro_interactive(
    vm: VoiceprintManager,
    voice_name: Optional[str] = None,
    duration_s: float = 6.0,
):
    """
    Calibrates machine acoustic echo for a selected Kokoro voice:
    1. Selects voice from list if not passed.
    2. Prompts user to preview Kokoro sound.
    3. Prompts user for a silent room environment.
    4. Plays through laptop speakers while recording mic to capture acoustic profile.
    """
    if not voice_name:
        voice_name = select_kokoro_voice_interactive()

    clean_voice = voice_name.strip().lower().replace(" ", "_").replace("kokoro_", "")
    console.print(f"\n[bold yellow]🔊 Kokoro Voice Registration & Calibration:[/bold yellow] [bold green]'{clean_voice}'[/bold green]")

    from voice_pipeline.tts import KokoroTTS
    console.print(f"[cyan]Initializing Kokoro TTS engine for voice '{clean_voice}'...[/cyan]")
    try:
        tts = KokoroTTS(voice=clean_voice, device="cpu")
    except Exception as e:
        console.print(f"[bold red]Failed to load Kokoro voice '{clean_voice}': {e}[/bold red]\n")
        return

    # Step 1: Prompt user to play / preview Kokoro sound
    preview_prompt = (
        f"Press [bold green][Enter][/bold green] to play a sound preview of [bold]{clean_voice}[/bold] "
        "through your speakers (or type [bold]'s'[/bold] to skip preview): "
    )
    console.print(preview_prompt, end="")
    user_resp = input().strip().lower()

    if user_resp not in ("s", "skip"):
        preview_text = f"Hello! This is Kokoro speaking with the {clean_voice} voice. How does this sound on your machine?"
        console.print(f"[dim]Synthesizing preview audio for '{clean_voice}'...[/dim]")
        prev_audio = tts.synthesize_chunk(preview_text, voice=clean_voice, speed=1.0)
        if prev_audio is not None:
            console.print("[cyan]Playing preview audio through speakers...[/cyan]")
            sd.play(prev_audio, samplerate=24000)
            sd.wait()
            console.print("[green]Preview finished.[/green]\n")

        proceed = input(f"Proceed with acoustic calibration for '{clean_voice}'? [Y/n]: ").strip().lower()
        if proceed in ("n", "no"):
            console.print("[yellow]Calibration cancelled.[/yellow]\n")
            return

    # Step 2: Environment Silence Notice
    notice = (
        "[bold yellow]🔇 ENVIRONMENT NOTICE: SILENT ROOM REQUIRED FOR ACOUSTIC CALIBRATION[/bold yellow]\n\n"
        "• [bold red]Unplug earphones/headphones[/bold red] so Kokoro audio plays directly from machine speakers.\n"
        "• Set laptop volume to your normal conversational listening level.\n"
        "• Please ensure the room is [bold]completely silent[/bold] (no talking, background music, or TV).\n"
        "• The system will play audio through your speakers and record the microphone simultaneously."
    )
    console.print(Panel(notice, border_style="yellow"))

    input("Press [Enter] when the room is silent to start acoustic calibration...")

    calib_text = (
        f"Hello, this is the Kokoro audio calibration for voice {clean_voice}. "
        "Recording machine speaker acoustics to eliminate feedback and self interruption."
    )
    console.print("[cyan]Synthesizing reference calibration speech...[/cyan]")
    audio_24k = tts.synthesize_chunk(calib_text, voice=clean_voice, speed=1.0)
    if audio_24k is None:
        console.print("[bold red]Failed to synthesize Kokoro audio![/bold red]\n")
        return

    # Save digital baseline voiceprint
    digital_emb = vm.extractor.extract(audio_24k, sample_rate=24000)
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    digital_path = DEFAULT_MODEL_DIR / f"kokoro_{clean_voice}_digital.npy"
    np.save(digital_path, digital_emb)

    # Play through speakers and record through mic simultaneously
    console.print("[cyan]Playing through machine speakers & recording microphone acoustics...[/cyan]")
    duration_audio = len(audio_24k) / 24000.0
    rec_frames = int((duration_audio + 0.5) * 16000)

    mic_recording = sd.rec(rec_frames, samplerate=16000, channels=1, dtype="float32")
    sd.play(audio_24k, samplerate=24000)
    sd.wait()

    mic_audio = mic_recording.flatten()
    console.print("[cyan]Extracting acoustic speaker voiceprint...[/cyan]")
    acoustic_emb = vm.extractor.extract(mic_audio, sample_rate=16000)

    acoustic_path = DEFAULT_MODEL_DIR / f"kokoro_{clean_voice}.npy"
    np.save(acoustic_path, acoustic_emb)

    # Refresh voiceprints in manager
    vm.kokoro_voiceprints[clean_voice] = acoustic_emb
    vm.kokoro_voiceprints[f"{clean_voice}_digital"] = digital_emb
    vm.load_all_kokoro_voiceprints()

    sim = float(np.dot(digital_emb, acoustic_emb))
    console.print(f"\n[bold green]✓ Successfully calibrated acoustic profile for Kokoro voice '{clean_voice}'![/bold green]")
    console.print(f"[dim]Acoustic file: {acoustic_path}[/dim]")
    console.print(f"[dim]Digital vs Acoustic similarity: {sim:.3f}[/dim]\n")


def delete_user_interactive(vm: VoiceprintManager, name: Optional[str] = None):
    """Interactively or directly deletes an enrolled speaker voiceprint."""
    users = vm.list_users()
    if not users:
        console.print("[yellow]No enrolled user voiceprints found to delete.[/yellow]\n")
        return

    if name:
        target = name.strip().lower().replace(" ", "_")
    else:
        console.print("\n[bold yellow]Select a Speaker Voiceprint to Delete:[/bold yellow]")
        for idx, u in enumerate(sorted(users), 1):
            p = vm.voiceprints_dir / f"{u}.npy"
            sz = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "N/A"
            console.print(f"  [{idx}] [bold green]{u}[/bold green] [dim]({sz})[/dim]")
        console.print("  [c] Cancel")
        choice = input("\nEnter number or user name to delete: ").strip()
        if choice.lower() in ("c", "cancel", ""):
            console.print("[dim]Cancelled.[/dim]\n")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(users):
            target = sorted(users)[int(choice) - 1]
        else:
            target = choice.lower().replace(" ", "_")

    if vm.delete_user(target):
        vm.load_all_user_voiceprints()
        console.print(f"[bold green]✓ Successfully deleted voiceprint for speaker '{target}'.[/bold green]\n")
    else:
        console.print(f"[bold red]User voiceprint '{target}' not found.[/bold red]\n")


def delete_kokoro_interactive(vm: VoiceprintManager, voice_name: Optional[str] = None):
    """Interactively or directly deletes an enrolled Kokoro voiceprint."""
    voices = vm.list_kokoro_voices()
    if not voices:
        console.print("[yellow]No enrolled or cached Kokoro voiceprints found to delete.[/yellow]\n")
        return

    if voice_name:
        target = voice_name.strip().lower().replace(" ", "_").replace("kokoro_", "")
    else:
        console.print("\n[bold yellow]Select a Kokoro Voiceprint to Delete:[/bold yellow]")
        for idx, v in enumerate(sorted(voices), 1):
            ac_file = DEFAULT_MODEL_DIR / f"kokoro_{v}.npy"
            dig_file = DEFAULT_MODEL_DIR / f"kokoro_{v}_digital.npy"
            status = "Acoustic Calibrated" if (ac_file.exists() and dig_file.exists()) else "Reference Voiceprint"
            console.print(f"  [{idx}] [bold cyan]{v}[/bold cyan] [dim]({status})[/dim]")
        console.print("  [c] Cancel")
        choice = input("\nEnter number or voice name to delete: ").strip()
        if choice.lower() in ("c", "cancel", ""):
            console.print("[dim]Cancelled.[/dim]\n")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(voices):
            target = sorted(voices)[int(choice) - 1]
        else:
            target = choice.lower().replace(" ", "_").replace("kokoro_", "")

    if vm.delete_kokoro_voiceprint(target):
        vm.load_all_kokoro_voiceprints()
        console.print(f"[bold green]✓ Successfully deleted Kokoro voiceprint for '{target}'.[/bold green]\n")
    else:
        console.print(f"[bold red]Kokoro voiceprint '{target}' not found.[/bold red]\n")


def list_voiceprints(vm: VoiceprintManager):
    """Displays a formatted table of all registered user and Kokoro voiceprints."""
    console.print("\n[bold cyan]🎙️ Registered Voiceprints:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", width=12)
    table.add_column("Identifier", width=22)
    table.add_column("Profile Type", width=24)
    table.add_column("Status / Details", width=24)
    table.add_column("File Location")

    # 1. Enrolled Users
    users = vm.list_users()
    if users:
        for u in sorted(users):
            p = vm.voiceprints_dir / f"{u}.npy"
            sz = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "N/A"
            table.add_row(
                "Speaker",
                f"[bold green]{u}[/bold green]",
                "Physical Mic Voiceprint",
                f"[green]Enrolled[/green] ({sz})",
                str(p),
            )
    else:
        table.add_row("Speaker", "[dim](None enrolled)[/dim]", "Physical Mic Voiceprint", "[dim]No voiceprints[/dim]", "")

    # 2. Enrolled Kokoro Voices
    kokoro_voices = vm.list_kokoro_voices()
    if kokoro_voices:
        for v in sorted(kokoro_voices):
            ac_file = DEFAULT_MODEL_DIR / f"kokoro_{v}.npy"
            dig_file = DEFAULT_MODEL_DIR / f"kokoro_{v}_digital.npy"
            if ac_file.exists() and dig_file.exists():
                ptype = "[bold yellow]Acoustic Calibrated[/bold yellow]"
                stat = "[green]Physical Speakers Profile[/green]"
            elif ac_file.exists():
                ptype = "Active Voiceprint"
                stat = "[green]Calibrated / Reference[/green]"
            else:
                ptype = "Digital Synthesis"
                stat = "[cyan]Reference Only[/cyan]"
            table.add_row(
                "Kokoro TTS",
                f"[bold cyan]{v}[/bold cyan]",
                ptype,
                stat,
                str(ac_file if ac_file.exists() else dig_file),
            )
    else:
        table.add_row("Kokoro TTS", "[dim](None cached)[/dim]", "Acoustic / Reference", "[dim]Will auto-generate on start[/dim]", "")

    console.print(table)
    console.print()


def test_live_recognition(vm: VoiceprintManager, duration_s: float = 4.0, device=None):
    """Runs a live microphone test to verify speaker identification and Kokoro echo detection."""
    console.print("\n[bold yellow]🎙️ Live Speaker Recognition Test[/bold yellow]")
    console.print(f"Speak for {duration_s:.1f} seconds. The system will identify the speaker...")
    input("Press [Enter] to start...")
    audio = record_audio(duration_s=duration_s, sample_rate=16000, device=device)

    console.print("\n[bold cyan]Analysis Results:[/bold cyan]")
    # 1. Check users
    is_user, user_name, user_score = vm.matches_authorized_user(audio, threshold=0.65)
    if is_user and user_name:
        console.print(f"  • [bold green]User Match:[/bold green] [bold]{user_name}[/bold] (Similarity: {user_score:.3f} >= 0.65)")
    else:
        console.print(f"  • [yellow]User Match:[/yellow] None (Best score: {user_score:.3f} < 0.65)")

    # 2. Check Kokoro
    is_kokoro, kokoro_score, matched_voice = vm.is_kokoro_echo(audio, threshold=0.60)
    if is_kokoro:
        console.print(f"  • [bold red]Kokoro Echo:[/bold red] Detected as Kokoro '{matched_voice}' ({kokoro_score:.3f} >= 0.60)")
    else:
        console.print(f"  • [dim green]Kokoro Echo:[/dim green] Not Kokoro (Best score: {kokoro_score:.3f} < 0.60)")
    console.print()


def interactive_studio_menu(vm: VoiceprintManager):
    """Interactive CLI menu for Swar Voiceprint Studio."""
    while True:
        console.print("\n" + "═" * 60)
        console.print("[bold cyan]         🎙️  Swar Voiceprint & Acoustic Studio[/bold cyan]")
        console.print("═" * 60)
        console.print("  [bold green]1.[/bold green] Enroll Speaker Voice (Authorized User)")
        console.print("  [bold green]2.[/bold green] Enroll / Calibrate Kokoro Voice (Machine Acoustic Echo)")
        console.print("  [bold green]3.[/bold green] List All Enrolled Voiceprints (Users & Kokoro)")
        console.print("  [bold green]4.[/bold green] Delete a Speaker Voiceprint")
        console.print("  [bold green]5.[/bold green] Delete a Kokoro Voiceprint")
        console.print("  [bold green]6.[/bold green] Test Live Speaker Recognition (Mic Verification)")
        console.print("  [bold green]7.[/bold green] Exit")
        console.print("═" * 60)

        choice = input("Enter your choice [1-7]: ").strip()
        if choice == "1":
            enroll_user_interactive(vm)
        elif choice == "2":
            calibrate_kokoro_interactive(vm)
        elif choice == "3":
            list_voiceprints(vm)
        elif choice == "4":
            delete_user_interactive(vm)
        elif choice == "5":
            delete_kokoro_interactive(vm)
        elif choice == "6":
            test_live_recognition(vm)
        elif choice in ("7", "q", "exit", "quit"):
            console.print("[dim]Exiting Voiceprint Studio. Goodbye![/dim]\n")
            break
        else:
            console.print("[yellow]Invalid option. Please enter 1-7.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Swar Voiceprint Enrollment & Acoustic Calibration Studio")
    parser.add_argument("--user", type=str, nargs="?", const="", help="Enroll an authorized speaker voiceprint (optionally provide name)")
    parser.add_argument("--kokoro", action="store_true", help="Calibrate Kokoro TTS acoustic voiceprint through machine speakers")
    parser.add_argument("--voice", type=str, default=None, help="Kokoro voice name for calibration (e.g. af_bella, am_adam)")
    parser.add_argument("--list", action="store_true", help="List all enrolled voiceprints (speakers & Kokoro)")
    parser.add_argument("--delete", type=str, nargs="?", const="", help="Delete an enrolled user voiceprint")
    parser.add_argument("--delete-user", type=str, nargs="?", const="", help="Delete an enrolled user voiceprint")
    parser.add_argument("--delete-kokoro", type=str, nargs="?", const="", help="Delete an enrolled Kokoro voiceprint")
    parser.add_argument("--test", action="store_true", help="Run live microphone test to verify speaker identification")
    parser.add_argument("--duration", type=float, default=6.0, help="Recording duration in seconds (default: 6.0)")
    parser.add_argument("--menu", action="store_true", help="Launch interactive studio menu")

    args = parser.parse_args()

    config = PipelineConfig.load_default_or_file()
    guard_cfg = getattr(config, "barge_in_guard", None)
    vp_dir = getattr(guard_cfg, "voiceprints_dir", DEFAULT_VOICEPRINTS_DIR) if guard_cfg else DEFAULT_VOICEPRINTS_DIR

    vm = VoiceprintManager(voiceprints_dir=vp_dir)

    # Check CLI actions
    if args.user is not None:
        user_name = args.user if args.user else None
        enroll_user_interactive(vm, name=user_name, duration_s=args.duration)
    elif args.kokoro:
        calibrate_kokoro_interactive(vm, voice_name=args.voice, duration_s=args.duration)
    elif args.delete_user is not None or args.delete is not None:
        target = args.delete_user if args.delete_user is not None else args.delete
        delete_user_interactive(vm, name=target if target else None)
    elif args.delete_kokoro is not None:
        target = args.delete_kokoro if args.delete_kokoro else None
        delete_kokoro_interactive(vm, voice_name=target)
    elif args.test:
        test_live_recognition(vm, duration_s=args.duration)
    elif args.list:
        list_voiceprints(vm)
    elif args.menu or len(sys.argv) == 1:
        interactive_studio_menu(vm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
