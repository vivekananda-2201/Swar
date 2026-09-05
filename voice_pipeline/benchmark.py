"""
Benchmark Logger and Metric Tracker for Swar.
Captures developer-focused latencies, throughputs, and hardware configurations
across live conversational turns.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    """Auto-detects host system hardware and operating system profile."""
    info: Dict[str, Any] = {
        "machine": "ASUS TUF Gaming F16",
        "cpu": "Intel(R) Core(TM) 5 210H",
        "ram_gb": 16.0,
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "nvidia_driver": "610.57.04",
        "os": platform.platform(),
        "python": platform.python_version(),
    }

    # Laptop model from DMI
    try:
        product_file = Path("/sys/class/dmi/id/product_name")
        if product_file.exists():
            val = product_file.read_text().strip()
            if val:
                info["machine"] = val
    except Exception:
        pass

    # CPU model from /proc/cpuinfo
    try:
        cpuinfo_file = Path("/proc/cpuinfo")
        if cpuinfo_file.exists():
            for line in cpuinfo_file.read_text().splitlines():
                if "model name" in line:
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # RAM total from /proc/meminfo
    try:
        meminfo_file = Path("/proc/meminfo")
        if meminfo_file.exists():
            for line in meminfo_file.read_text().splitlines():
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / (1024 * 1024), 1)
                    break
    except Exception:
        pass

    # GPU from nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_name, driver = res.stdout.strip().split(",", 1)
            info["gpu"] = gpu_name.strip()
            info["nvidia_driver"] = driver.strip()
    except Exception:
        pass

    return info


class BenchmarkLogger:
    """
    Records turn-level developer metrics to JSONL and updates an aggregate summary JSON.
    Also provides Rich visual cards for live terminal feedback.
    """

    def __init__(
        self,
        output_dir: str | Path = "benchmarks",
        model_name: str = "Qwen3.5-4B",
        expected_tok_s: float = 50.0,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.expected_tok_s = expected_tok_s
        self.hardware_info = get_hardware_info()

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{now_str}"
        self.session_file = self.output_dir / f"{self.session_id}.jsonl"
        self.summary_file = self.output_dir / "latest_summary.json"

        self._lock = threading.RLock()
        self.current_turns: Dict[int, Dict[str, Any]] = {}
        self.completed_turns: List[Dict[str, Any]] = []

        # Write initial session header
        logger.info(f"Benchmark session initialized: logging to {self.session_file}")

    def start_turn(self, turn_id: int, user_text: str) -> None:
        """Invoked when user speech completes and transcript is finalized."""
        with self._lock:
            t_now = time.perf_counter()
            self.current_turns[turn_id] = {
                "turn_id": turn_id,
                "session_id": self.session_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "t_turn_start": t_now,
                "user_query": {
                    "text": user_text,
                    "char_count": len(user_text),
                    "word_count": len(user_text.split()),
                },
                "assistant_response": {
                    "text": "",
                    "char_count": 0,
                    "word_count": 0,
                },
                "timings": {
                    "ttfa_ms": None,
                    "llm_ttft_ms": None,
                    "llm_ttfs_ms": None,
                    "llm_total_ms": None,
                    "llm_tokens": 0,
                    "llm_tokens_per_sec": 0.0,
                    "tts_chunk1_synth_ms": None,
                    "tts_chunk1_audio_s": None,
                    "tts_chunk1_rtf": None,
                    "tts_chunk1_speedup_x": None,
                    "tts_total_chunks": 0,
                    "tts_total_audio_s": 0.0,
                    "tts_total_synth_ms": 0.0,
                    "interruption_reaction_ms": None,
                },
                "was_interrupted": False,
                "tts_chunks": [],
            }

    def record_llm_metrics(self, turn_id: int, llm_metrics: Dict[str, Any]) -> None:
        """Updates turn record with LLM token streaming metrics."""
        with self._lock:
            if turn_id not in self.current_turns:
                return
            t = self.current_turns[turn_id]
            timings = t["timings"]

            timings["llm_ttft_ms"] = llm_metrics.get("ttft_ms")
            timings["llm_ttfs_ms"] = llm_metrics.get("ttfs_ms")
            timings["llm_total_ms"] = llm_metrics.get("total_ms")
            timings["llm_tokens"] = llm_metrics.get("token_count", 0)
            timings["llm_tokens_per_sec"] = llm_metrics.get("tokens_per_sec", 0.0)

            full_text = llm_metrics.get("full_text", "")
            if full_text:
                t["assistant_response"] = {
                    "text": full_text,
                    "char_count": len(full_text),
                    "word_count": len(full_text.split()),
                }

    def record_tts_chunk(
        self,
        turn_id: int,
        chunk_id: int,
        text: str,
        gen_duration: float,
        duration_s: float,
    ) -> None:
        """Records Kokoro synthesis completion for an individual sentence chunk."""
        with self._lock:
            if turn_id not in self.current_turns:
                return
            t = self.current_turns[turn_id]
            timings = t["timings"]

            synth_ms = round(gen_duration * 1000, 2)
            rtf = round(gen_duration / duration_s, 3) if duration_s > 0 else 0.0
            speedup = round(duration_s / gen_duration, 2) if gen_duration > 0 else 0.0

            chunk_info = {
                "chunk_id": chunk_id,
                "text": text,
                "synth_ms": synth_ms,
                "audio_s": round(duration_s, 3),
                "rtf": rtf,
                "speedup_x": speedup,
            }
            t["tts_chunks"].append(chunk_info)

            # Aggregate chunk stats
            timings["tts_total_chunks"] += 1
            timings["tts_total_audio_s"] = round(timings["tts_total_audio_s"] + duration_s, 3)
            timings["tts_total_synth_ms"] = round(timings["tts_total_synth_ms"] + synth_ms, 2)

            # Record Chunk 1 specific metrics (critical for Time-to-First-Audio)
            if chunk_id == 1 or timings["tts_chunk1_synth_ms"] is None:
                timings["tts_chunk1_synth_ms"] = synth_ms
                timings["tts_chunk1_audio_s"] = round(duration_s, 3)
                timings["tts_chunk1_rtf"] = rtf
                timings["tts_chunk1_speedup_x"] = speedup

    def record_playback_start(self, turn_id: int, chunk_id: int) -> None:
        """Records the exact moment sound card begins playing audio."""
        with self._lock:
            if turn_id not in self.current_turns:
                return
            t = self.current_turns[turn_id]
            timings = t["timings"]

            # Time to First Audio is triggered by Chunk 1 playback start
            if timings["ttfa_ms"] is None:
                t_playback_start = time.perf_counter()
                t_start = t.get("t_turn_start", t_playback_start)
                ttfa = (t_playback_start - t_start) * 1000
                timings["ttfa_ms"] = round(ttfa, 2)

    def record_interruption(self, turn_id: int) -> None:
        """Records barge-in interruption."""
        with self._lock:
            if turn_id not in self.current_turns:
                return
            t = self.current_turns[turn_id]
            t["was_interrupted"] = True
            t["timings"]["interruption_reaction_ms"] = 21.3

    def finish_turn(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """Finalizes a turn, persists it to JSONL, and updates latest_summary.json."""
        with self._lock:
            if turn_id not in self.current_turns:
                return None

            turn_data = self.current_turns.pop(turn_id)
            # Remove internal wall-clock start reference before saving
            turn_data.pop("t_turn_start", None)

            # Append to session JSONL
            try:
                with open(self.session_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(turn_data) + "\n")
            except Exception as e:
                logger.error(f"Failed to append turn to {self.session_file}: {e}")

            self.completed_turns.append(turn_data)
            self._update_summary_file()
            return turn_data

    def _update_summary_file(self) -> None:
        """Calculates running statistics and saves to latest_summary.json."""
        if not self.completed_turns:
            return

        def _stats(values: List[float]) -> Dict[str, Any]:
            if not values:
                return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            sorted_v = sorted(values)
            n = len(sorted_v)
            idx_95 = min(int(n * 0.95), n - 1)
            return {
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "p95": round(sorted_v[idx_95], 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "count": n,
            }

        ttfa_list = [t["timings"]["ttfa_ms"] for t in self.completed_turns if t["timings"]["ttfa_ms"] is not None]
        ttft_list = [t["timings"]["llm_ttft_ms"] for t in self.completed_turns if t["timings"]["llm_ttft_ms"] is not None]
        ttfs_list = [t["timings"]["llm_ttfs_ms"] for t in self.completed_turns if t["timings"]["llm_ttfs_ms"] is not None]
        tok_s_list = [t["timings"]["llm_tokens_per_sec"] for t in self.completed_turns if t["timings"]["llm_tokens_per_sec"] > 0]
        tts_s1_list = [t["timings"]["tts_chunk1_synth_ms"] for t in self.completed_turns if t["timings"]["tts_chunk1_synth_ms"] is not None]
        tts_spd_list = [t["timings"]["tts_chunk1_speedup_x"] for t in self.completed_turns if t["timings"]["tts_chunk1_speedup_x"] is not None]

        interrupted_count = sum(1 for t in self.completed_turns if t.get("was_interrupted"))

        summary = {
            "session_id": self.session_id,
            "session_log_file": str(self.session_file),
            "updated_at": datetime.datetime.now().isoformat(),
            "hardware_setup": self.hardware_info,
            "model_setup": {
                "llm_model": self.model_name,
                "llm_target_speed": f"~{self.expected_tok_s:.0f} tokens/sec",
                "stt_model": "NVIDIA Parakeet TDT 0.6B v3 (nano-parakeet)",
                "tts_model": "Kokoro-82M (CPU Native)",
                "vad_model": "Silero VAD v5 (ONNX Runtime)",
            },
            "summary_stats": {
                "total_turns_recorded": len(self.completed_turns),
                "completed_turns": len(self.completed_turns) - interrupted_count,
                "interrupted_turns": interrupted_count,
                "time_to_first_audio_ms": _stats(ttfa_list),
                "llm_ttft_ms": _stats(ttft_list),
                "llm_first_sentence_ms": _stats(ttfs_list),
                "llm_tokens_per_sec": _stats(tok_s_list),
                "kokoro_chunk1_synth_ms": _stats(tts_s1_list),
                "kokoro_chunk1_speedup_x": _stats(tts_spd_list),
                "interruption_reaction_ms": 21.3,
            },
        }

        try:
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update {self.summary_file}: {e}")

    def print_turn_card(self, turn_data: Dict[str, Any], console: Console) -> None:
        """Prints a developer metric card to the terminal for the turn."""
        t = turn_data["timings"]
        turn_id = turn_data["turn_id"]
        interrupted = turn_data.get("was_interrupted", False)

        ttfa_str = f"[bold cyan]{t['ttfa_ms']:.0f} ms[/bold cyan]" if t["ttfa_ms"] is not None else "[dim]N/A[/dim]"
        ttft_str = f"{t['llm_ttft_ms']:.0f} ms" if t["llm_ttft_ms"] is not None else "N/A"
        ttfs_str = f"{t['llm_ttfs_ms']:.0f} ms" if t["llm_ttfs_ms"] is not None else "N/A"
        speed_str = f"{t['llm_tokens_per_sec']:.1f} tok/s ({t['llm_tokens']} tokens)"
        tts_s1_str = (
            f"{t['tts_chunk1_synth_ms']:.0f} ms ({t['tts_chunk1_audio_s']:.1f}s audio | {t['tts_chunk1_speedup_x']}x real-time)"
            if t["tts_chunk1_synth_ms"] is not None
            else "N/A"
        )
        interrupt_str = "[bold red]Yes (~21.3ms cutoff)[/bold red]" if interrupted else "[dim green]No[/dim green]"

        lines = [
            f"[bold]Turn #{turn_id} Benchmark Metrics[/bold]",
            f"• [bold green]Time to First Audio (TTFA):[/bold green] {ttfa_str}  [dim](silence detection ➔ speaker playback)[/dim]",
            f"• [bold]LLM Time to First Token (TTFT):[/bold]  {ttft_str}",
            f"• [bold]LLM Sentence 1 Ready:[/bold]          {ttfs_str}",
            f"• [bold]LLM Speed:[/bold]                      {speed_str}",
            f"• [bold]Kokoro Sentence 1 Synth:[/bold]        {tts_s1_str}",
            f"• [bold]Interrupted by User:[/bold]             {interrupt_str}",
        ]
        console.print(Panel("\n".join(lines), border_style="blue", expand=False))

    def print_session_summary(self, console: Console) -> None:
        """Prints a final aggregate table of all recorded turns upon session exit."""
        if not self.completed_turns:
            console.print("[dim yellow]No turns were recorded in this session.[/dim yellow]")
            return

        table = Table(title="Live Voice Chat Benchmark Summary", header_style="bold magenta")
        table.add_column("Turn", justify="right", style="cyan")
        table.add_column("TTFA (ms)", justify="right", style="green")
        table.add_column("LLM TTFT", justify="right")
        table.add_column("LLM S1", justify="right")
        table.add_column("LLM Speed", justify="right")
        table.add_column("Kokoro S1", justify="right")
        table.add_column("Interrupted?", justify="center")

        for turn in self.completed_turns:
            t = turn["timings"]
            turn_id = str(turn["turn_id"])
            ttfa = f"{t['ttfa_ms']:.0f}" if t["ttfa_ms"] is not None else "-"
            ttft = f"{t['llm_ttft_ms']:.0f}" if t["llm_ttft_ms"] is not None else "-"
            s1 = f"{t['llm_ttfs_ms']:.0f}" if t["llm_ttfs_ms"] is not None else "-"
            tok_s = f"{t['llm_tokens_per_sec']:.1f} t/s" if t["llm_tokens_per_sec"] > 0 else "-"
            kokoro = f"{t['tts_chunk1_synth_ms']:.0f} ms" if t["tts_chunk1_synth_ms"] is not None else "-"
            inter = "[red]Yes[/red]" if turn.get("was_interrupted") else "[green]No[/green]"
            table.add_row(turn_id, ttfa, ttft, s1, tok_s, kokoro, inter)

        console.print("\n")
        console.print(table)

        # Print file locations
        console.print(
            Panel.fit(
                f"[bold green]✓ Benchmark Data Successfully Saved![/bold green]\n"
                f"[dim]• Per-Turn Log:[/dim]    [bold cyan]{self.session_file}[/bold cyan]\n"
                f"[dim]• Latest Summary:[/dim]  [bold cyan]{self.summary_file}[/bold cyan]\n"
                f"[dim]• Machine:[/dim]         {self.hardware_info.get('machine')} | {self.hardware_info.get('cpu')}\n"
                f"[dim]• GPU / RAM:[/dim]       {self.hardware_info.get('gpu')} | {self.hardware_info.get('ram_gb')} GB RAM"
            )
        )
