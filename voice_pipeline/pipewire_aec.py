"""
PipeWire WebRTC Acoustic Echo Cancellation (AEC) Manager for Swar.
Exclusively for Linux systems running PipeWire (e.g. Arch Linux, Fedora, Ubuntu 24.04+).

Features:
1. Detects PipeWire and PulseAudio server status via pactl.
2. Programmatically loads and configures module-echo-cancel (WebRTC AEC3) in PipeWire.
3. Automatically sets default virtual source and sink for zero-overhead cancellation.
4. Restores system audio defaults and unloads module gracefully on pipeline exit.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PipeWireAECManager:
    """
    Manages PipeWire's module-echo-cancel (WebRTC AEC3) for hardware/driver-level
    echo cancellation on Linux.
    """

    def __init__(
        self,
        auto_load: bool = True,
        source_name: str = "echo-cancel-source",
        sink_name: str = "echo-cancel-sink",
        restore_defaults: bool = True,
    ):
        self.auto_load = auto_load
        self.source_name = source_name
        self.sink_name = sink_name
        self.restore_defaults = restore_defaults

        self.loaded_module_id: Optional[str] = None
        self.original_source: Optional[str] = None
        self.original_sink: Optional[str] = None
        self.is_active: bool = False

        # Register exit handler for clean recovery
        atexit.register(self.stop)

    @staticmethod
    def is_pipewire_supported() -> Tuple[bool, str]:
        """Checks if the system is Linux and running PipeWire with PulseAudio emulation."""
        if not sys.platform.startswith("linux"):
            return False, "PipeWire AEC is only supported on Linux."

        if not shutil.which("pactl"):
            return False, "'pactl' utility not found. Please install libpulse / pipewire-pulse."

        try:
            out = subprocess.check_output(
                ["pactl", "info"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="ignore")
            if "PipeWire" in out or "pulse" in out.lower():
                return True, "PipeWire detected."
            return False, "PulseAudio server detected, but PipeWire not confirmed."
        except Exception as e:
            return False, f"Error querying pactl info: {e}"

    def is_module_loaded(self) -> bool:
        """Checks if module-echo-cancel or echo-cancel-source already exists."""
        if not shutil.which("pactl"):
            return False
        try:
            out = subprocess.check_output(
                ["pactl", "list", "sources", "short"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="ignore")
            return self.source_name in out
        except Exception:
            return False

    def start(self) -> bool:
        """
        Activates PipeWire AEC:
        1. Checks support.
        2. Records current default source and sink.
        3. Loads module-echo-cancel if needed.
        4. Sets echo-cancel-source and echo-cancel-sink as defaults.
        5. Sets PULSE_SOURCE and PULSE_SINK environment variables.
        """
        supported, reason = self.is_pipewire_supported()
        if not supported:
            logger.warning(f"Cannot enable PipeWire AEC: {reason}")
            return False

        try:
            # 1. Save original defaults
            try:
                self.original_source = subprocess.check_output(
                    ["pactl", "get-default-source"],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                ).decode("utf-8", errors="ignore").strip()
            except Exception:
                self.original_source = None

            try:
                self.original_sink = subprocess.check_output(
                    ["pactl", "get-default-sink"],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                ).decode("utf-8", errors="ignore").strip()
            except Exception:
                self.original_sink = None

            # 2. Check if already loaded
            if not self.is_module_loaded():
                if self.auto_load:
                    logger.info("Loading PipeWire module-echo-cancel (WebRTC AEC3)...")
                    mod_out = subprocess.check_output(
                        ["pactl", "load-module", "module-echo-cancel"],
                        stderr=subprocess.PIPE,
                        timeout=5,
                    ).decode("utf-8", errors="ignore").strip()
                    self.loaded_module_id = mod_out
                    logger.info(f"PipeWire module-echo-cancel loaded with ID {self.loaded_module_id}.")
                else:
                    logger.warning("module-echo-cancel is not loaded and auto_load is False.")
                    return False
            else:
                logger.info(f"PipeWire echo-cancel source '{self.source_name}' is already present.")

            # 3. Set default source and sink to echo-cancel
            subprocess.check_call(
                ["pactl", "set-default-source", self.source_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            subprocess.check_call(
                ["pactl", "set-default-sink", self.sink_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )

            # 4. Set process environment variables for sounddevice/PortAudio
            os.environ["PULSE_SOURCE"] = self.source_name
            os.environ["PULSE_SINK"] = self.sink_name

            self.is_active = True
            logger.info(
                f"PipeWire AEC active: source='{self.source_name}', sink='{self.sink_name}'"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start PipeWire AEC: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        """Restores audio configuration and unloads module if Swar loaded it."""
        if not self.is_active and not self.loaded_module_id:
            return

        logger.info("Restoring audio devices and shutting down PipeWire AEC...")

        # 1. Restore defaults
        if self.restore_defaults:
            if self.original_source:
                try:
                    subprocess.call(
                        ["pactl", "set-default-source", self.original_source],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                    )
                except Exception:
                    pass
            if self.original_sink:
                try:
                    subprocess.call(
                        ["pactl", "set-default-sink", self.original_sink],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                    )
                except Exception:
                    pass

        # 2. Unload module if we loaded it
        if self.loaded_module_id:
            try:
                subprocess.call(
                    ["pactl", "unload-module", self.loaded_module_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                logger.info(f"Unloaded PipeWire module-echo-cancel ({self.loaded_module_id}).")
            except Exception as e:
                logger.warning(f"Could not unload module-echo-cancel: {e}")
            self.loaded_module_id = None

        # 3. Clean environment
        os.environ.pop("PULSE_SOURCE", None)
        os.environ.pop("PULSE_SINK", None)

        self.is_active = False
