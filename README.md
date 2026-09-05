# Swar (स्वर) — High-Performance Real-Time Voice Pipeline

**100% Offline, CPU-Native Cascaded Voice Pipeline**  
*Silero VAD + NVIDIA Parakeet TDT STT + Kokoro-82M TTS*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: Linux / macOS / Windows](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![Hardware: CPU Native (GPU Supported)](https://img.shields.io/badge/hardware-CPU%20Native%20(GPU%20Supported)-orange.svg)]()

---

## 🎙️ What is Swar?

**Swar (स्वर)** is a fully open-source, ultra-low-latency voice pipeline designed from the ground up to run **100% locally and natively on consumer CPUs**, while also providing out-of-the-box GPU acceleration. 

### Why CPU-First?
Running fast voice pipelines on high-end GPUs is straightforward. The true engineering feat is achieving sub-second, fluid, conversational voice interactions on standard multi-core laptop and server **CPUs** without burning thermal budgets, requiring external cloud APIs, or compromising synthesis fidelity. 

Swar delivers:
- **Instantaneous Voice Activity Detection (VAD)** via Silero VAD (< 1% CPU utilization).
- **Live Progressive Speech-to-Text (STT)** via NVIDIA Parakeet TDT 0.6B running locally on CPU.
- **Natural 24kHz Text-to-Speech (TTS)** via Kokoro-82M with style-based acoustic synthesis.
- **Barge-In (Interruption Handling)** with zero ALSA/sound card driver crashes and zero lost user words.
- **Compute-Ahead Decoupled TTS**: Synthesizes upcoming sentences in the background while earlier sentences are actively playing through the speakers.
- **Multi-Phrase Wake Word & Wake Sentence Detection**: Case-insensitive, sentence-wide detection that preserves the complete user transcript.
- **Thinking LLM Monologue Suppression**: Real-time filtering of `<think>...</think>` internal reasoning blocks from modern reasoning models (DeepSeek R1, Qwen 2.5, etc.).

---

## 📐 Architecture Overview

```
                          ┌───────────────────────────┐
                          │     Microphone Input      │
                          │     (16kHz 16-bit PCM)    │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Silero VAD (v5 ONNX)    │
                          │   Speech Start / Silence  │
                          └─────────────┬─────────────┘
                                        │
                         Speech Event   │   Audio Frames
                                        ▼
                          ┌───────────────────────────┐
                          │  NVIDIA Parakeet TDT 0.6B │
                          │  Smart Progressive STT    │
                          └─────────────┬─────────────┘
                                        │
                       Final Transcript │ (e.g. "Hello Relic, can you hear me?")
                                        ▼
                          ┌───────────────────────────┐
                          │  Stateful Wake Engine     │
                          │  (STANDBY vs ACTIVE Mode) │
                          └─────────────┬─────────────┘
                                        │
                      Verified Query    │ Complete Original Transcription
                                        ▼
                          ┌───────────────────────────┐
                          │   Any LLM / Agent Brain   │
                          │   (Local Ollama/vLLM/API) │
                          └─────────────┬─────────────┘
                                        │
                       Token Stream     │ (Real-time sentence chunking)
                                        ▼
                          ┌───────────────────────────┐
                          │   Decoupled Kokoro TTS    │
                          │ ┌───────────────────────┐ │
                          │ │ Generation Worker     │ │ <── Computes ahead in queue
                          │ └───────────┬───────────┘ │
                          │             ▼             │
                          │ ┌───────────────────────┐ │
                          │ │ Playback Worker       │ │ <── Streams to speaker
                          │ └───────────────────────┘ │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │      Speaker Audio        │
                          │        (24kHz PCM)        │
                          └───────────────────────────┘
```

---

## ⚡ Key Highlights

| Feature | Swar Implementation | Traditional Pipelines |
| :--- | :--- | :--- |
| **Compute Hardware** | **Optimized for CPU** (AVX2/NEON vectorization); GPU optional | Requires dedicated 8GB+ VRAM GPU |
| **STT Latency** | Emits interim words every **500ms** while user is still speaking | Waits for user to stop speaking before transcribing |
| **TTS Concurrency** | **Decoupled**: Generation thread computes sentence $N+1$ while sentence $N$ speaks | Blocks generation until previous audio finishes playing |
| **Barge-In (Interruption)** | Instantaneous audio cutoff (21ms slice window) without ALSA driver faults | Audio buffer latency, PortAudio buffer mmap crashes |
| **Interruption Quality** | Interrupting words are preserved in VAD memory for the next turn | First 1–2 words clipped or lost during interruption |
| **Wake Detection** | Multi-phrase, case-insensitive, configurable timeouts per phrase | Single fixed word, strict beginning-of-sentence prefix |
| **Reasoning Model Support**| Automatic real-time stripping of `<think>` reasoning monologues | Model speaks aloud its internal thoughts for 15+ seconds |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+, Debian, Fedora, Arch), macOS (Apple Silicon or Intel), Windows (WSL2 or native).
- **Python**: 3.10, 3.11, or 3.12.
- **Audio Drivers**: PortAudio and ALSA / PulseAudio / PipeWire (`libasound2-dev`, `portaudio19-dev`).

On Linux (Debian/Ubuntu):
```bash
sudo apt update && sudo apt install -y libasound2-dev portaudio19-dev libsndfile1
```

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/swar.git
cd swar

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install swar in editable mode
pip install -e .
```

---

## 🧪 How to Test

Swar comes with self-contained, interactive test examples in the `examples/` directory.

### Test 1: Real-Time Microphone Transcription (STT Only)
Test the microphone capture, Silero VAD, and Parakeet TDT real-time progressive streaming. You will see words appear in your terminal **as you speak**:

```bash
python examples/01_realtime_stt.py
```

### Test 2: Full Voice Chat with Any LLM (Continuous Mode)
Connect Swar to any local or remote OpenAI-compatible endpoint (e.g. Ollama, vLLM, LM Studio, or llama.cpp):

```bash
python examples/02_llm_voice_chat.py \
  --url http://127.0.0.1:8080/v1 \
  --model default
```

Speak into your mic:
- Watch live transcription appear.
- The assistant streams responses sentence-by-sentence.
- Sentence 1 speaks immediately while Sentence 2 is synthesizing in the background.
- **Test Barge-In**: Start talking while the assistant is speaking. It will instantly stop speaking and immediately transcribe your interruption without cutting off your opening words!

### Test 3: Wake Word & Wake Sentence Mode
Run voice chat in hands-free wake word standby:

```bash
python examples/02_llm_voice_chat.py \
  --url http://127.0.0.1:8080/v1 \
  --wake
```

- **Say**: `"Hello Relic, can you hear me?"`
- Swar recognizes `"relic"` (case-insensitively, anywhere in the sentence), wakes up, and sends `"Hello Relic, can you hear me?"` to the model.
- An interactive conversation window opens (default 10–20s timeout). Follow-up queries do not require repeating the wake word!

### Test 4: Custom Agent with `<speak>` Tag Parsing
Test an agent workflow where the model outputs thoughts, tool calls, and diagnostics on-screen, but **only speaks text wrapped in `<speak>...</speak>`**:

```bash
python examples/03_custom_agent.py
```

---

## 💻 Production Usage Guide

Swar is built with clean separation of concerns. You can use the high-level `VoicePipeline` orchestrator or use any sub-engine (`KokoroTTS`, `TTSPipeline`, `VADHandler`, `WakeWordEngine`) standalone.

### 1. Basic Production Implementation

```python
import sys
from voice_pipeline import VoicePipeline, PipelineConfig

# 1. Load configuration from YAML or create programmatically
config = PipelineConfig.from_yaml("config.yaml")

# 2. Define event handlers
def handle_partial(live_text: str):
    """Called every 500ms while user is speaking."""
    sys.stdout.write(f"\r[Live]: {live_text}")
    sys.stdout.flush()

def handle_final(final_transcript: str):
    """Called when user finishes a sentence."""
    print(f"\nUser said: {final_transcript}")
    
    # Send user query to your business logic / LLM
    response_text = my_llm_agent(final_transcript)
    
    # Speak response directly through the decoupled TTS
    pipeline.speak_text(response_text)

def handle_interrupted():
    """Called immediately when user speaks over assistant audio."""
    print("\n[Barge-in triggered: Assistant playback cut off]")

# 3. Instantiate Voice Pipeline
pipeline = VoicePipeline(
    config=config,
    on_partial_transcript=handle_partial,
    on_final_transcript=handle_final,
    on_interrupted=handle_interrupted,
)

# 4. Start pipeline
pipeline.start()

try:
    import time
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pipeline.stop()
```

---

### 2. Streaming LLM Tokens to Decoupled TTS in Production

To achieve minimum Time-to-First-Audio (TTFA), stream LLM tokens directly to `pipeline.stream_text_to_tts()`:

```python
from voice_pipeline import VoicePipeline

def handle_final_transcript(user_text: str):
    # LLM yields tokens or sentences iteratively
    token_generator = my_llm_client.stream(user_text)
    
    # TTS chunks tokens on sentence boundaries, computes audio ahead of time,
    # and automatically halts if user interrupts!
    pipeline.stream_text_to_tts(token_generator)
```

---

### 3. Standalone Decoupled TTS Pipeline

Use Kokoro TTS as an independent, high-throughput text-to-speech server:

```python
from voice_pipeline.tts import TTSPipeline, KokoroTTS
from voice_pipeline.speak_out_parser import ChunkingConfig

# Initialize Kokoro on CPU (or set device="cuda" for GPU)
tts_engine = KokoroTTS(voice="af_bella", device="cpu", speed=1.0)

# Create decoupled pipeline (Producer-Consumer architecture)
tts_pipeline = TTSPipeline(
    tts_engine=tts_engine,
    chunking_config=ChunkingConfig(
        progressive_stages=((4, "clause"), (8, "sentence")),
        max_chunk_words=25,
    )
)
tts_pipeline.start()

# Feed sentences — sentence 1 plays while sentence 2 synthesizes in parallel
tts_pipeline.feed_text("Hello! Swar is now speaking this first sentence out loud.")
tts_pipeline.feed_text("Notice how this second sentence was already computed before the first finished!")

# In case of user interruption:
# tts_pipeline.interrupt()
```

---

### 4. Standalone Wake Word & Sentence Engine

Integrate wake word detection into existing audio or text streams:

```python
from voice_pipeline.wake_word import WakeWordConfig, WakeWordEngine, WakeTriggerConfig

config = WakeWordConfig(
    enabled=True,
    prefix_only=False,          # Match anywhere in sentence
    strip_wake_phrase=False,    # Forward complete original text
    triggers=[
        WakeTriggerConfig(phrase="hey assistant", timeout=20.0),
        WakeTriggerConfig(phrase="jarvis", timeout=15.0),
        WakeTriggerConfig(phrase="relic", timeout=10.0),
    ]
)

engine = WakeWordEngine(config=config)
engine.start()

# Process incoming finalized speech transcripts
is_valid, text_to_forward = engine.process_transcript("Hello Relic, can you hear me?")
if is_valid:
    print(f"Wake triggered! Forwarding query: '{text_to_forward}'")
```

---

## ⚙️ Configuration Reference (`config.yaml`)

```yaml
general:
  device: "cpu"                # "cpu" or "cuda" (optimized for CPU)
  sample_rate: 16000           # Microphone audio sample rate
  allow_barge_in: true         # Enable user interruption during speech
  wake_mode: false             # Master toggle for wake word standby mode

vad:
  threshold: 0.5               # Silero probability threshold (0.1 - 0.9)
  min_silence_ms: 800          # Silence duration required to finalize a turn
  speech_pad_ms: 500           # Padding prepended to prevent clipped first words

stt:
  model_name: "nvidia/parakeet-tdt-0.6b-v3"
  language: "en"
  enable_live_transcription: true   # Real-time interim progressive updates

tts:
  model_name: "hexgrad/Kokoro-82M"
  voice: "af_bella"            # Voices: af_bella, af_sarah, am_adam, am_michael, etc.
  speed: 1.0                   # Speech speed multiplier (0.5 to 2.0)

wake_word:
  enabled: false
  default_timeout: 15.0        # Conversation exchange timeout in seconds
  strip_wake_phrase: false     # Forward complete original user transcript
  prefix_only: false           # Match wake word anywhere in sentence (case-insensitive)
  triggers:
    - phrase: "hey assistant"
      timeout: 20.0
    - phrase: "jarvis"
      timeout: 15.0
    - phrase: "relic"
      timeout: 10.0
    - phrase: "take a note"
      timeout: 30.0
```

---

## 📚 In-Depth Engineering Documentation

For a comprehensive deep-dive into the architectural decisions, math, and every challenging bug solved during development, read:

👉 **[`docs/DEVELOPMENT_JOURNEY_AND_ARCHITECTURE.md`](docs/DEVELOPMENT_JOURNEY_AND_ARCHITECTURE.md)**

Included in the deep-dive:
1. The transition from monolithic scripts to decoupled producer-consumer workers.
2. PortAudio / ALSA driver memory-map (`alsa_snd_pcm_mmap_begin`) segmentation crash resolution.
3. Recursive locking deadlocks in cross-thread TTS scheduling.
4. Speculative audio bleeding across conversational turns.
5. Real-time streaming suppression of LLM thinking tokens (`<think>...</think>`).
6. The "Relic" wake word bug and conversational salutation parsing.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
All underlying models retain their original open-source licenses (Silero: MIT, Parakeet TDT: CC-BY-4.0, Kokoro: Apache-2.0).
