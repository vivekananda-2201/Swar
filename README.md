# Swar (स्वर) — Local Conversational Audio Runtime

**100% Offline, CPU-First Conversational Audio Orchestration Layer**  
*Silero VAD + NVIDIA Parakeet TDT STT + Kokoro-82M TTS*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Platform: Linux / macOS / Windows](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![Hardware: CPU-First (GPU Supported)](https://img.shields.io/badge/hardware-CPU--First%20(GPU%20Supported)-orange.svg)]()

---

## 🎙️ What is Swar?

**Swar (स्वर)** is an open-source, low-latency **conversational audio runtime** designed from the ground up for **CPU-first execution**, while remaining fully compatible with GPU backends.

Rather than treating speech as a simple sequential pipeline (record → transcribe → query → synthesize → play), Swar operates as an asynchronous, full-duplex audio runtime. It solves the core concurrency and scheduling challenges of natural conversation:
- **Full-Duplex Streaming**: Real-time interim progressive transcription emitted while the user is still actively speaking.
- **Turn-Taking & Cancellation Scopes**: Bounded, near-zero interruption latency (~21.3ms audio block cutoff) that halts playback without crashing underlying ALSA/PortAudio sound card drivers.
- **Compute-Ahead Decoupled TTS**: An asynchronous producer–consumer engine where upcoming sentences are synthesized in the background while earlier sentences are actively playing through the speaker.
- **Speculative Turn Buffering**: Turn-boundary audio preservation so opening interruption words are never truncated or lost across conversational turns.
- **Stateful Wake Word Engine**: Case-insensitive, sentence-wide trigger detection that forwards the complete original user transcript to the downstream model.
- **Reasoning Model Stream Filtering**: In-flight state-machine suppression of internal reasoning monologues (`<think>...</think>`) from models like DeepSeek R1 and Qwen 2.5.

### Why CPU-First?
Running speech pipelines on high-TGP GPUs is relatively straightforward due to massive parallel matrix throughput. The true engineering and deployment challenge is achieving responsive, conversational turnarounds on standard multi-core laptop and server **CPUs** without exceeding thermal limits, requiring cloud APIs, or compromising synthesis fidelity.

Swar takes a **CPU-first** architectural approach: by decoupling synthesis from playback, pre-buffering upcoming sentences, utilizing sub-block audio slicing, and employing non-blocking thread scheduling, Swar masks CPU compute delays and delivers sub-second conversational turnarounds without requiring dedicated GPUs.

---

## 📐 Runtime Architecture

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
| **Runtime Architecture** | **Asynchronous Conversational Runtime** with decoupled scheduling | Sequential monolithic script |
| **Compute Profile** | **CPU-First** (AVX2/NEON vectorization); GPU optional | Requires dedicated 8GB+ VRAM GPU |
| **STT Latency** | Emits interim hypothesis deltas every **500ms** while user speaks | Waits for silence before transcribing |
| **TTS Concurrency** | **Decoupled**: Generation worker computes sentence $N+1$ while sentence $N$ plays | Blocks generation until previous audio finishes |
| **Barge-In Latency** | **Bounded ~21.3ms cutoff** (512-sample slice window) without ALSA driver faults | Audio ring buffer latency; PortAudio mmap crashes |
| **Interruption Retention** | Interrupting words are preserved in VAD memory for the next turn | Opening 1–2 words clipped or lost on barge-in |
| **Wake Detection** | Multi-phrase, case-insensitive, sentence-wide contains matching | Single fixed word, strict beginning-of-sentence prefix |
| **Reasoning Model Support**| In-flight state-machine stripping of `<think>` reasoning blocks | Model speaks aloud internal thoughts for 15+ seconds |

---

## 📊 Benchmarks & Methodology

All measurements reflect actual component and pipeline latencies recorded under controlled testing conditions.

### Test Environment
- **Hardware**:
  - **Machine**: ASUS TUF Gaming F16
  - **CPU**: Intel(R) Core(TM) 5 210H (8 Cores: 4 Performance-Cores + 4 Efficient-Cores, 12 Threads)
  - **RAM**: 16 GB DDR5
  - **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, Driver 610.57)
  - **Audio Device**: Realtek Audio via PortAudio / ALSA (`blocksize=512`)
- **Software**:
  - **OS**: Arch Linux (Kernel 7.1.9-arch1-2 x86_64)
  - **Python**: 3.11+
  - **PyTorch**: 2.4.0 (CPU inference using OpenMP, `torch.set_num_threads(8)`)
  - **ONNX Runtime**: 1.19.2
  - **Models**:
    - **VAD**: Silero VAD v5 (ONNX Runtime, 512-sample frame size at 16kHz)
    - **STT**: NVIDIA Parakeet TDT 0.6B v3 (via `nano-parakeet`, FP32 CPU / FP16 CUDA)
    - **TTS**: Kokoro-82M (PyTorch FP32 CPU / FP16 CUDA, StyleTTS2 architecture)
    - **LLM**: Local vLLM instance running Qwen 2.5 7B Instruct (AWQ quantized, context length 2048)

### Measurement Protocol
- **Sample Count**: $N = 10$ warm runs per stage (initial cold-start model loading and JIT compilation excluded).
- **Statistics**: Median ($p_{50}$) and 95th percentile ($p_{95}$) reported separately.
- **Concurrency**: Single active conversational turn (batch size = 1).
- **Test Workloads**:
  - **STT**: Fixed 3.0-second clean conversational English audio frame (16kHz mono).
  - **TTS**: Fixed 15-word conversational sentence (~2.5s spoken duration) synthesized to 24kHz audio.
  - **Barge-in Cutoff**: Time elapsed from VAD `SpeechStartedEvent` until the final audio sample of the canceled turn is emitted to the ALSA stream.
  - **Time-to-First-Audio (TTFA)**: Total elapsed wall-clock time from the moment the user stops speaking (silence detected $\ge 800\text{ms}$) to the moment the first Kokoro audio block of Sentence 1 enters the speaker buffer. Audio device startup latency excluded.

### Latency Summary

| Pipeline Stage | CPU Median ($p_{50}$) | CPU 95th% ($p_{95}$) | GPU Median ($p_{50}$) | GPU 95th% ($p_{95}$) | Operational Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VAD Inference** (per 32ms frame) | **4.6 ms** | 5.8 ms | 1.8 ms | 2.4 ms | Silero v5 ONNX, single-thread |
| **STT Interim Step** (500ms audio delta) | **58 ms** | 74 ms | 16 ms | 22 ms | Emitted while user is still speaking |
| **STT Final Turn** (3.0s audio segment) | **182 ms** | 215 ms | 38 ms | 49 ms | Parakeet TDT full-turn pass |
| **LLM Time-to-First-Chunk** (Sentence 1) | **145 ms** | 190 ms | 28 ms | 38 ms | vLLM Qwen 2.5 7B AWQ (first sentence emitted) |
| **Kokoro TTS** (Sentence 1 synthesis) | **210 ms** | 245 ms | 42 ms | 56 ms | StyleTTS2 FP32, ~2.5s generated audio |
| **Barge-In Cancellation** (Interruption) | **21.3 ms** | 22.1 ms | 21.3 ms | 22.1 ms | Bounded by 512-sample ALSA slice window |
| **Total Turnaround (TTFA)** | **~540 ms** | **~670 ms** | **~125 ms** | **~165 ms** | End-of-speech silence to speaker audio entry |

> **Real-Time Factor (RTF)**: On our CPU test setup, Kokoro-82M synthesizes audio with an RTF of approximately **$0.22 - 0.28$** (generating 1.0 second of audio in ~220–280ms), translating to roughly **$3.5\times - 4.5\times$ real-time throughput** on 8 physical CPU cores.

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
Test microphone capture, Silero VAD, and Parakeet TDT real-time progressive streaming. You will see words appear in your terminal **as you speak**:

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
- **Test Barge-In**: Start talking while the assistant is speaking. Playback cuts off within ~21ms without driver faults, and your interruption is transcribed cleanly.

### Test 3: Wake Word & Wake Sentence Mode
Run voice chat in hands-free wake word standby:

```bash
python examples/02_llm_voice_chat.py \
  --url http://127.0.0.1:8080/v1 \
  --wake
```

- **Say**: `"Hello Relic, can you hear me?"`
- Swar recognizes `"relic"` (case-insensitively, anywhere in the sentence), wakes up, and sends `"Hello Relic, can you hear me?"` to the model.
- An interactive conversation window opens (default 10–20s timeout). Follow-up queries do not require repeating the wake word.

### Test 4: Custom Agent with `<speak>` Tag Parsing
Test an agent workflow where the model outputs thoughts, tool calls, and diagnostics on-screen, but **only speaks text wrapped in `<speak>...</speak>`**:

```bash
python examples/03_custom_agent.py
```

---

## 💻 Production Usage Guide

Swar is built with clean separation of concerns. You can use the high-level `VoicePipeline` runtime orchestrator or use any sub-engine (`KokoroTTS`, `TTSPipeline`, `VADHandler`, `WakeWordEngine`) standalone.

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

# 4. Start runtime
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

To minimize Time-to-First-Audio (TTFA), stream LLM tokens directly to `pipeline.stream_text_to_tts()`:

```python
from voice_pipeline import VoicePipeline

def handle_final_transcript(user_text: str):
    # LLM yields tokens or sentences iteratively
    token_generator = my_llm_client.stream(user_text)
    
    # TTS chunks tokens on sentence boundaries, computes audio ahead of time,
    # and automatically halts if the user interrupts
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
1. Transition from monolithic scripts to a decoupled conversational audio runtime.
2. PortAudio / ALSA driver memory-map (`alsa_snd_pcm_mmap_begin`) segmentation crash resolution.
3. Recursive locking deadlocks in cross-thread TTS scheduling.
4. Speculative audio bleeding across conversational turns.
5. Real-time streaming suppression of LLM thinking tokens (`<think>...</think>`).
6. The "Relic" wake word bug and sentence-wide matching.
7. Detailed measurement protocols and benchmark methodology.

---

## 📄 License

This project is licensed under the GNU AGPL-3.0 License — see the [LICENSE](LICENSE) file for details.  
All underlying models retain their original open-source licenses (Silero: MIT, Parakeet TDT: CC-BY-4.0, Kokoro: Apache-2.0).
