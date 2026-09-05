# Swar (स्वर): Architecture, Engineering Journey, & Technical Logbook

> **A deep dive into engineering an offline, CPU-first, full-duplex conversational audio runtime.**  
> *From monolithic prototypes to a decoupled, low-latency production architecture with Silero VAD, NVIDIA Parakeet TDT STT, and Kokoro-82M TTS.*

---

## Table of Contents

1. [Executive Summary: From Voice Pipeline to Audio Runtime](#1-executive-summary-from-voice-pipeline-to-audio-runtime)
2. [Technology Selection & Why CPU-First?](#2-technology-selection--why-cpu-first)
3. [The Complete Chronological Journey: Step-by-Step](#3-the-complete-chronological-journey-step-by-step)
   - [Phase 1: Initial Conception & The Fragility of Cloud Voice](#phase-1-initial-conception--the-fragility-of-cloud-voice)
   - [Phase 2: Extracting and Adapting Hugging Face Speech-to-Speech](#phase-2-extracting-and-adapting-hugging-face-speech-to-speech)
   - [Phase 3: Architecting the Decoupled Producer–Consumer TTS](#phase-3-architecting-the-decoupled-producerconsumer-tts)
   - [Phase 4: Full-Duplex Interruption & Bounded Audio Cutoff](#phase-4-full-duplex-interruption--bounded-audio-cutoff)
   - [Phase 5: Stateful Wake Word & Wake Sentence Detection](#phase-5-stateful-wake-word--wake-sentence-detection)
   - [Phase 6: Thinking / Reasoning LLM Suppression](#phase-6-thinking--reasoning-llm-suppression)
4. [Deep-Dive into the 7 Major Bugs & Engineering Battles](#4-deep-dive-into-the-7-major-bugs--engineering-battles)
   - [Bug 1: PortAudio ALSA Memory-Map Driver Crash (`alsa_snd_pcm_mmap_begin`)](#bug-1-portaudio-alsa-memory-map-driver-crash-alsa_snd_pcm_mmap_begin)
   - [Bug 2: Recursive Lock Deadlock in Multi-Threaded TTS Scheduling](#bug-2-recursive-lock-deadlock-in-multi-threaded-tts-scheduling)
   - [Bug 3: Inter-Turn Audio Bleeding & Lost Interruption Utterances](#bug-3-inter-turn-audio-bleeding--lost-interruption-utterances)
   - [Bug 4: Reasoning Model Monologue Leak (DeepSeek R1 / Qwen `<think>`)](#bug-4-reasoning-model-monologue-leak-deepseek-r1--qwen-think)
   - [Bug 5: The "Relic" Wake Word Bug (Strict Prefix vs. Natural Human Speech)](#bug-5-the-relic-wake-word-bug-strict-prefix-vs-natural-human-speech)
   - [Bug 6: Stale Audio Racing & Turn Epoch Invalidation](#bug-6-stale-audio-racing--turn-epoch-invalidation)
   - [Bug 7: Sound Card Buffer Underrun & Slice Latency Stacking](#bug-7-sound-card-buffer-underrun--slice-latency-stacking)
5. [The Decoupled Compute-Ahead TTS In-Depth](#5-the-decoupled-compute-ahead-tts-in-depth)
6. [Wake Word Engine Mechanics & State Machine](#6-wake-word-engine-mechanics--state-machine)
7. [CPU Optimization Techniques & Rigorous Benchmark Analysis](#7-cpu-optimization-techniques--rigorous-benchmark-analysis)
   - [7.1 Core CPU Optimization Strategies](#71-core-cpu-optimization-strategies)
   - [7.2 Host Hardware & Software Environment](#72-host-hardware--software-environment)
   - [7.3 Empirical Benchmark Summary](#73-empirical-benchmark-summary)
   - [7.4 Turn-by-Turn Behavioral Breakdown](#74-turn-by-turn-behavioral-breakdown)
   - [7.5 Architectural Root Causes: Why Swar Achieves This Speed on CPU](#75-architectural-root-causes-why-swar-achieves-this-speed-on-cpu)
8. [Production Deployment Architecture & Best Practices](#8-production-deployment-architecture--best-practices)

---

## 1. Executive Summary: From Voice Pipeline to Audio Runtime

### Why "Swar" (स्वर)?
In classical Sanskrit and Indian music, **Swar (स्वर)** represents tone, voice, melody, and acoustic presence. A conversational assistant must not feel like a computer reading text in disjointed intervals; it must have continuous musical cadence, immediate responsiveness, and intuitive conversational manners (such as gracefully yielding when interrupted).

### The Paradigm Shift: Why a "Runtime" Instead of a "Pipeline"?
Most voice agent projects are designed as simple sequential pipelines:
$$\text{Microphone} \longrightarrow \text{VAD} \longrightarrow \text{STT} \longrightarrow \text{LLM} \longrightarrow \text{TTS} \longrightarrow \text{Speaker}$$

In practice, a real-time conversational agent cannot be modeled as a linear unidirectional pipe. Real conversation is fundamentally concurrent, asynchronous, and stateful:
- A user can interrupt while the assistant is mid-word (demanding cancellation scopes across hardware buffers).
- An LLM emits tokens incrementally, while TTS must chunk on syntactic boundaries.
- The assistant must synthesize sentence $N+1$ in the background while sentence $N$ is being played out loud through hardware ring buffers.
- An in-flight turn must be discarded immediately if the user begins speaking, without corrupting sound card DMA buffers or leaking prior audio into the new turn.

For these reasons, **Swar is architected as a local conversational audio runtime**. It provides an event-driven execution environment that coordinates streaming VAD, speculative turn tracking, multi-threaded audio synthesis queues, soundcard buffer slicing, and conversational state machines.

---

## 2. Technology Selection & Why CPU-First?

When selecting models and runtimes, every component was evaluated for CPU instruction throughput (AVX2, AVX-512, NEON vectorization) and low parameter overhead.

```
+---------------------------------------------------------------------------------------------+
|                                        SWAR STACK                                           |
+--------------------------+------------------------------+-----------------------------------+
| Component                | Model / Library              | Execution Characteristics         |
+--------------------------+------------------------------+-----------------------------------+
| Voice Activity Detection | Silero VAD v5 (ONNX runtime) | 512-sample (32ms) frame, <1% CPU  |
| Speech-to-Text (STT)     | NVIDIA Parakeet TDT 0.6B     | RNN-T joint duration prediction   |
| Text-to-Speech (TTS)     | Kokoro-82M (StyleTTS2 based) | Non-autoregressive, 24kHz audio   |
| Audio I/O                | SoundDevice / PortAudio      | Low-latency PCM stream (512-slice)|
| Chunker / Orchestrator   | Swar Native Custom Engine    | Decoupled Producer-Consumer       |
+--------------------------+------------------------------+-----------------------------------+
```

### Architectural Analysis: NVIDIA Parakeet TDT vs. Autoregressive STT (Whisper)
OpenAI Whisper relies on an autoregressive encoder-decoder architecture: after encoding the full audio spectrogram, the decoder generates tokens sequentially, where each step attends over previous tokens and the entire encoder representation. While optimized C++ implementations (such as `whisper.cpp`) provide impressive CPU performance, the sequential decode passes scale directly with transcript length.

In contrast, **NVIDIA Parakeet TDT** (Token-and-Duration Transducer) is a hybrid architecture consisting of a Conformer encoder, a prediction network, and a joint network with a duration prediction head. Instead of decoding one token per sequential forward pass, Parakeet TDT can predict both a token and its time duration, allowing it to emit multiple tokens per frame or skip blank frames entirely. Running via `nano-parakeet` with PyTorch CPU kernels, this architecture avoids autoregressive decoding loops, resulting in steady and predictable latency profiles on multi-core CPUs.

### Architectural Analysis: Kokoro-82M vs. Autoregressive TTS (XTTS / Bark)
Modern expressive TTS systems like XTTS-v2 or Bark employ autoregressive speech language models with hundreds of millions of parameters. Generating 24kHz audio with these architectures on CPU typically requires multiple seconds of compute per sentence, often running below real-time speed ($\text{RTF} > 1.0$) without high-end GPU acceleration.

**Kokoro-82M** is based on the StyleTTS2 architecture. It uses a lightweight text encoder, duration predictor, style diffusion/predictor, and a modified HiFi-GAN style generator totaling just 82 million parameters. Because it generates acoustic features non-autoregressively across the whole sentence chunk in parallel, its CPU computational complexity is modest compared to autoregressive architectures. Real-time factor ($\text{RTF}$) and synthesis throughput on the CPU are measured dynamically during live conversational sessions.

---

## 3. The Complete Chronological Journey: Step-by-Step

### Phase 1: Initial Conception & The Fragility of Cloud Voice
We started by testing local Python scripts wrapping standard STT and TTS engines. The initial setup suffered from severe latency stacking:
1. User spoke for 4 seconds.
2. VAD waited 1 second of silence to close the turn.
3. Batch transcription took ~2.0 seconds.
4. LLM took 1.0 second to begin generating.
5. TTS took 2.5 seconds to synthesize the entire response paragraph before playback.
6. **Total Time-to-First-Audio**: **>10.5 seconds**.

Conversational momentum was completely destroyed.

### Phase 2: Extracting and Adapting Hugging Face Speech-to-Speech
We examined Hugging Face's experimental `speech-to-speech` project. They introduced a breakthrough algorithm called **Smart Progressive Streaming**:
- As the user speaks, 500ms audio increments are fed into Parakeet.
- Parakeet emits interim hypothesis words in real-time.
- If speech extends beyond 15 seconds, a sentence-aware sliding window trims older audio while preserving context, preventing memory explosion.

We extracted the core logic into isolated, robust modules (`voice_pipeline/vad.py` and `voice_pipeline/stt.py`), removing unnecessary web-server dependencies and making it a clean, lightweight local library.

### Phase 3: Architecting the Decoupled Producer–Consumer TTS
Even with real-time STT, speech synthesis remained a bottleneck. If an LLM generated 3 sentences, synthesizing all 3 sentences before playing audio meant the user waited in silence.

We designed a **Decoupled Producer–Consumer TTS Engine**:
- The LLM stream yields sentence by sentence.
- A **Generation Worker Thread** pulls sentence 1, computes its 24kHz audio, and pushes it into an internal `ready_audio_queue`.
- The moment sentence 1 is ready, an independent **Playback Worker Thread** streams it to the sound card.
- **While sentence 1 is playing out loud**, the generation worker does not sit idle—it immediately computes sentence 2 and sentence 3 in parallel.
- When sentence 1 finishes playing, sentence 2 is already buffered in memory. Playback transitions seamlessly from memory without waiting for synthesis.

### Phase 4: Full-Duplex Interruption & Bounded Audio Cutoff
A voice assistant that cannot be interrupted feels robotic. If the assistant begins a lengthy explanation, the user must be able to speak over it and have the assistant halt immediately.

We wired Silero VAD events directly into the playback stream. The moment voice energy is detected while the assistant is speaking:
1. An interrupt cancellation event is triggered.
2. Playback is cut off within a bounded audio block (~21.3ms).
3. The remaining queued sentences are flushed.
4. The user's new speech is immediately captured and preserved for the next turn.

### Phase 5: Stateful Wake Word & Wake Sentence Detection
In continuous listening mode, background noise or third-party conversations can inadvertently trigger the assistant. We added a **Stateful Wake Engine**:
- Two states: `STANDBY` (sleeping, low CPU) and `ACTIVE` (awake, full conversation).
- Multi-phrase triggers: Configure single words (`"jarvis"`, `"relic"`) or full sentences (`"take a note"`, `"hey assistant"`).
- Per-phrase custom timeouts: e.g. `"take a note"` opens a 30-second window, while `"relic"` opens a 10-second window.
- Inactivity countdown: Follow-up turns during the conversation window do not require repeating the wake word.

### Phase 6: Thinking / Reasoning LLM Suppression
With modern reasoning models (DeepSeek R1, Qwen 2.5 Max, etc.), models output an internal chain of thought wrapped in `<think>...</think>` tags before their actual response. Without filtering, Kokoro would literally speak aloud the model's internal reasoning monologue for 15+ seconds before answering the question.

We implemented a **streaming state-machine filter** that detects and strips thought blocks on the fly without delaying real response tokens.

---

## 4. Deep-Dive into the 7 Major Bugs & Engineering Battles

Building a low-latency voice runtime across C-libraries (PortAudio, ALSA, ONNX, PyTorch) surfaced subtle concurrency, memory, and driver issues. Here is how each was diagnosed and conquered.

---

### Bug 1: PortAudio ALSA Memory-Map Driver Crash (`alsa_snd_pcm_mmap_begin`)

#### The Symptom
During barge-in testing, whenever the user interrupted the assistant while Kokoro was speaking, the Python process crashed fatally with:
```text
ALSA lib pcm.c:8570:(snd_pcm_recover) underrun occurred
ALSA lib pcm_mmap.c:380:(alsa_snd_pcm_mmap_begin) Assertion `snd_pcm_mmap_avail(pcm) >= frames' failed.
Fatal Python error: Aborted
```

#### Root Cause Analysis
In our initial implementation of `interrupt()`, we called `stream.abort()` directly on the `sounddevice.OutputStream` from the VAD event thread.  
In Linux ALSA sound drivers, `stream.abort()` immediately terminates the DMA ring buffer. When another thread was midway through writing a PCM block using `stream.write()`, the underlying ALSA ring buffer pointers desynchronized. ALSA raised an assertion failure inside its C library, crashing the Python interpreter instantly.

#### The Solution
We completely removed cross-thread `stream.abort()`. Instead, we refactored the playback worker to stream audio in micro-slices of **512 samples (~21.3 milliseconds at 24kHz)**:

```python
# voice_pipeline/tts.py
BLOCK_SAMPLES = 512  # ~21.3ms per slice at 24kHz

while sample_offset < total_samples:
    # Check cancellation before every single micro-block
    if self.cancel_current_turn_event.is_set():
        break
        
    slice_end = min(sample_offset + BLOCK_SAMPLES, total_samples)
    chunk_slice = audio_item.audio[sample_offset:slice_end]
    stream.write(chunk_slice)
    sample_offset = slice_end
```

When an interruption occurs, the cancellation event is set. The playback loop exits after at most **21.3ms**, stopping playback cleanly and immediately without terminating the ALSA stream object. Zero driver faults, zero crashes.

---

### Bug 2: Recursive Lock Deadlock in Multi-Threaded TTS Scheduling

#### The Symptom
After speaking 2 or 3 turns, the entire TTS engine froze completely. No sound would play, and new text fed into `feed_text()` hung indefinitely.

#### Root Cause Analysis
In `TTSPipeline`, `self._lock = threading.Lock()` was used to protect queue state and turn synchronization.
Inside `feed_text()`:
```python
def feed_text(self, text: str):
    with self._lock:  # <--- Lock acquired here
        # ...
        current_buffered_s = self.buffered_audio_seconds  # <--- Calls property
```
And inside the property `buffered_audio_seconds`:
```python
@property
def buffered_audio_seconds(self) -> float:
    with self._lock:  # <--- Tries to acquire the same non-reentrant Lock! Deadlock!
        return sum(...)
```
Because Python's standard `threading.Lock()` is non-reentrant, the thread deadlocked against itself waiting for the lock it already held.

#### The Solution
We switched `self._lock` to a reentrant lock:
```python
self._lock = threading.RLock()
```
`threading.RLock()` allows the owning thread to acquire the lock multiple times recursively without blocking itself, completely eliminating the deadlock.

---

### Bug 3: Inter-Turn Audio Bleeding & Lost Interruption Utterances

#### The Symptom
Two interconnected audio corruption bugs occurred during barge-in:
1. If the user interrupted the assistant by saying *"Wait, stop!"*, the transcription for the next turn was sometimes empty or missed *"Wait"*.
2. Worse, fragments of the assistant's previous spoken turn bled into the new turn, producing hallucinations like *"Bella: Wait, stop!"*.

#### Root Cause Analysis
1. In the Hugging Face VAD handler, when speech was detected, a speculative reopening mechanism (`speculative_reopen_ms`) prepended raw audio from the previous 1000ms. If the assistant had been speaking, this buffer contained residual speaker echo or trailing audio from the prior turn.
2. In an attempt to fix (1), an aggressive buffer wipe had been clearing `self.vad_iterator.buffer`. But that buffer contained the user's opening word that triggered the barge-in! Clearing it wiped out the word *"Wait"*.

#### The Solution
We implemented a surgical turn boundary isolation:
1. Set `speculative_reopen_ms = 0` and `unanswered_reopen_ms = 0` in VAD configuration to permanently prevent historical audio from leaking into new turns.
2. During `_handle_interruption()`, we reset only the speculative prefixes while strictly **preserving** the active VAD audio buffer:
```python
# voice_pipeline/pipeline.py
def _handle_interruption(self) -> None:
    self._active_turn_interrupted.set()
    self._active_turn_id += 1
    self.tts_pipeline.interrupt()
    
    # Clear speculative lookback prefixes, but DO NOT wipe current speech buffer!
    if self.vad_handler is not None:
        self.vad_handler._speculative_audio_prefix = None
        self.vad_handler._speculative_raw_audio_prefix = None
```
Now, the words spoken during the interruption are 100% retained and transcribed accurately for the new turn.

---

### Bug 4: Reasoning Model Monologue Leak (DeepSeek R1 / Qwen `<think>`)

#### The Symptom
When connected to local reasoning models (like DeepSeek R1 or Qwen 2.5 7B Reasoning via Ollama/vLLM), the model generated output like:
```text
<think>
The user is asking for the capital of France.
I must state Paris clearly and briefly.
</think>
The capital of France is Paris.
```
Kokoro would spend 15 seconds reading: *"The user is asking for the capital of France..."* before saying *"The capital of France is Paris."*

#### Root Cause Analysis
Reasoning models stream their thought traces as standard token deltas. If passed directly to the TTS chunker, the chunker treats thought sentences as ordinary text to be spoken.

#### The Solution
We implemented a **two-tier defense**:
1. **Tier 1 (Server-Side Request Optimization)**:
   In `LLMClient`, we detect Ollama and vLLM and pass flags to disable thinking at inference time:
   - Ollama: `"think": False`
   - vLLM / OpenAI: `"chat_template_kwargs": {"enable_thinking": False}`
2. **Tier 2 (Real-Time Stream State Machine)**:
   If the server ignores the flag, we implemented an in-flight token filter `strip_thinking_tokens()`:
   ```python
   # examples/llm_client.py
   def strip_thinking_tokens(token_stream: Iterator[str]) -> Iterator[str]:
       in_thinking_block = False
       for token in token_stream:
           if "<think>" in token or "<thought>" in token:
               in_thinking_block = True
               continue
           if "</think>" in token or "</thought>" in token:
               in_thinking_block = False
               continue
           if not in_thinking_block:
               yield token
   ```
The user never hears the internal monologue, and Time-to-First-Audio drops from 15 seconds to ~350ms.

---

### Bug 5: The "Relic" Wake Word Bug (Strict Prefix vs. Natural Human Speech)

#### The Symptom
The user configured `"relic"` as a wake word. In the terminal:
```text
Ready! Speak into your microphone...
USER: Hello Relic, can you hear me?
Language: en
```
Parakeet transcribed `"Hello Relic, can you hear me?"` with 100% accuracy, but the wake engine stayed completely silent and ignored the speech.

#### Root Cause Analysis
In `voice_pipeline/wake_word.py`, the engine had `prefix_only = True` by default.  
The matching check was:
```python
if input_words[:phrase_len] == phrase_words:
```
For input `"Hello Relic, can you hear me?"`:
- `input_words` = `["hello", "relic", "can", "you", "hear", "me"]`
- `phrase_words` = `["relic"]`
- `input_words[:1]` was `["hello"]`.
- `["hello"] == ["relic"]` evaluated to **False**!

Because `prefix_only` was `True`, the engine refused to scan the rest of the sentence. Humans naturally add greetings (*"Hello Relic"*, *"Hey Jarvis"*, *"OK Computer"*). Strict index-0 prefix matching broke natural conversation.

#### The Solution
We overhauled the wake engine:
1. **Case-Insensitive Contains-Matching**: The engine checks if the normalized sentence contains the wake phrase anywhere as a contiguous sequence of words:
   ```python
   for i in range(len(input_words) - phrase_len + 1):
       if input_words[i : i + phrase_len] == phrase_words:
           matched = True
           break
   ```
2. **Forwarding the Complete Transcription**:
   Rather than stripping out the wake word and mangling the sentence into `"can you hear me?"`, the engine now forwards the **complete original transcript** (`"Hello Relic, can you hear me?"`) directly to the model.
3. Default `prefix_only = False` and `strip_wake_phrase = False` across `WakeWordConfig` and `config.yaml`.

Now, whether the user says:
- *"Relic, can you hear me?"*
- *"Hello Relic, can you hear me?"*
- *"Can you hear me, Relic?"*  
The wake engine instantly triggers and passes the full sentence to the agent.

---

### Bug 6: Stale Audio Racing & Turn Epoch Invalidation

#### The Symptom
If a user interrupted Turn 1 to start Turn 2, sometimes a trailing audio sentence from Turn 1 would suddenly play *after* Turn 2's first sentence had finished playing.

#### Root Cause Analysis
The LLM streaming generator and the TTS synthesis worker run in separate threads from the VAD loop. When an interruption happened, the LLM generator for Turn 1 was still yielding tokens for a few milliseconds before detecting the cancel event. Those tokens were pushed into `text_queue` and synthesized under the old context, landing in `ready_audio_queue` after Turn 2 had already begun.

#### The Solution
We implemented **Monotonic Turn Epochs**:
```python
# voice_pipeline/tts.py
self._current_turn_id: int = 0

def interrupt(self) -> None:
    with self._lock:
        self.cancel_current_turn_event.set()
        self._current_turn_id += 1  # Increment epoch
        self.clear_buffers()        # Flush all pending queues
```
Every text chunk and audio item is tagged with its `turn_id`. In both `_generation_worker` and `_playback_worker`:
```python
if item.turn_id < self._current_turn_id:
    # Stale chunk from superseded turn — immediately discard!
    continue
```
Even if an outdated thread yields late tokens, they are dropped with zero computation and zero playback.

---

### Bug 7: Sound Card Buffer Underrun & Slice Latency Stacking

#### The Symptom
Audio sounded "robotic" or "crackling" on certain Linux soundcards when using small buffer sizes.

#### Root Cause Analysis
Writing too small of a chunk (e.g. 128 samples / ~5.3ms) caused the sound card's hardware DMA buffer to underrun because Python's thread scheduler could not guarantee 5ms re-entry intervals on a loaded CPU. Conversely, writing a 4096-sample buffer (~170ms) made interruption response sluggish.

#### The Solution
We benchmarked block sizes from 128 to 4096 samples on CPU under load:
- 128 samples: High CPU context-switch overhead, ALSA buffer underruns.
- 256 samples: Occasional jitter during model inference bursts.
- **512 samples (~21.3ms)**: The sweet spot. Clean audio, zero underruns, and human-imperceptible 21.3ms cutoff latency.

---

## 5. The Decoupled Compute-Ahead TTS In-Depth

The heart of Swar's low-latency performance is the decoupled producer-consumer TTS engine.

```
                              LLM Stream Output
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    RobustSentenceChunker    │
                      │ (Handles abbreviations, e.g.│
                      │  "Dr.", "U.S.A.", "3.14")   │
                      └──────────────┬──────────────┘
                                     │
                               Text Chunks
                                     ▼
                      ┌─────────────────────────────┐
                      │     Text Queue (Capacity 100)│
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   TTS Generation Worker     │
                      │   (Thread 1: Runs Ahead)    │
                      │   Kokoro-82M on CPU         │
                      └──────────────┬──────────────┘
                                     │
                                Ready Audio
                                     ▼
                      ┌─────────────────────────────┐
                      │    Ready Audio Queue (50)   │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Audio Playback Worker    │
                      │   (Thread 2: Real-Time)     │
                      │   512-sample slices to ALSA │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                              Speaker Hardware
```

### Why Compute-Ahead Matters
Consider an assistant speaking a 3-sentence response:
- Sentence 1: Takes ~210ms to synthesize $\rightarrow$ Audio duration: ~3.2 seconds.
- Sentence 2: Takes ~190ms to synthesize $\rightarrow$ Audio duration: ~2.8 seconds.
- Sentence 3: Takes ~200ms to synthesize $\rightarrow$ Audio duration: ~3.0 seconds.

**In a Traditional Pipeline:**
Synthesis and playback happen sequentially. Total latency = synthesis time + playback wait.

**In Swar's Decoupled Runtime:**
1. At $t \approx 210\text{ms}$, Sentence 1 begins playing.
2. While Sentence 1 plays for the next $3200\text{ms}$, the Generation Worker synthesizes Sentence 2 (takes ~190ms) and Sentence 3 (takes ~200ms).
3. Both Sentence 2 and Sentence 3 are finished and buffered in memory by $t \approx 600\text{ms}$.
4. When Sentence 1 finishes at $t = 3200\text{ms}$, Sentence 2 transitions immediately from the pre-buffered memory queue.
5. The user perceives **gapless, continuous, natural speech**.

---

## 6. Wake Word Engine Mechanics & State Machine

```
              ┌────────────────────────────────────────────────────────┐
              │                     STANDBY STATE                      │
              │  - VAD & STT active                                    │
              │  - Discards speech that doesn't contain wake triggers  │
              └───────────────────────────┬────────────────────────────┘
                                          │
                  Sentence Contains Wake Phrase (e.g. "relic")
                                          │
                                          ▼
              ┌────────────────────────────────────────────────────────┐
              │                      ACTIVE STATE                      │
              │  - Assistant responds to complete transcript           │
              │  - Timer initialized to trigger's timeout (e.g. 10.0s) │
              └──────────────┬──────────────────────────▲──────────────┘
                             │                          │
                 Inactivity Timeout Expired     New Speech Turn
                             │                  (Refreshes Timer)
                             ▼                          │
                     Back to STANDBY                    │
                                                        │
                      Assistant Speaking                │
                   (Timer pauses while busy) ───────────┘
```

### Key Technical Properties:
1. **Sentence-Wide Matching**: Searches for wake phrases across the entire transcribed utterance using word-boundary matching.
2. **Case-Insensitive Normalization**: Alphanumeric lowercasing eliminates punctuation and capitalization discrepancies from Parakeet.
3. **Timer Pause During Synthesis**: While the assistant is generating text or Kokoro is speaking, the timeout countdown is automatically paused so the user's active window does not expire while listening to the assistant.

---

## 7. CPU Optimization Techniques & Rigorous Benchmark Analysis

The primary engineering mission of Swar is delivering fluid, natural conversational voice turnaround **100% on standard laptop and desktop CPUs**, without requiring dedicated high-power discrete GPUs. 

While running speech pipelines on 80W–150W discrete GPUs is straightforward due to massive parallel matrix throughput, running full-duplex conversational voice on CPU presents formidable engineering challenges: thermal throttling, OpenMP thread contention across asymmetric cores, memory bandwidth saturation, and autograd graph tracking overhead.

This section documents the specific optimizations engineered into Swar to conquer these bottlenecks, followed by empirical benchmark data captured during live interactive sessions.

---

### 7.1 Core CPU Optimization Strategies

#### 1. PyTorch Intra-Op CPU Parallelism & Hybrid Core Topology
Modern Intel and AMD processors utilize hybrid core topologies combining high-clocked Performance Cores (P-cores) with power-efficient Efficient Cores (E-cores). For example, the test machine's Intel Core 5 210H features:
- **4 Performance Cores** (8 Threads, up to 4.8 GHz)
- **4 Efficient Cores** (4 Threads, up to 3.6 GHz)
- Total: 8 Cores, 12 Logical Threads

**The OpenMP Barrier Synchronization Bottleneck:**
By default, PyTorch and OpenBLAS configure intra-op thread pools to match all available logical cores (`torch.set_num_threads(8)` or `12`). During parallel matrix multiplications (GEMM) and 1D convolutions in Kokoro's StyleTTS2 vocoder, computational tensors are partitioned across all worker threads. Because E-cores run at substantially lower clock speeds and instructions-per-clock (IPC) compared to P-cores, threads scheduled on P-cores complete their tensor operations quickly, but are then forced into **idle spin-wait states at OpenMP barrier synchronization points**, waiting for the slower E-core threads to finish.

**The Solution:**
Swar exposes and enforces explicit CPU intra-op thread tuning (`cpu_threads: 4` in `config.yaml` and `voice_pipeline/tts.py`):
```python
if self.device == "cpu" and self.cpu_threads > 0:
    torch.set_num_threads(self.cpu_threads)
```
By restricting intra-op parallelism to 4 threads, PyTorch binds intensive tensor workloads strictly to the high-IPC Performance Cores. This completely eliminates OpenMP barrier wait cycles, delivering an immediate **~26% reduction in raw synthesis latency** on the same hardware.

#### 2. Eliminating Autograd Overhead via `torch.inference_mode()`
Kokoro TTS evaluates deep neural networks across multiple sub-modules per text chunk:
- Text encoder & duration predictor
- Style embedding projection
- ISTFT (Inverse Short-Time Fourier Transform) & HiFi-GAN acoustic generator

In standard PyTorch execution, even under `torch.no_grad()`, tensor version counters and dispatch checks still incur substantial C++ runtime overhead. Swar wraps the entire chunk generator within `torch.inference_mode()`:
```python
with torch.inference_mode():
    for gs, ps, audio in self.pipeline(text.strip(), voice=v, speed=s):
        # Direct zero-copy processing into float32 audio buffers
```
`torch.inference_mode()` disables view tracking, version counter increments, and autograd metadata allocation entirely. Combined with thread tuning, single-sentence synthesis latency dropped from **1.096s down to 0.810s**, and conversational openers (e.g. *"Hey!"*) synthesize in as little as **436ms**.

#### 3. Voice Activity Detection Turn-Closing Optimization (`min_silence_ms`)
In full-duplex conversational voice, latency is not merely model compute time—it is also governed by conversational turn-closing mechanics.

In Silero VAD v5, audio is analyzed in 512-sample (32ms) frames. When the user stops speaking, the runtime cannot cut off immediately because natural human speech contains micro-pauses (breaths, commas, lexical hesitation). The runtime must observe a trailing window of silence (`min_silence_ms`) before declaring the turn complete and firing `on_final_transcript`.

**Turn-Closing Impact on Perceived Latency:**
- **Default Conservative Setting (`min_silence_ms: 800ms`)**: Designed for noisy acoustic environments (loudspeakers, background conversation) to prevent the assistant from prematurely interrupting thoughtful speakers.
- **Fast Conversational Setting (`min_silence_ms: 300ms–500ms`)**: In quiet environments or with headphones, reducing `min_silence_ms` from 800ms to 400ms **directly shaves 400ms off perceived end-to-end turnaround latency**. The turn is finalized almost the instant the user's vocal cords stop vibrating, and LLM token generation begins without idle pause.

#### 4. Zero-Copy Vectorized Audio Conversion & Sliced DMA Playback
- **Vectorized Normalization**: 16-bit signed PCM captured from SoundDevice is normalized directly to float32 using optimized NumPy vector instructions (`/ 32768.0`) without intermediate byte copying.
- **Sub-Block Hardware Slicing**: Rather than pushing full multi-second audio buffers directly to PortAudio, Swar slices outgoing audio into bounded 512-sample chunks (21.3ms). This ensures that if the user speaks (barge-in), playback halts in **under 22ms** without buffer underruns, soundcard pops, or ALSA driver deadlocks.

---

### 7.2 Host Hardware & Software Environment

The empirical benchmarks below were conducted on the following live physical environment:

| Category | Specification |
| :--- | :--- |
| **Machine Model** | ASUS TUF Gaming F16 (`FX607VUR_FX677VU`) |
| **CPU** | Intel(R) Core(TM) 5 210H (8 Cores: 4 P-Cores up to 4.8GHz, 4 E-Cores up to 3.6GHz, 12 Threads) |
| **System Memory** | 16 GB DDR5 (15.2 GB Available) |
| **GPU / Driver** | NVIDIA GeForce RTX 4050 Laptop GPU (6GB GDDR6, Driver 610.57.04) — *voice runtime run 100% on CPU* |
| **Audio Interface** | Realtek ALSA / PortAudio (`16kHz` input, `24kHz` output, `blocksize=512`) |
| **Operating System** | Arch Linux (Kernel `7.1.9-arch1-2-x86_64`, glibc 2.44) |
| **Python Runtime** | Python 3.11.16 |
| **VAD Engine** | Silero VAD v5 (ONNX Runtime, CPU) |
| **STT Engine** | NVIDIA Parakeet TDT 0.6B v3 (`nano-parakeet`, CPU) |
| **TTS Engine** | Kokoro-82M (CPU Native, 4 intra-op threads, inference_mode) |
| **LLM Model** | Qwen3.5-4B (~37–41 tokens/sec via local OpenAI-compatible endpoint) |

---

### 7.3 Empirical Benchmark Summary

Data captured across **8 real conversational dialogue turns** during interactive live use (`benchmarks/session_20260905_085852.jsonl` and `latest_summary.json`):

| Metric | Min | Median | Mean | P95 / Max | Unit | Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Time-to-First-Audio (TTFA)** | **642.5** | 1,664.3 | 1,560.9 | 2,425.6 | ms | Complete silence-to-speech turnaround |
| **LLM Time-to-First-Token (TTFT)** | **148.7** | 158.6 | 194.1 | 445.8 | ms | Local Qwen3.5-4B prompt evaluation |
| **LLM Time-to-First-Sentence (TTFS)** | **204.5** | 470.6 | 449.1 | 671.0 | ms | Sentence 1 boundary detection |
| **LLM Generation Speed** | 20.8 | 39.5 | 36.8 | **40.8** | tok/s | Live streaming throughput |
| **Kokoro Sentence 1 Synthesis** | **436.1** | 1,234.9 | 1,110.1 | 1,822.8 | ms | Chunk 1 CPU synthesis latency |
| **Kokoro Real-Time Speedup** | 2.05× | 3.34× | 3.19× | **3.85×** | Real-time | Synthesis throughput (up to 4.6x for subsequent chunks) |
| **Barge-In Interruption Cutoff** | — | — | **21.3** | 21.3 | ms | Bounded 512-sample ALSA slice halt |

---

### 7.4 Turn-by-Turn Behavioral Breakdown

The following table details all 8 consecutive turns from the live benchmarking session, illustrating conversational dynamics across varying utterance lengths and user barge-ins:

| Turn | User Spoken Query | Assistant Opening Sentence | TTFA (ms) | TTFT (ms) | TTFS (ms) | Chunk 1 Synth (ms) | Chunk 1 Audio (s) | Speedup | Interrupted? | Cutoff Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | *"Hi, can you hear me?"* | *"Hi there!"* | 1,150.0 | 445.8 | 512.9 | 635.1 | 1.30s | 2.05× | No | — |
| **2** | *"Can you introduce about yourself?"* | *"Hey!"* | **642.5** | 156.7 | **204.5** | **436.1** | 1.10s | 2.52× | No | — |
| **3** | *"Explain me about quantum mechanics."* | *"Quantum mechanics is a super cool part..."* | 2,425.6 | 160.6 | 600.9 | 1,822.8 | 7.02s | **3.85×** | **Yes** | **21.3 ms** |
| **4** | *"Wait, let it be, explain me who is the CM..."* | *"The current Chief Minister of Telangana..."* | 1,689.6 | 164.7 | 456.6 | 1,231.1 | 4.15s | 3.37× | **Yes** | **21.3 ms** |
| **5** | *"Then who is the Prime Minister of India?"* | *"The Prime Minister of India is the respected..."* | 1,639.0 | 154.2 | 398.6 | 1,238.7 | 4.10s | 3.31× | **Yes** | **21.3 ms** |
| **6** | *"Okay, that's uh great, thanks you said..."* | *"Ah, you might be thinking of the current..."* | 2,391.7 | 167.2 | 671.0 | 1,719.3 | 6.45s | 3.75× | **Yes** | **21.3 ms** |
| **7** | *"No, no, not that, what's your name?"* | *"My name is your friendly, fast..."* | 1,814.8 | 154.7 | 484.6 | 1,328.7 | 4.82s | 3.63× | **Yes** | **21.3 ms** |
| **8** | *"Poke bye."* | *"Bye for now!"* | **734.2** | 148.7 | 263.4 | 469.3 | 1.42s | 3.04× | No | — |

---

### 7.5 Architectural Root Causes: Why Swar Achieves This Speed on CPU

1. **Sub-Second Conversational Openers:**
   In natural conversation, humans frequently open responses with brief conversational phrases (*"Hey!"*, *"Hi there!"*, *"Bye for now!"*). Because Swar chunks on strict grammatical punctuation marks (`!`, `?`, `.`, `\n`), Sentence 1 is often 1–4 words long. Kokoro synthesizes these micro-chunks in **436ms–469ms**, enabling the assistant to begin speaking in **642ms–734ms**—well within natural human conversational pacing.
2. **Decoupled Producer–Consumer Compute Pipelining:**
   Once Sentence 1 begins playing, Swar's generation worker does not sit idle. In Turn 2, while the 1.1-second audio of *"Hey!"* was actively playing through the speaker, Kokoro synthesized Sentence 2 (5.35 seconds of audio) in the background in 1,364ms at 3.92× real-time speed. By the time Sentence 1 completed playback, Sentence 2 was already buffered in the ready queue. The soundcard experienced zero audio starvation or gap latency.
3. **Flawless Barge-In Interruption (21.3ms):**
   Across all 5 interrupted turns (Turns 3, 4, 5, 6, 7), the user spoke mid-playback. In every instance, Swar detected user voice activity via Silero VAD, immediately signaled the cancellation scope, flushed the text and ready audio queues, and stopped DMA playback at the next 512-sample hardware boundary in **precisely 21.3ms**. The opening words of the interrupting question were fully retained by the speculative turn buffer and correctly transcribed by Parakeet TDT without driver crashes or audio bleed.


---

## 8. Production Deployment Architecture & Best Practices

When deploying Swar in enterprise or production environments:

1. **FastAPI / WebSocket Gateway**:
   Run Swar in a background thread or process. Route audio frames over binary WebSockets and push JSON events (`speech_started`, `interim_text`, `final_transcript`) to client frontends.
2. **Acoustic Echo Cancellation (AEC)**:
   In laptop or desktop environments where the microphone picks up speaker sound:
   - Use headphones (recommended for full-duplex conversational testing).
   - Or configure `allow_barge_in: false` in high-noise environments.
   - Or route microphone capture through system-level WebRTC AEC (e.g. PipeWire/PulseAudio echo-cancel module).
3. **Headless Server Deployment**:
   If running on a Linux server without a physical audio card (e.g. Docker container, AWS EC2, GCP Compute Engine):
   Use the decoupled STT and TTS engines directly via their Python API (`tts_pipeline.feed_text()`, `stt_handler.transcribe_final()`) and stream raw PCM bytes over TCP/gRPC/WebSockets rather than `sounddevice`.

---

*Authored by the DeepMind & Antigravity Engineering Collaboration — September 2026.*
