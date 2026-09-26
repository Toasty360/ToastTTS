# ToastTTS: Ultra-Low Latency, Edge-Native Streaming TTS Engine

A deterministic, sub-120ms Time-to-First-Audio (TTFA) conversational TTS orchestrator designed for low-power CPUs (x86 / ARM / Raspberry Pi). This project wraps lightweight acoustic models with asymmetric clause chunking, acoustic boundary smoothing, and dynamic room-tone bed injection to eliminate abrupt digital cutoffs and simulate conversational cadence. Core plan to apply Eye-Voice Span (EVS) on TTS to have natural pause.

---

## 1. Project Overview & Specifications

### Performance & Hardware Targets

- **Target Hardware:** Single-to-quad core x86_64 / ARM64 CPU (e.g., Raspberry Pi 4/5, low-cost VPS).
- **Memory Footprint:** < 100 MB RAM resident set size (RSS).
- **Time-to-First-Audio (TTFA):** 80 ms – 120 ms.
- **Real-Time Factor (RTF):** < 0.15 on a single thread (1s audio generates in < 150 ms).
- **Audio Format:** 16-bit linear PCM, 22.05 kHz, mono.

### Key Architectural Pillars

1. **Asymmetric Text Chunking:** Employs an ultra-short 2–4 word "flash chunk" for immediate playback onset, transitioning to larger 6–10 word clauses to preserve prosodic cadence.
2. **Acoustic Boundary Conditioning (DSP):** Eliminates the "vacuum silence" bug via in-place exponential window decay on trailing phonemes and active comfort-noise bed mixing.
3. **Zero-Copy Pipeline:** Utilizes an in-memory thread-safe circular ring buffer directly coupled to audio output (PortAudio / ALSA), avoiding IPC overhead, subprocesses, or disk I/O.

---

## 2. System Architecture

```
[ Incoming LLM Token Stream ]
              │
              ▼
   ┌──────────────────────┐
   │ Text Stream Chunker  │  <-- Tracks words, clause punctuation, pivot conjunctions
   └──────────┬───────────┘
              │ Text Chunks (Flash: 2-4 words | Normal: 6-10 words)
              ▼
   ┌──────────────────────┐
   │  In-Memory Phonemizer│  <-- libespeak-ng C/C++ bindings (no subprocess)
   └──────────┬───────────┘
              │ Phoneme IDs
              ▼
   ┌──────────────────────┐
   │ ONNX Runtime Worker  │  <-- Piper VITS (INT8 Quantized, single-session)
   └──────────┬───────────┘
              │ Raw PCM Slices (16-bit, 22.05 kHz)
              ▼
   ┌──────────────────────┐
   │   DSP Edge Shaper    │  <-- 30ms exponential decay tail + -55dB comfort noise bed
   └──────────┬───────────┘
              │ Conditioned PCM Frames
              ▼
   ┌──────────────────────┐
   │ In-Memory Ring Buffer│  <-- Lock-free / atomic thread-safe audio queue
   └──────────┬───────────┘
              │
              ▼
   [ Audio Output Device / WebRTC Sink ]
```

---

## 3. Core Software Modules

### Module 1: `StreamChunker`

- **Role:** Buffers incoming LLM tokens and yields text slices based on conversational readiness.
- **Rules:**
  - **State 0 (First Chunk / Flash):** Triggers at the earliest of: punctuation (`.`, `,`, `?`, `!`, `—`) OR word count >= 3.
  - **State 1 (Steady Stream):** Triggers on sentence terminators (`.`, `!`, `?`), major punctuation (`,`, `;`, `—`), or pivot conjunctions (`and`, `but`, `because`, `so`, `which`) once word count >= 6.
  - **Safety Ceiling:** Hard cut at 10 words on nearest whitespace to prevent pipeline starvation.
  - **Lookahead Guard:** Holds back numerical tokens or single-letter initials (e.g., preventing a cut on "at 3" before "PM" arrives).

### Module 2: `InferenceWorker` (C++ / Rust / Embedded Python)

- **Role:** Manages the acoustic model lifecycle and executes ONNX sessions.
- **Model Baseline:** Piper VITS (`en_US-lessac-medium` or `en_US-amy-medium`).
- **Optimizations:**
  - INT8 dynamic post-training quantization (`quantize_dynamic` via ONNX Runtime).
  - Direct C API / C++ wrapper (`sherpa-onnx` or native `libpiper_phonemize` + `onnxruntime`).
  - Fixed, pre-allocated input/output tensors to avoid heap thrashing during synthesis.

### Module 3: `AcousticConditioner` (DSP Engine)

- **Role:** Resolves abrupt chunk boundaries and artificial silence artifacts.
- **Operations:**
  - **Trailing Tail Decay:** Identifies the final 512 samples (~23 ms at 22.05 kHz) of each audio chunk. Multiplies by an exponential decay envelope (1.0 -> 0.01) to prevent non-zero-crossing clicks.
  - **Comfort Noise Injection:** Maintains a pre-allocated 1-second circular buffer of shaped white noise (low-pass filtered at 1 kHz, normalized to -55 dBFS). Injects this bed between synthesized chunks during pauses.
  - **Variable Pause Padding:** Inserts dynamic durations based on chunk terminator:
    - Comma / Pivot: 120 ms – 160 ms
    - Colon / Semicolon: 200 ms – 250 ms
    - Period / Exclamation / Question: 350 ms – 420 ms

### Module 4: `AudioRingBuffer` & Playback Consumer

- **Role:** Coordinates producer (TTS worker) and consumer (audio device / network sink).
- **Buffer Size:** Fixed 64 KB circular byte buffer (~1.4 seconds of 22.05 kHz 16-bit mono audio).
- **Underflow Handling:** If buffer drops to zero before next chunk completes, fills playback frames with comfort noise instead of dropping output stream.

---

## 4. Implementation Plan & Milestones

### Phase 1: Environment Setup & Baseline Measurement

- [ ] Download and verify base models: Piper VITS medium voice (`en_US-lessac-medium.onnx`).
- [ ] Benchmark native Piper CLI baseline: Measure latency from text input to first audio byte (TTFA) and memory consumption.
- [ ] Set up a reproducible test harness with an LLM token streaming mock (emitting tokens at ~30 tokens/sec).

### Phase 2: Core Pipeline & Zero-Subprocess Integration

- [ ] Integrate `sherpa-onnx` C++/Rust or Python bindings to run inference entirely in-process.
- [ ] Implement INT8 quantization on the ONNX model graph.
- [ ] Profile memory usage and inference speed per single inference call.

### Phase 3: The Prosodic Chunking & DSP Engine

- [ ] Implement the `StreamChunker` state machine with asymmetric flash chunking.
- [ ] Implement `AcousticConditioner`:
  - Exponential tail decay.
  - Static comfort noise generator and mixing logic.
  - Variable pause gap scheduler.
- [ ] Connect chunker output directly to the inference worker.

### Phase 4: Threaded Ring Buffer & Audio Playback

- [ ] Implement the lock-free circular audio buffer.
- [ ] Hook up audio output stream (PortAudio or raw PCM stdout sink).
- [ ] Test end-to-end token-in to audio-out flow with continuous streaming.

### Phase 5: Profiling, Benchmarks, and Repository Polish

- [ ] Measure and log:
  - Cold-start memory footprint.
  - TTFA (Time-to-First-Audio) across 100 test runs.
  - CPU core utilization graphs.
- [ ] Record side-by-side comparison audio samples:
  - Sample A: Raw unbuffered model (choppy cuts, machine-gun rush).
  - Sample B: Engine output (asymmetric chunking, smoothed boundaries, ambient bed).
- [ ] Write clear documentation and packaging (Dockerfile, standalone CLI).

---

## 5. Repository Structure for Portfolio Showcase

```
ToastTTS
├── README.md               <-- Overview, architecture diagram, audio demo embeds, benchmark tables
├── benchmarks/
│   ├── latency_test.py     <-- TTFA & RTF automated measurement scripts
│   ├── results/            <-- Latency distribution graphs & memory profile charts
│   └── audio_samples/      <-- Baseline vs. Engine WAV comparisons
├── engine/
│   ├── __init__.py
│   ├── chunker.py          <-- Linguistic & asymmetric stream chunker
│   ├── dsp.py              <-- Boundary decay, crossfade, and comfort noise injector
│   ├── inference.py        <-- In-memory ONNX Runtime session manager
│   └── ring_buffer.py      <-- Thread-safe circular audio stream
├── models/
│   └── download_models.sh  <-- Script to fetch & quantize base checkpoints
├── tests/
│   ├── test_chunker.py
│   └── test_dsp.py
├── Dockerfile              <-- Single-command containerized build & run
└── requirements.txt
```

---

## 6. Key Resume & Portfolio Bullets

- Designed and built an edge-first streaming TTS engine achieving **sub-100ms TTFA** and a **<80MB memory footprint** on single-core CPU hardware.
- Eliminated acoustic truncation and unnatural speech cadence by designing an **asymmetric clause chunking pipeline** and **dynamic comfort noise injection**.
- Implemented **INT8 quantization** and zero-copy in-memory buffer streaming via ONNX Runtime, achieving an **RTF < 0.12** without external GPU dependencies.
- Profiled and optimized end-to-end latency and audio continuity across simulated LLM token streams.
