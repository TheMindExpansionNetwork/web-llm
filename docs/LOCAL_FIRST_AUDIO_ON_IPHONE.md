# Local-First Browser Audio Lane for iPhone 17 Pro

This note corrects the direction for Sonic-Forage browser-local work: the base repo to fork and extend is [`mlc-ai/web-llm`](https://github.com/mlc-ai/web-llm), not the xAI/Grok CLI. WebLLM gives us the local in-browser **LLM / prompt brain**. Audio generation needs adjacent browser runtimes.

## Target product

A link or Home Screen web app that runs on iPhone 17 Pro with no Modal/API endpoint after first load:

```text
Safari / Home Screen PWA
  ├─ WebLLM via WebGPU: local prompt brain, router, captions, JSON plans
  ├─ Web Audio API: playback, looping, crossfade, mixing
  ├─ ONNX Runtime Web / Transformers.js: small local TTS / SFX / audio models
  ├─ Cache/OPFS/IndexedDB: model files cached after first download
  └─ Optional backend only for heavyweight generation during development
```

## Reality check: Stable Audio 3 in iPhone browser

**Full Stable Audio 3 music generation is not the first local-iPhone browser target.** The small SA3 checkpoints are multi-GB diffusion models and the current Stability repo is Python/PyTorch-oriented. A browser-local SA3 path would require conversion/partitioning to ONNX/WebGPU or WebNN/MLX-style browser artifacts, custom schedulers, memory tests, and model-size reductions.

Practical decision:

- **Now:** use WebLLM to plan prompts/sections and Web Audio to play/loop/crossfade.
- **Now:** use small browser models for voice/TTS and maybe short SFX.
- **Near:** build a tiny local SFX/texture lane using ONNX Runtime Web or Transformers.js.
- **Later:** investigate distilled/converted audio diffusion models; only claim SA3 local when a real browser artifact loads and generates on-device.

## Music lane options

### Lane A — Web Audio procedural music, immediately local

Good for endless ambience, drones, pulses, binaural-ish beds, risers, stingers, and game-style reactive loops. No ML model download needed.

- Status: build now.
- Runtime: Web Audio API + AudioWorklet.
- Cache: normal app assets only.
- iPhone risk: low.

### Lane B — WebLLM as arrangement brain

Use WebLLM to turn user prompts into JSON music plans:

```json
{
  "bpm": 92,
  "mood": "deep echo healing signal",
  "layers": ["sub drone", "glassy pluck", "soft tape hiss"],
  "next_variation": "raise shimmer, keep bass minimal"
}
```

The Web Audio engine then renders the plan locally.

- Status: first useful WebLLM integration.
- Model: `Llama-3.2-1B-Instruct-q4f16_1-MLC` first.
- Cache: WebLLM cache backend; upstream now includes OPFS support.

### Lane C — local audio diffusion / Stable Audio-style generation

Use only after a small ONNX/WebGPU-compatible audio model is selected and measured. Do not promise SA3 on iPhone until proof exists.

Candidate research paths:

- ONNX Runtime Web WebGPU for small diffusion or transformer audio models.
- Hugging Face Transformers.js for supported audio/TTS models.
- Liquid AI `LFM2.5-Audio-1.5B-ONNX` as a browser-local ASR/TTS reference, not a music generator.
- Small TTS/SFX models such as Kokoro/Piper/KittenTTS-style browser ports.

## SFX lane

Start with short, local, low-memory sounds:

1. Procedural Web Audio SFX: whooshes, UI chirps, impacts, risers, drones.
2. Tiny sample pack cached in the PWA for known-good assets.
3. ONNX/Transformers.js SFX model only after model selection + browser smoke.
4. SA3-small-sfx remains a Modal/server or desktop lane until a browser artifact exists.

## Voice lane

The user’s “Onyx / ONNX small” preference maps to the BlueTTS/on-device small ONNX lane, not Qwen/OmniVoice by default.

Browser-local first candidates:

- native browser speech synthesis for fallback demos;
- Kokoro.js / small ONNX TTS for higher quality local browser speech;
- Piper/KittenTTS browser ports where license/runtime fit;
- Liquid LFM2.5-Audio browser demo as proof that ONNX Runtime Web can do local audio ASR/TTS.

Avoid promising custom MindExpander voice cloning in-browser until the actual ONNX voice artifact is selected, licensed, quantized, and smoke-tested on the phone.

## Cache/offline plan

- Serve over HTTPS; do not use `file://`.
- Use WebLLM built-in cache backend; prefer OPFS/Cache API where supported.
- Add a compatibility panel:
  - `navigator.gpu`
  - `navigator.storage.estimate()`
  - user-agent / iOS version hint
  - selected model sizes
- Add visible first-load progress.
- Keep the tab foregrounded while downloading models.
- Verify refresh does not fully redownload.
- Then test Add-to-Home-Screen launch.

## First implementation slice

1. WebLLM local 1B model loads and emits a JSON music plan.
2. Web Audio renders the plan into an endless crossfaded loop.
3. Procedural SFX buttons render locally.
4. Voice fallback speaks generated captions with browser SpeechSynthesis.
5. Later swap voice fallback for a small ONNX TTS model.

## Verification checklist

- [ ] Fork is synced with `mlc-ai/web-llm` upstream.
- [ ] iPhone shows WebGPU available at `https://webgpureport.org/`.
- [ ] Demo runs over HTTPS.
- [ ] WebLLM 1B q4f16 loads, initializes, and streams a JSON plan.
- [ ] Web Audio endless loop plays without endpoint calls.
- [ ] Refresh uses cached model files.
- [ ] SFX buttons work offline after load.
- [ ] Voice fallback works; ONNX TTS lane separately smoke-tested before replacing fallback.
