---
name: webllm-local-audio-browser
description: Use when building local-first browser AI on mlc-ai/web-llm for iPhone/iPad Safari/WebGPU, especially Sonic-Forage music plans, Web Audio crossfades, local SFX, ONNX/Transformers.js TTS, and no-endpoint cached PWAs.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [webllm, webgpu, iphone, pwa, web-audio, onnx, tts, sfx, sonic-forage]
    related_skills: [iphone-webllm-browser-lane, stable-audio3-mindexpander-engine]
---

# WebLLM Local Audio Browser

## Overview

Use this skill for the corrected local-first browser lane based on `mlc-ai/web-llm`. WebLLM is the local in-browser LLM/prompt brain. Music/SFX/voice generation must be layered with Web Audio API and browser ML runtimes such as ONNX Runtime Web or Transformers.js.

The intended product is a Safari/Home Screen web app for iPhone 17 Pro that downloads model assets once, caches them, then runs locally without Modal/API endpoints.

## Key Decision

Do not claim full Stable Audio 3 runs locally in iPhone Safari until there is a verified browser artifact. Stable Audio 3 is currently a Python/PyTorch multi-GB diffusion lane. Browser-local SA3 requires conversion/distillation and real iPhone memory/perf proof.

Use this practical split:

- WebLLM: local prompt brain, JSON arrangement plans, labels, routing.
- Web Audio API: immediate local music rendering, loops, crossfades, procedural SFX.
- ONNX Runtime Web / Transformers.js: small local TTS and possibly tiny SFX/audio models.
- Stable Audio 3: Modal/server/desktop lane until browser artifacts are proven.

## First Implementation Slice

1. Fork/sync `mlc-ai/web-llm` to `TheMindExpansionNetwork/web-llm`.
2. Add docs under `docs/LOCAL_FIRST_AUDIO_ON_IPHONE.md`.
3. Build a PWA example that checks `navigator.gpu`, storage estimate, and cache state.
4. Load `Llama-3.2-1B-Instruct-q4f16_1-MLC` first.
5. Ask WebLLM for strict JSON:
   ```json
   {"bpm":92,"mood":"deep echo","layers":["drone","pluck"],"next_variation":"raise shimmer"}
   ```
6. Render JSON with Web Audio; keep current loop playing while next plan renders.
7. Add procedural SFX buttons before ML SFX.
8. Use browser SpeechSynthesis as voice fallback; swap to ONNX TTS only after phone smoke.

## ONNX / “Onyx” Voice Lane

When the user says Onyx/ONNX small model, interpret it as the BlueTTS/on-device small ONNX lane unless they explicitly ask for Qwen/OmniVoice. Candidate browser-local TTS paths:

- Kokoro.js / Transformers.js;
- Piper or KittenTTS browser ports;
- Liquid LFM2.5-Audio ONNX demo as proof of ASR/TTS feasibility;
- native browser SpeechSynthesis as a zero-model fallback.

## Cache Requirements

- Serve over HTTPS.
- Prefer WebLLM cache backend; upstream supports OPFS in recent commits.
- Show `initProgressCallback` text.
- Show storage estimate via `navigator.storage.estimate()`.
- Keep the tab foregrounded during first download.
- Verify refresh does not redownload everything.
- Test Add-to-Home-Screen after successful browser load.

## Verification Checklist

- [ ] Fork exists and is synced with upstream.
- [ ] iPhone WebGPU confirmed at `webgpureport.org`.
- [ ] WebLLM 1B q4f16 initializes and streams a JSON plan.
- [ ] Web Audio endless loop plays locally with no endpoint calls.
- [ ] Procedural SFX works offline after load.
- [ ] Voice fallback works; ONNX TTS is separately smoke-tested before claiming.
- [ ] Refresh/cache behavior verified.
- [ ] Any SA3 claim says server/Modal unless a browser artifact has been proven.
