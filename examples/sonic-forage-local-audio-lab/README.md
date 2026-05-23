# Sonic-Forage Local Audio Lab

This is the first browser-local design target for the fork:

- WebLLM = local prompt/arrangement brain.
- Web Audio API = actual immediate music/SFX rendering, loop playback, and crossfade.
- ONNX Runtime Web / Transformers.js = future local voice and tiny SFX model lanes.
- No Modal/API endpoint required for the first slice.

## Why this is not full Stable Audio 3 yet

Stable Audio 3 checkpoints are too large and not currently packaged as browser WebGPU artifacts. The browser lane starts with local procedural music + small local TTS/SFX models, while SA3 remains a server/Modal or desktop lane until converted/distilled artifacts are proven.

## First target

Build a PWA that:

1. checks WebGPU/storage;
2. loads `Llama-3.2-1B-Instruct-q4f16_1-MLC` through WebLLM;
3. asks it for a strict JSON music plan;
4. renders that plan with Web Audio;
5. keeps an endless crossfade going;
6. adds procedural SFX buttons;
7. uses SpeechSynthesis as a temporary local voice fallback.

See `../../docs/LOCAL_FIRST_AUDIO_ON_IPHONE.md` for the full runbook.
