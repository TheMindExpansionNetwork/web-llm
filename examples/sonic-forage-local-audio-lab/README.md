# Sonic-Forage Local Audio Lab

Runnable browser-local proof for the WebLLM fork.

- WebLLM = local prompt/arrangement brain.
- Web Audio API = local music/SFX rendering, loop playback, and crossfade.
- ONNX Runtime Web / Transformers.js = future local voice and tiny SFX model lanes.
- No Modal/API endpoint required for the core slice.

## Run

```bash
cd examples/sonic-forage-local-audio-lab
npm install
npm run build
npm start
```

Open `http://127.0.0.1:8891` locally. For iPhone, deploy `lib/` over HTTPS or expose the dev server with an HTTPS tunnel.

## Smoke without downloading a model

Click **Render local Web Audio plan**. This starts the procedural endless local loop and SFX without WebLLM.

## WebLLM smoke

Click **Load WebLLM + plan**. The browser downloads and caches the selected WebLLM model, then asks for strict JSON and renders it locally. First target is `Llama-3.2-1B-Instruct-q4f16_1-MLC`.

## Why this is not full Stable Audio 3 yet

Stable Audio 3 checkpoints are too large and not currently packaged as browser WebGPU artifacts. The local iPhone lane starts with WebLLM + Web Audio + small browser TTS/SFX models while SA3 remains a server/Modal or desktop lane until converted/distilled artifacts are proven.
