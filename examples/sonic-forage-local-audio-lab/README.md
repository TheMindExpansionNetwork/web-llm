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

## Tiny realtime director model

A first from-scratch synthetic controller model was trained on Modal for instant local command routing:

```bash
python3 scripts/generate_sonic_forage_director_dataset.py
modal run scripts/train_sonic_forage_director_modal.py --epochs 90
```

Artifacts are in `artifacts/sonic_forage_director_v0/`:

- `director_model.onnx` — tiny browser-target model for ONNX Runtime Web.
- `director_feature_spec.json` — vocab, labels, and output decoding.
- `training_receipt.json` — Modal metrics and receipt.

Report: `docs/SONIC_FORAGE_DIRECTOR_V0_MODAL_TRAINING.md`.

## Why this is not full Stable Audio 3 yet

Stable Audio 3 checkpoints are too large and not currently packaged as browser WebGPU artifacts. The local iPhone lane starts with WebLLM + Web Audio + small browser TTS/SFX models while SA3 remains a server/Modal or desktop lane until converted/distilled artifacts are proven.
