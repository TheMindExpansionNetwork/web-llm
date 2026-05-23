# Sonic-Forage Local Audio Lab run receipt — 2026-05-23

## Goal

Run the corrected `mlc-ai/web-llm` browser-local lane as a first Sonic-Forage audio lab:

- WebLLM as local prompt/arrangement brain.
- Web Audio API as local music/SFX renderer.
- No Modal/API endpoint required for the fallback/core loop.
- ONNX/Transformers.js TTS/SFX remains the next model lane after iPhone smoke.

## Local path

`/opt/data/workspace/github-forks/web-llm/examples/sonic-forage-local-audio-lab`

## Commands run

```bash
cd /opt/data/workspace/github-forks/web-llm/examples/sonic-forage-local-audio-lab
npm install
npm run build
npm start
```

## Verified locally

- `npm install`: passed, 0 vulnerabilities.
- `npm run build`: passed with Parcel.
- Dev server: `http://127.0.0.1:8891/` returned HTTP 200.
- Browser page loaded with title `Sonic Forage Local Audio Lab`.
- Compatibility panel showed WebGPU available in the test browser and storage quota visible.
- Clicked/rendered the no-download path via `Render local Web Audio plan`.
- Web Audio status changed to `playing local deck 1, crossfade 3.5s`.
- Plan JSON rendered from the Sonic-Forage prompt.
- Local SFX button produced log entry `Played local SFX: chirp`.
- Browser console had 0 JavaScript errors.

## What was intentionally not run yet

The full WebLLM model download/init was not forced in this automated run because the first selected model is a large browser cache download. The UI is wired to load `Llama-3.2-1B-Instruct-q4f16_1-MLC` and then request JSON music plans. On iPhone, this should be tested over HTTPS with the tab foregrounded.

## Next iPhone smoke

1. Deploy or tunnel the demo over HTTPS.
2. Open on iPhone 17 Pro Safari.
3. Confirm WebGPU at `https://webgpureport.org/`.
4. Click `Load WebLLM + plan`.
5. Verify first download completes and refresh uses browser cache/OPFS instead of fully redownloading.
