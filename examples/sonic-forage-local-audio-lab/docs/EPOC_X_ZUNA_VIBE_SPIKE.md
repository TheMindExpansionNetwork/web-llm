# EPOC X + ZUNA + Sonic Forage vibe-control spike

## Goal

Use the Emotiv EPOC X as a playful BCI controller for Wolf/Sonic-Forage music: headset signal -> normalized vibe controls -> Web Audio plan changes. This is for creative/research use only, not diagnosis or medical feedback.

## What I found

### Emotiv EPOC X / Cortex

- Cortex API is JSON-RPC over WebSocket, normally at `wss://localhost:6868`.
- EPOC X is supported by Cortex over USB dongle or BLE 4.0.
- Data streams include `eeg`, `mot`, `dev`, `eq`, `pow`, `met`, `com`, `fac`, `sys`.
- For EPOC X, free/basic access covers BCI-style streams; raw EEG is a premium Developer API scope.
- Useful first streams for music control:
  - `met`: performance metrics; 2 Hz with `pm` scope, otherwise low-res 0.1 Hz.
  - `pow`: band powers; 8 Hz, good enough for vibe mapping.
  - `com`: mental commands; 8 Hz after profile/training.
  - `eeg`: raw 14-channel EPOC X data; requires `eeg` license scope, 128/256 Hz.
- EPOC X EEG column order documented by Cortex:
  - `COUNTER`, `INTERPOLATED`, `AF3`, `F7`, `F3`, `FC5`, `T7`, `P7`, `O1`, `O2`, `P8`, `T8`, `FC6`, `F4`, `F8`, `AF4`, `RAW_CQ`, `MARKER_HARDWARE`, `MARKERS`.

### ZUNA

- ZUNA is a 380M-parameter masked diffusion autoencoder for scalp EEG denoising/reconstruction/upsampling.
- It expects `.fif` inputs with montage/3D channel positions.
- Its preprocessing is fixed around 256 Hz, 5-second epochs, normalization, and `.fif -> .pt -> .fif` pipeline.
- Best use for this project: offline/nearline cleanup and superresolution of recorded EPOC sessions, not direct browser-local realtime control yet.
- EPOC X's 14 channels are a plausible source montage for ZUNA experiments if exported/converted to MNE `.fif` with correct channel names/positions.

## Spike implemented now

Added a testable browser lane before touching real headset credentials:

- Browser UI section: `EPOC X / BCI vibe bridge`.
- Built-in browser simulator button: `Simulate BCI vibe`.
- Local WebSocket bridge input/button: `ws://127.0.0.1:8765`.
- Mapping schema:
  - `focus`, `excitement`, `relaxation`, `stress`, `engagement`, `alpha`, `beta`, `theta`, `gamma`, `quality` in normalized 0-1 range.
- Audio mapping:
  - excitement -> BPM/pulse/drop energy
  - relaxation/theta -> bass/warm drift
  - focus/engagement -> brightness/glassy plucks/locked-in mode
  - stress/gamma -> noise/tension/radio static/calm action

## Run the no-hardware test

From this example directory:

```bash
uv run --with websockets scripts/emotiv_bci_vibe_bridge.py --simulate
```

Then open the lab and click:

1. `Connect BCI bridge`, or
2. `Simulate BCI vibe` directly in the browser.

## Real EPOC X connection checklist

1. Install/open EMOTIV Launcher/Cortex on the machine with the headset.
2. Pair EPOC X and confirm contact quality.
3. Register a Cortex app and get client id/secret.
4. Confirm available license scopes:
   - `pow`/`met`/`com` first if raw EEG is unavailable.
   - `eeg` if premium raw EEG access is active.
5. Use EMOTIV's official `Emotiv/cortex-v2-example` Python client to authorize, create a session, and subscribe.
6. Pipe callbacks through `scripts/emotiv_bci_vibe_bridge.py::map_cortex_sample()` and broadcast the normalized vibe JSON to the browser.
7. Record a short session to `.fif` with MNE montage for ZUNA cleanup tests.

## ZUNA test plan after real capture

1. Export/convert an EPOC X recording to MNE `.fif`.
2. Set channel names exactly to the EPOC X list and apply a standard montage.
3. Run ZUNA preprocessing/inference/pt_to_fif on 5-second chunks.
4. Compare raw vs denoised band-power stability.
5. If denoised features are smoother, train a tiny controller like the current ONNX director on `(EEG/band features -> MusicPlan)` pairs.

## Verdict

PARTIAL / READY FOR HARDWARE SMOKE.

- The web/audio side can now react to normalized BCI-vibe samples.
- The no-hardware bridge can generate test packets for end-to-end music control.
- Real Cortex connection still needs headset + EMOTIV app credentials/license on the host machine.
- ZUNA is promising for cleanup/superresolution after recording, but not the first realtime loop dependency.
