# Sonic-Forage Director v0 — synthetic from-scratch Modal training receipt

Date: 2026-05-25

## What was trained

A tiny realtime **audio-director** model for the browser-local Sonic-Forage lab.

- Model ID: `sonic_forage_director_v0`
- Type: bag-of-words MLP trained from random initialization, not a fine-tune
- Purpose: map short voice/jam commands into structured Web Audio controls
- Browser target: ONNX Runtime Web / local inference helper for the existing Web Audio loop engine
- Not a waveform generator and not Stable Audio 3 in-browser

## Why this dataset/model

For the realtime lane, the immediate bottleneck is not generating full audio from scratch inside Safari. The useful first trainable model is a tiny controller that can respond instantly to commands like:

- `deepen the healing signal with warm drone and tape hiss`
- `drop at the next bar, urgent rave, add portal riser`
- `calm the loop, minimal slow, less clutter`

The Web Audio engine can then apply the predicted controls immediately while heavier WebLLM/Modal/ACE-Step lanes generate richer phrases in the background.

## Dataset

Local dataset folder:

`datasets/sonic_forage_director_v0/`

Rows:

- Train: 3444
- Validation: 378
- Test: 378
- Total: 4200

Schema:

- `input`: short natural-language jam/control prompt
- `target.bpm`
- `target.brightness`
- `target.pulse`
- `target.bass`
- `target.noise`
- `target.tension`
- `target.action`
- `target.mood`
- `target.layers[]`
- `target.next_variation`

Synthetic generator:

`scripts/generate_sonic_forage_director_dataset.py`

## Modal training run

Command:

```bash
modal run scripts/train_sonic_forage_director_modal.py --epochs 90
```

Modal app:

`sonic-forage-director-v0-train`

Latest run URL:

`https://modal.com/apps/m1ndb0t-2045/main/ap-AhTFF36DWEQpNj3nlbthsy`

Training notes:

- CPU Modal function, no orphan GPU spend.
- PyTorch model initialized from scratch.
- Exported both PyTorch state dict and ONNX.
- ONNX checker verified locally with `uvx --from onnx python ...`.

## Metrics

Test split:

- Rows: 378
- Action accuracy: 1.0000
- Mood accuracy: 1.0000
- BPM MAE: 7.334 BPM
- Control MAE: 0.0336
- Layer micro-F1: 0.9161
- Loss: 0.22177

Validation split:

- Rows: 378
- Action accuracy: 1.0000
- Mood accuracy: 0.9974
- BPM MAE: 8.066 BPM
- Control MAE: 0.0326
- Layer micro-F1: 0.9084
- Loss: 0.24682

## Artifacts

Folder:

`artifacts/sonic_forage_director_v0/`

Files:

- `director_model.onnx` — 95,950 bytes, SHA256 `69523e3ba7bd6b23f12f5cae77c11a255f91cd56f5d3cdd5b4056a26ea726bc9`
- `director_model.pt` — 98,226 bytes, SHA256 `f6ceb989f3c5e6fe265e5d5bfc17013cd9068b95ce57513d0c12b650aee0054c`
- `director_feature_spec.json` — 2,326 bytes, SHA256 `32644771dbda1f19609bb81b8bf75f859ab00b2c0ecc7036d285d60cb8447698`
- `eval_predictions.json` — 10,464 bytes, SHA256 `a7b3dd858764dcfc980b38e10940114b528df9318084b29cf073f38bc2f8d408`
- `training_receipt.json` — 3,352 bytes, SHA256 `8cdee51dd3d605e5aac9335ccad6afc92cb038d41e033ec686c7c33a7c50210f`

## Integration path

Next browser integration should:

1. Add `onnxruntime-web` to the local lab package.
2. Load `artifacts/sonic_forage_director_v0/director_model.onnx` and `director_feature_spec.json`.
3. Convert user prompt into the feature vector using the spec vocab.
4. Run local ONNX inference.
5. Convert outputs into the existing `MusicPlan` shape:
   - `bpm = 64 + bpm_norm * 86`
   - `brightness`, `pulse` direct from controls
   - optional `bass/noise/tension` can expand the Web Audio renderer
   - action/mood/layers from logits
6. Keep WebLLM as the heavier creative planner; use this model as the instant realtime controller.

## Caveats

- This is synthetic supervision, so it proves the train/deploy/integration lane and fast controller behavior, not human musical preference yet.
- It does not generate audio waveforms; it controls the local Web Audio synthesizer/looper.
- The next quality jump should come from collecting real operator commands + accepted parameter states during live sessions, then mixing those with the synthetic dataset.
