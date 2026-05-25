#!/usr/bin/env python3
"""Train the Sonic-Forage tiny realtime director model on Modal.

This is a from-scratch supervised model, not a fine-tune: a compact MLP starts
from random weights and learns synthetic prompt/control -> Web Audio controls.
The exported ONNX is small enough for a browser/ONNX Runtime Web integration.
"""
from __future__ import annotations

import base64
import io
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "sonic-forage-director-v0-train"
DATASET_DIR = Path("datasets/sonic_forage_director_v0")
ARTIFACT_DIR = Path("artifacts/sonic_forage_director_v0")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "onnx==1.16.1", "numpy==1.26.4")
)

STOPWORDS = {"the", "and", "with", "now", "please", "make", "keep", "more", "less", "next", "bar", "loop", "signal", "voice", "command", "forage", "sonic"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@app.function(image=image, timeout=900, cpu=4, memory=4096)
def train_remote(dataset_payload: dict[str, str], epochs: int = 90, seed: int = 424242) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    splits: dict[str, list[dict[str, Any]]] = {}
    for split, b64 in dataset_payload.items():
        text = base64.b64decode(b64.encode()).decode("utf-8")
        splits[split] = [json.loads(line) for line in text.splitlines() if line.strip()]

    actions = sorted({r["target"]["action"] for rows in splits.values() for r in rows})
    moods = sorted({r["target"]["mood"] for rows in splits.values() for r in rows})
    layers = sorted({layer for rows in splits.values() for r in rows for layer in r["target"]["layers"]})
    vocab_counts: dict[str, int] = {}
    for row in splits["train"]:
        for tok in tokenize(row["input"]):
            vocab_counts[tok] = vocab_counts.get(tok, 0) + 1
    vocab = [tok for tok, count in sorted(vocab_counts.items(), key=lambda kv: (-kv[1], kv[0])) if count >= 2][:96]
    tok_to_i = {t: i for i, t in enumerate(vocab)}
    action_to_i = {a: i for i, a in enumerate(actions)}
    mood_to_i = {m: i for i, m in enumerate(moods)}
    layer_to_i = {l: i for i, l in enumerate(layers)}

    def featurize(text: str) -> np.ndarray:
        x = np.zeros(len(vocab), dtype=np.float32)
        toks = tokenize(text)
        for tok in toks:
            if tok in tok_to_i:
                x[tok_to_i[tok]] = min(1.0, x[tok_to_i[tok]] + 0.5)
        return x

    def targets(row: dict[str, Any]):
        t = row["target"]
        regs = np.array([
            (float(t["bpm"]) - 64.0) / (150.0 - 64.0),
            float(t["brightness"]), float(t["pulse"]), float(t["bass"]), float(t["noise"]), float(t["tension"]),
        ], dtype=np.float32)
        layer_vec = np.zeros(len(layers), dtype=np.float32)
        for layer in t["layers"]:
            layer_vec[layer_to_i[layer]] = 1.0
        return regs, action_to_i[t["action"]], mood_to_i[t["mood"]], layer_vec

    def make_arrays(rows: list[dict[str, Any]]):
        X = np.stack([featurize(r["input"]) for r in rows])
        regs, act, mood, lay = zip(*[targets(r) for r in rows])
        return (
            torch.tensor(X),
            torch.tensor(np.stack(regs)),
            torch.tensor(act, dtype=torch.long),
            torch.tensor(mood, dtype=torch.long),
            torch.tensor(np.stack(lay)),
        )

    train = make_arrays(splits["train"])
    val = make_arrays(splits["validation"])
    test = make_arrays(splits["test"])

    class DirectorNet(nn.Module):
        def __init__(self, input_dim: int, n_actions: int, n_moods: int, n_layers: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.08),
                nn.Linear(128, 96), nn.ReLU(),
            )
            self.reg = nn.Linear(96, 6)
            self.action = nn.Linear(96, n_actions)
            self.mood = nn.Linear(96, n_moods)
            self.layers = nn.Linear(96, n_layers)

        def forward(self, x):
            h = self.net(x)
            return self.reg(h), self.action(h), self.mood(h), self.layers(h)

    model = DirectorNet(len(vocab), len(actions), len(moods), len(layers))
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    bs = 128
    start = time.time()

    def loss_for(batch):
        x, yreg, yact, ymood, ylay = batch
        preg, pact, pmood, play = model(x)
        return (
            F.mse_loss(torch.sigmoid(preg), yreg) * 3.0
            + F.cross_entropy(pact, yact)
            + F.cross_entropy(pmood, ymood)
            + F.binary_cross_entropy_with_logits(play, ylay)
        )

    train_n = train[0].shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_n)
        total = 0.0
        for s in range(0, train_n, bs):
            idx = perm[s:s + bs]
            batch = tuple(t[idx] for t in train)
            loss = loss_for(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        if epoch == 1 or epoch % 15 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                vloss = float(loss_for(val).item())
            print(f"epoch={epoch:03d} train_loss={total/train_n:.4f} val_loss={vloss:.4f}", flush=True)

    def evaluate(split_name: str, data):
        model.eval()
        x, yreg, yact, ymood, ylay = data
        with torch.no_grad():
            preg, pact, pmood, play = model(x)
            preg = torch.sigmoid(preg)
            action_acc = (pact.argmax(1) == yact).float().mean().item()
            mood_acc = (pmood.argmax(1) == ymood).float().mean().item()
            bpm_mae = (torch.abs((preg[:, 0] - yreg[:, 0]) * 86.0)).mean().item()
            control_mae = torch.abs(preg[:, 1:] - yreg[:, 1:]).mean().item()
            layer_pred = (torch.sigmoid(play) > 0.45).float()
            layer_f1_num = (2 * (layer_pred * ylay).sum()).item()
            layer_f1_den = (layer_pred.sum() + ylay.sum()).item() + 1e-8
            return {
                "rows": int(x.shape[0]),
                "loss": float(loss_for(data).item()),
                "action_acc": round(action_acc, 4),
                "mood_acc": round(mood_acc, 4),
                "bpm_mae": round(bpm_mae, 3),
                "control_mae": round(control_mae, 4),
                "layer_micro_f1": round(layer_f1_num / layer_f1_den, 4),
            }

    metrics = {"train": evaluate("train", train), "validation": evaluate("validation", val), "test": evaluate("test", test)}

    # Package artifacts.
    model.eval()
    buf_pt = io.BytesIO()
    torch.save(model.state_dict(), buf_pt)
    dummy = torch.zeros(1, len(vocab), dtype=torch.float32)
    buf_onnx = io.BytesIO()
    torch.onnx.export(
        model, dummy, buf_onnx,
        input_names=["features"],
        output_names=["controls", "action_logits", "mood_logits", "layer_logits"],
        dynamic_axes={"features": {0: "batch"}, "controls": {0: "batch"}, "action_logits": {0: "batch"}, "mood_logits": {0: "batch"}, "layer_logits": {0: "batch"}},
        opset_version=17,
    )

    spec = {
        "model_id": "sonic_forage_director_v0",
        "model_type": "bag_of_words_mlp_from_scratch",
        "input_dim": len(vocab),
        "controls": ["bpm_norm", "brightness", "pulse", "bass", "noise", "tension"],
        "bpm_range": [64, 150],
        "vocab": vocab,
        "actions": actions,
        "moods": moods,
        "layers": layers,
        "thresholds": {"layer_sigmoid": 0.45},
        "training": {"epochs": epochs, "seed": seed, "duration_sec": round(time.time() - start, 2)},
        "metrics": metrics,
    }

    examples = []
    for row in splits["test"][:12]:
        x = torch.tensor(featurize(row["input"])).unsqueeze(0)
        with torch.no_grad():
            r, a, m, l = model(x)
            r = torch.sigmoid(r)[0].numpy().tolist()
            ls = torch.sigmoid(l)[0].numpy().tolist()
        pred_layers = [layers[i] for i, p in enumerate(ls) if p > 0.45][:5]
        examples.append({
            "input": row["input"],
            "target": row["target"],
            "prediction": {
                "bpm": round(64 + r[0] * 86),
                "brightness": round(r[1], 3),
                "pulse": round(r[2], 3),
                "bass": round(r[3], 3),
                "noise": round(r[4], 3),
                "tension": round(r[5], 3),
                "action": actions[int(torch.argmax(a, dim=1)[0])],
                "mood": moods[int(torch.argmax(m, dim=1)[0])],
                "layers": pred_layers,
            },
        })

    def enc(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    return {
        "ok": True,
        "app_name": APP_NAME,
        "metrics": metrics,
        "artifacts": {
            "director_model.pt": enc(buf_pt.getvalue()),
            "director_model.onnx": enc(buf_onnx.getvalue()),
            "director_feature_spec.json": enc(json.dumps(spec, indent=2, sort_keys=True).encode()),
            "eval_predictions.json": enc(json.dumps(examples, indent=2, sort_keys=True).encode()),
            "training_receipt.json": enc(json.dumps({"ok": True, "app_name": APP_NAME, "metrics": metrics, "spec": spec}, indent=2, sort_keys=True).encode()),
        },
    }


@app.local_entrypoint()
def main(epochs: int = 90):
    if not DATASET_DIR.exists():
        raise SystemExit(f"Dataset not found: {DATASET_DIR}. Run scripts/generate_sonic_forage_director_dataset.py first.")
    payload = {}
    for split in ["train", "validation", "test"]:
        payload[split] = base64.b64encode((DATASET_DIR / f"{split}.jsonl").read_bytes()).decode("ascii")
    result = train_remote.remote(payload, epochs=epochs)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, b64 in result["artifacts"].items():
        (ARTIFACT_DIR / name).write_bytes(base64.b64decode(b64.encode("ascii")))
    print(json.dumps({"ok": result["ok"], "artifact_dir": str(ARTIFACT_DIR), "metrics": result["metrics"]}, indent=2, sort_keys=True))
