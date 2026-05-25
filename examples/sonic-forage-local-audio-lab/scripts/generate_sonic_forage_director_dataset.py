#!/usr/bin/env python3
"""Generate a synthetic Sonic-Forage realtime audio-director dataset.

The target model is intentionally tiny: it maps short jam/voice commands into
structured controls that the Web Audio engine can apply in <1 ms locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 424242

ACTIONS = ["deepen", "echo", "brighten", "calm", "drop", "freeze", "mutate", "save"]
MOODS = ["healing", "dark", "cosmic", "playful", "ritual", "dreamy", "urgent", "minimal"]
TEXTURES = ["warm drone", "glassy pluck", "tape hiss", "sub bass", "radio static", "alien chirp", "portal riser", "soft impact", "shimmer pad", "didgeridoo", "bell dust", "noise wash"]
TEMPOS = [72, 84, 92, 104, 118, 126, 138]
PROMPT_PATTERNS = [
    "{action} the {mood} signal with {texture_a} and {texture_b}, {tempo_word}, no vocals",
    "make it {mood}; {action} now; keep {texture_a}, add {texture_b}, bpm {bpm}",
    "voice command: {action} the loop, {mood} {tempo_word}, more {texture_a}, less clutter",
    "Sonic Forage jam: {mood} {texture_a}, {texture_b}, please {action} at the next bar",
    "{action} / {mood} / {tempo_word} / layers: {texture_a}, {texture_b}, {texture_c}",
]
TEMPO_WORDS = {
    "slow": 78,
    "half-time": 84,
    "steady": 92,
    "walking": 104,
    "fast": 126,
    "rave": 138,
}
ACTION_HINTS = {
    "deepen": {"brightness": -0.18, "pulse": 0.04, "bass": 0.24, "tension": 0.02},
    "echo": {"brightness": 0.08, "pulse": -0.03, "bass": -0.02, "tension": -0.02},
    "brighten": {"brightness": 0.28, "pulse": 0.04, "bass": -0.08, "tension": 0.02},
    "calm": {"brightness": -0.08, "pulse": -0.22, "bass": -0.02, "tension": -0.25},
    "drop": {"brightness": 0.10, "pulse": 0.28, "bass": 0.18, "tension": 0.20},
    "freeze": {"brightness": -0.03, "pulse": -0.35, "bass": -0.18, "tension": -0.15},
    "mutate": {"brightness": 0.12, "pulse": 0.11, "bass": 0.02, "tension": 0.18},
    "save": {"brightness": 0.0, "pulse": -0.05, "bass": 0.0, "tension": -0.05},
}
MOOD_HINTS = {
    "healing": {"brightness": 0.62, "pulse": 0.36, "bass": 0.42, "noise": 0.20, "tension": 0.18},
    "dark": {"brightness": 0.26, "pulse": 0.52, "bass": 0.72, "noise": 0.38, "tension": 0.58},
    "cosmic": {"brightness": 0.74, "pulse": 0.42, "bass": 0.45, "noise": 0.32, "tension": 0.35},
    "playful": {"brightness": 0.80, "pulse": 0.64, "bass": 0.34, "noise": 0.16, "tension": 0.22},
    "ritual": {"brightness": 0.42, "pulse": 0.48, "bass": 0.68, "noise": 0.28, "tension": 0.44},
    "dreamy": {"brightness": 0.66, "pulse": 0.30, "bass": 0.36, "noise": 0.22, "tension": 0.16},
    "urgent": {"brightness": 0.70, "pulse": 0.80, "bass": 0.55, "noise": 0.24, "tension": 0.72},
    "minimal": {"brightness": 0.45, "pulse": 0.22, "bass": 0.28, "noise": 0.10, "tension": 0.12},
}

LAYER_WEIGHTS = {
    "warm drone": ["healing", "dreamy", "ritual", "dark"],
    "glassy pluck": ["cosmic", "playful", "healing"],
    "tape hiss": ["dreamy", "dark", "ritual"],
    "sub bass": ["dark", "ritual", "urgent"],
    "radio static": ["cosmic", "dark", "urgent"],
    "alien chirp": ["playful", "cosmic", "mutate"],
    "portal riser": ["urgent", "drop", "mutate"],
    "soft impact": ["drop", "ritual", "urgent"],
    "shimmer pad": ["healing", "dreamy", "cosmic", "brighten"],
    "didgeridoo": ["ritual", "deep", "dark"],
    "bell dust": ["playful", "dreamy", "brighten"],
    "noise wash": ["dark", "freeze", "echo"],
}


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def choose_layers(action: str, mood: str, explicit: list[str]) -> list[str]:
    layers = set(explicit)
    for layer, hints in LAYER_WEIGHTS.items():
        if mood in hints or action in hints or (action == "deepen" and "deep" in hints):
            if len(layers) < 5:
                layers.add(layer)
    return sorted(layers)[:5]


def build_row(rng: random.Random, idx: int) -> dict:
    action = rng.choice(ACTIONS)
    mood = rng.choice(MOODS)
    texture_a, texture_b, texture_c = rng.sample(TEXTURES, 3)
    tempo_word = rng.choice(list(TEMPO_WORDS))
    # Keep BPM learnable from the prompt for the tiny realtime model.
    # Earlier drafts injected a hidden random BPM half the time, which made
    # the BPM target partly unobservable and hurt validation MAE.
    bpm = TEMPO_WORDS[tempo_word]
    pattern = rng.choice(PROMPT_PATTERNS)
    prompt = pattern.format(
        action=action,
        mood=mood,
        texture_a=texture_a,
        texture_b=texture_b,
        texture_c=texture_c,
        tempo_word=tempo_word,
        bpm=bpm,
    )
    explicit_layers = [texture_a, texture_b]
    if rng.random() < 0.35:
        explicit_layers.append(texture_c)

    base = MOOD_HINTS[mood].copy()
    adj = ACTION_HINTS[action]
    target = {
        "bpm": int(max(64, min(150, bpm + rng.choice([-4, 0, 0, 4])))),
        "brightness": round(clamp(base["brightness"] + adj["brightness"] + rng.uniform(-0.04, 0.04)), 3),
        "pulse": round(clamp(base["pulse"] + adj["pulse"] + rng.uniform(-0.04, 0.04)), 3),
        "bass": round(clamp(base["bass"] + adj["bass"] + rng.uniform(-0.04, 0.04)), 3),
        "noise": round(clamp(base["noise"] + (0.12 if "hiss" in prompt or "static" in prompt else 0) + rng.uniform(-0.03, 0.03)), 3),
        "tension": round(clamp(base["tension"] + adj["tension"] + rng.uniform(-0.04, 0.04)), 3),
        "action": action,
        "mood": mood,
        "layers": choose_layers(action, mood, explicit_layers),
        "next_variation": f"{action} next bar, keep {mood} identity, preserve loop-safe no-vocal texture",
    }
    return {
        "id": f"sf_director_{idx:05d}",
        "input": prompt,
        "target": target,
        "source": "synthetic_rule_v1",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/sonic_forage_director_v0")
    ap.add_argument("--rows", type=int, default=4200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows = [build_row(rng, i) for i in range(args.rows)]
    rng.shuffle(rows)
    n_train = int(len(rows) * 0.82)
    n_val = int(len(rows) * 0.09)
    splits = {
        "train": rows[:n_train],
        "validation": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }
    hashes = {}
    for split, split_rows in splits.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": "sonic_forage_director_v0",
        "purpose": "tiny realtime audio-director model: prompt/control text -> Web Audio plan controls",
        "training_started": False,
        "synthetic": True,
        "seed": SEED,
        "rows_total": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "actions": ACTIONS,
        "moods": MOODS,
        "layers": TEXTURES,
        "files_sha256": hashes,
        "schema": {"input": "string", "target": "bpm/brightness/pulse/bass/noise/tension/action/mood/layers/next_variation"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
