#!/usr/bin/env python3
"""
Local EPOC X / BCI vibe bridge for Sonic Forage.

Default mode is a no-hardware simulator so the browser path is testable now:

    uv run --with websockets scripts/emotiv_bci_vibe_bridge.py --simulate

Then open the lab and click "Connect BCI bridge" or "Simulate BCI vibe".

Real Cortex mode is intentionally conservative: EMOTIV Cortex requires Launcher/Cortex
running locally, a registered client id/secret, and the right license scopes for raw EEG.
Use the official Emotiv/cortex-v2-example client for the headset session, then adapt the
`map_cortex_sample` function below for the streams you can access (met/pow basic, eeg premium).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

try:
    import websockets
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: websockets\n"
        "Run: uv run --with websockets scripts/emotiv_bci_vibe_bridge.py --simulate"
    ) from exc


@dataclass
class BciVibeSample:
    source: str
    focus: float
    excitement: float
    relaxation: float
    stress: float
    engagement: float
    alpha: float
    beta: float
    theta: float
    gamma: float
    quality: float
    timestamp: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def simulated_sample(t: float) -> BciVibeSample:
    """Generate a plausible changing vibe signal for end-to-end audio testing."""
    focus = 0.5 + 0.35 * math.sin(t * 0.7)
    excitement = 0.48 + 0.32 * math.sin(t * 1.1 + 0.6)
    relaxation = 0.5 + 0.3 * math.sin(t * 0.45 + 2.2)
    stress = 0.18 + 0.22 * max(0.0, math.sin(t * 0.9 + 1.4))
    # add tiny jitter so repeated packets exercise smoothing in the browser/player
    jitter = lambda scale=0.025: random.uniform(-scale, scale)
    return BciVibeSample(
        source="simulated EPOC X vibe bridge",
        focus=clamp01(focus + jitter()),
        excitement=clamp01(excitement + jitter()),
        relaxation=clamp01(relaxation + jitter()),
        stress=clamp01(stress + jitter()),
        engagement=clamp01((focus + excitement) / 2 + jitter()),
        alpha=clamp01(relaxation + jitter()),
        beta=clamp01(excitement + jitter()),
        theta=clamp01(0.42 + 0.18 * math.sin(t * 0.38) + jitter()),
        gamma=clamp01(stress + jitter()),
        quality=1.0,
        timestamp=time.time(),
    )


def map_cortex_sample(stream: str, cols: list[str], values: list[Any]) -> dict[str, float | str]:
    """Map a Cortex sample into the browser's small normalized vibe schema.

    Cortex streams discovered from docs:
    - met: performance metrics. Best first real-time lane if available.
    - pow: band powers, 8 Hz. Good fallback without raw EEG.
    - eeg: raw 14-channel EPOC X microvolts, premium license required, 128/256 Hz.

    This mapper is deliberately tolerant because exact metric labels vary by Cortex version
    and license. Unknown labels are ignored; output fields default in the browser.
    """
    row = {label.lower(): value for label, value in zip(cols, values)}
    out: dict[str, float | str] = {"source": f"cortex:{stream}"}

    aliases = {
        "focus": ["focus", "attention", "att"],
        "engagement": ["eng", "engagement"],
        "excitement": ["exc", "excitement", "interest"],
        "relaxation": ["rel", "relaxation", "meditation"],
        "stress": ["str", "stress"],
    }
    for target, names in aliases.items():
        for name in names:
            if name in row and isinstance(row[name], (int, float)):
                out[target] = clamp01(float(row[name]))
                break

    # For pow stream labels often include channel/band combinations. Normalize band means.
    for band in ["alpha", "beta", "theta", "gamma"]:
        vals = [float(v) for k, v in row.items() if band in k and isinstance(v, (int, float))]
        if vals:
            # Band powers are not naturally 0-1. This log squasher is for vibe control only.
            mean = sum(vals) / len(vals)
            out[band] = clamp01(math.log1p(max(0.0, mean)) / 10.0)
    return out


async def broadcast_simulation(host: str, port: int, interval: float) -> None:
    clients: set[Any] = set()

    async def handler(ws: Any) -> None:
        clients.add(ws)
        print(f"client connected: {len(clients)} total")
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print(f"client disconnected: {len(clients)} total")

    async with websockets.serve(handler, host, port):
        print(f"BCI vibe bridge listening on ws://{host}:{port}")
        print("mode=simulate; open the lab and click Connect BCI bridge")
        t = 0.0
        while True:
            t += interval
            payload = json.dumps(asdict(simulated_sample(t)))
            if clients:
                await asyncio.gather(*(client.send(payload) for client in list(clients)), return_exceptions=True)
            print(payload)
            await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local EPOC X / BCI vibe bridge for Sonic Forage")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--simulate", action="store_true", help="Run no-hardware test stream")
    args = parser.parse_args()

    if not args.simulate:
        raise SystemExit(
            "Real Cortex mode is not enabled in this throwaway bridge yet.\n"
            "Use --simulate for browser/audio smoke now. For real EPOC X, run EMOTIV Launcher/Cortex,\n"
            "confirm your app client id/secret and license scopes, then wire the official\n"
            "Emotiv/cortex-v2-example subscription callbacks into map_cortex_sample()."
        )
    asyncio.run(broadcast_simulation(args.host, args.port, args.interval))


if __name__ == "__main__":
    main()
