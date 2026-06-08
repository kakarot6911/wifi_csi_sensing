"""Generate a synthetic, ESP32-style CSI dataset.

This lets the whole pipeline (preprocess → features → train → dashboard) run
end-to-end **before** you flash an ESP32 or download a public dataset. Each
class is synthesised from a physically-motivated channel model:

  empty    – static multipath + thermal noise only (no body)
  sit/stand– a near-static reflector; tiny, slow channel drift
  walk     – a strong moving reflector → large broadband amplitude variance
  fall     – brief violent transient (the spike falls detectors key on) → settle
  breathe  – a stationary body modulating a few subcarriers at ~0.2–0.4 Hz

The output is one .npz with the raw complex CSI stream, per-sample labels and
session ids, plus a held-out `replay_stream.csv` for the live dashboard demo.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from . import config


def _static_channel(rng: np.random.Generator) -> np.ndarray:
    """A random but fixed multipath channel: complex gain per subcarrier."""
    taps = rng.normal(0, 1, (5, 1)) + 1j * rng.normal(0, 1, (5, 1))
    delays = rng.uniform(0, 0.5, (5, 1))
    f = np.arange(config.NUM_SUBCARRIERS)[None, :]
    h = (taps * np.exp(-2j * np.pi * delays * f / config.NUM_SUBCARRIERS)).sum(0)
    return (h / np.abs(h).mean()).astype(np.complex64) * 20.0


def _session(rng: np.random.Generator, label: str, seconds: float) -> np.ndarray:
    n = int(seconds * config.SAMPLE_RATE_HZ)
    S = config.NUM_SUBCARRIERS
    h0 = _static_channel(rng)
    t = np.arange(n) / config.SAMPLE_RATE_HZ
    noise = (rng.normal(0, 0.4, (n, S)) + 1j * rng.normal(0, 0.4, (n, S))).astype(np.complex64)
    stream = np.tile(h0, (n, 1)).astype(np.complex64)

    if label == "empty":
        pass
    elif label in ("sit", "stand"):
        # slow shallow drift + a small static reflector offset
        drift = 0.6 * np.sin(2 * np.pi * rng.uniform(0.02, 0.08) * t)[:, None]
        offset = rng.normal(0, 0.3 if label == "sit" else 0.5, S)
        stream += (drift * offset).astype(np.complex64)
    elif label == "walk":
        # a reflector with time-varying Doppler → strong broadband variation
        dop = rng.uniform(0.8, 2.5)
        ph = rng.uniform(0, 2 * np.pi, S)[None, :]
        moving = 6.0 * np.sin((2 * np.pi * dop * t)[:, None] + ph)
        stream += moving.astype(np.complex64)
        stream += (rng.normal(0, 1.5, (n, S))).astype(np.complex64)
    elif label == "fall":
        # baseline calm, then a sharp transient burst, then settle
        stream += (0.5 * rng.normal(0, 1, (n, S))).astype(np.complex64)
        t0 = rng.integers(int(0.3 * n), int(0.7 * n))
        env = np.exp(-((np.arange(n) - t0) ** 2) / (2 * (0.15 * config.SAMPLE_RATE_HZ) ** 2))
        burst = 14.0 * env[:, None] * rng.normal(0, 1, (n, S))
        stream += burst.astype(np.complex64)
    elif label == "breathe":
        # stationary body: slow sinusoid on a subset of subcarriers
        bpm = rng.uniform(0.20, 0.40)           # 12–24 breaths/min
        affected = rng.choice(S, size=S // 3, replace=False)
        resp = np.zeros((n, S), dtype=np.float32)
        resp[:, affected] = (1.2 * np.sin(2 * np.pi * bpm * t)[:, None]
                             * rng.uniform(0.5, 1.5, len(affected)))
        stream += resp.astype(np.complex64)

    return stream + noise


def generate(seconds_per_session: float = 12.0, sessions_per_class: int = 12,
             seed: int = config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    streams, labels, sessions = [], [], []
    sid = 0
    for cls_idx, cls in enumerate(config.CLASSES):
        for _ in range(sessions_per_class):
            s = _session(rng, cls, seconds_per_session)
            streams.append(s)
            labels.append(np.full(len(s), cls_idx, dtype=np.int64))
            sessions.append(np.full(len(s), sid, dtype=np.int64))
            sid += 1
    csi = np.concatenate(streams)
    y = np.concatenate(labels)
    sess = np.concatenate(sessions)

    np.savez_compressed(config.SYNTHETIC_NPZ, csi=csi, label=y, session=sess,
                        classes=np.array(config.CLASSES))
    print(f"  wrote {config.SYNTHETIC_NPZ.name}: {csi.shape[0]:,} samples "
          f"x {csi.shape[1]} subcarriers, {sid} sessions")

    # held-out replay stream: one fresh session per class, concatenated
    rng2 = np.random.default_rng(seed + 999)
    rep, rep_lbl = [], []
    for cls_idx, cls in enumerate(config.CLASSES):
        secs = config.RESP_WINDOW_SEC + 4 if cls == "breathe" else 6.0
        s = _session(rng2, cls, secs)
        rep.append(np.abs(s))                   # amplitude is enough for replay
        rep_lbl.append(np.full(len(s), cls, dtype=object))
    amp = np.concatenate(rep)
    df = pd.DataFrame(amp, columns=[f"sc{i}" for i in range(amp.shape[1])])
    df.insert(0, "label", np.concatenate(rep_lbl))
    df.to_csv(config.REPLAY_CSV, index=False)
    print(f"  wrote {config.REPLAY_CSV.name}: {len(df):,} samples for the live demo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic CSI dataset")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--sessions", type=int, default=12)
    args = ap.parse_args()
    print("==> Generating synthetic ESP32-style CSI")
    generate(args.seconds, args.sessions)
