"""Loaders for public WiFi-CSI datasets, normalised to our format.

Every loader returns the same triple the synthetic generator does:
    csi   : (T, NUM_SUBCARRIERS) complex64  (or amplitude as real)
    label : (T,) int64 index into config.CLASSES
    info  : dict (source name, original label set, ...)

so train.py is agnostic to where the data came from.

Supported out of the box:
  * UT-HAR  – widely used HAR benchmark, 7 activities, single-link CSI.
    Download: https://github.com/ermongroup/Wifi_Activity_Recognition
    Drop the extracted files under  data/raw/ut_har/  then:
        python -m src.train --source ut_har
"""

from __future__ import annotations

import numpy as np

from . import config


def normalize_labels(raw_labels) -> np.ndarray:
    """Map free-text dataset labels onto config.CLASSES indices."""
    idx = {c: i for i, c in enumerate(config.CLASSES)}
    out = []
    for lab in raw_labels:
        key = str(lab).strip().lower().replace(" ", "").replace("_", "")
        coarse = config.PUBLIC_LABEL_MAP.get(key)
        out.append(idx.get(coarse, idx["empty"]) if coarse else idx["empty"])
    return np.array(out, dtype=np.int64)


def load_ut_har(root=None):
    """Load UT-HAR if present under data/raw/ut_har/ (numpy .npy or .csv shards).

    The repo ships the data as flattened CSV/NPY; we reshape to (T, S). Because
    UT-HAR reports 90 subcarriers (3 antennas x 30), we average antennas down to
    a single-link, 30→NUM_SUBCARRIERS interpolation so it matches the ESP32 shape.
    """
    root = root or (config.RAW_DIR / "ut_har")
    if not root.exists():
        raise FileNotFoundError(
            f"UT-HAR not found at {root}. See the docstring in src/datasets.py "
            "for the download link, or run on synthetic data instead."
        )
    xs, ys = [], []
    for f in sorted(root.glob("*.npy")):
        arr = np.load(f)
        xs.append(arr.reshape(arr.shape[0], -1))
        ys.append([f.stem.split("_")[0]] * arr.shape[0])
    if not xs:
        raise FileNotFoundError(f"No .npy shards in {root}.")
    X = np.concatenate(xs).astype(np.float32)
    y = normalize_labels(np.concatenate(ys))

    # collapse 3 antennas (90 → 30) then resample to NUM_SUBCARRIERS
    if X.shape[1] % 3 == 0:
        X = X.reshape(X.shape[0], 3, -1).mean(1)
    S = config.NUM_SUBCARRIERS
    xp = np.linspace(0, 1, X.shape[1])
    xq = np.linspace(0, 1, S)
    X = np.stack([np.interp(xq, xp, row) for row in X]).astype(np.float32)
    return X, y, {"source": "ut_har", "n": len(y)}


def load_synthetic():
    """Load the generated synthetic dataset as (csi, label, info)."""
    if not config.SYNTHETIC_NPZ.exists():
        raise FileNotFoundError(
            f"{config.SYNTHETIC_NPZ} missing — run `python -m src.generate_synthetic`."
        )
    d = np.load(config.SYNTHETIC_NPZ, allow_pickle=True)
    return d["csi"], d["label"], {"source": "synthetic", "session": d["session"]}


def load(source: str = "synthetic"):
    if source == "synthetic":
        return load_synthetic()
    if source == "ut_har":
        X, y, info = load_ut_har()
        return X.astype(np.complex64), y, info   # already amplitude → real CSI
    raise ValueError(f"unknown source: {source!r}")
