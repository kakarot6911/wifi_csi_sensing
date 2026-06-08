"""Hand-crafted features for the classical model + respiration-rate estimator.

A deep model (cnn_lstm.py) eats raw windows, but a Random-Forest on compact
statistical features is a strong, dependency-light baseline and is what we run
SHAP on for explainability — mirroring the NIDS project.
"""

from __future__ import annotations

import numpy as np

from . import config

try:
    from scipy.signal import welch
    _HAVE_SCIPY = True
except Exception:                               # pragma: no cover
    _HAVE_SCIPY = False


def _entropy(x: np.ndarray) -> float:
    h, _ = np.histogram(x, bins=16, density=True)
    h = h[h > 0]
    return float(-(h * np.log(h)).sum())


def window_features(win: np.ndarray) -> np.ndarray:
    """Compact feature vector for one (WINDOW_SIZE, S) window.

    Captures *how much / how fast* the channel moves (the motion signal) plus
    spectral spread, aggregated across subcarriers.
    """
    # per-subcarrier temporal stats, then aggregate across subcarriers
    std_sc = win.std(0)                         # motion magnitude per subcarrier
    rng_sc = win.max(0) - win.min(0)
    diff = np.diff(win, axis=0)
    motion = np.abs(diff).mean(0)               # mean abs velocity per subcarrier
    feats = [
        std_sc.mean(), std_sc.max(), std_sc.std(),
        rng_sc.mean(), rng_sc.max(),
        motion.mean(), motion.max(),
        np.abs(diff).std(),                     # jerkiness → falls spike here
        _entropy(win.ravel()),
        # correlation between adjacent subcarriers (de-correlates when moving)
        float(np.mean([np.corrcoef(win[:, i], win[:, i + 1])[0, 1]
                       for i in range(0, win.shape[1] - 1, 4)])),
    ]
    return np.nan_to_num(np.array(feats, dtype=np.float32))


FEATURE_NAMES = [
    "std_mean", "std_max", "std_spread",
    "range_mean", "range_max",
    "motion_mean", "motion_max",
    "jerk_std", "entropy", "subcarrier_corr",
]


def features_for_windows(windows: np.ndarray) -> np.ndarray:
    return np.stack([window_features(w) for w in windows])


# --------------------------------------------------------------------------- #
# Tier 3: respiration rate
# --------------------------------------------------------------------------- #
def respiration_rate(win: np.ndarray) -> float:
    """Estimate breaths-per-minute from a (RESP_WINDOW_SIZE, S) amplitude window.

    Project subcarriers onto their first principal component (the breathing
    signal dominates a stationary subject), then find the spectral peak inside
    the respiration band.
    """
    x = win - win.mean(0, keepdims=True)
    # PCA via SVD; first component = strongest shared oscillation
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        sig = u[:, 0] * s[0]
    except np.linalg.LinAlgError:
        sig = x.mean(1)

    fs = config.SAMPLE_RATE_HZ
    lo, hi = config.RESP_BAND_HZ
    if _HAVE_SCIPY:
        f, p = welch(sig, fs=fs, nperseg=min(len(sig), 512))
    else:
        p = np.abs(np.fft.rfft(sig)) ** 2
        f = np.fft.rfftfreq(len(sig), 1 / fs)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return float("nan")
    peak = f[band][np.argmax(p[band])]
    return float(peak * 60.0)                   # Hz → breaths/min
