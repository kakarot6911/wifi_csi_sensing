"""CSI preprocessing: denoise → select subcarriers → window.

Pipeline order matters and mirrors the WiFi-sensing literature:
  1. amplitude (and optional sanitized phase)
  2. Hampel filter   – kill impulsive outliers from packet bursts
  3. low-pass Butterworth – remove high-freq electronic noise, keep body motion
  4. drop null subcarriers
  5. per-subcarrier standardisation
  6. sliding windows → (n_windows, WINDOW_SIZE, n_subcarriers)

scipy is used for the Butterworth filter; if it is missing we fall back to a
moving-average so the pipeline still runs.
"""

from __future__ import annotations

import numpy as np

from . import config, csi_parser

try:
    from scipy.signal import butter, filtfilt
    _HAVE_SCIPY = True
except Exception:                               # pragma: no cover
    _HAVE_SCIPY = False


def hampel(x: np.ndarray, k: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Vectorised Hampel outlier filter along time (axis 0) for a (T, S) array."""
    x = x.copy()
    T = x.shape[0]
    win = 2 * k + 1
    if T < win:
        return x
    # rolling median / MAD via stride tricks
    pad = np.pad(x, ((k, k), (0, 0)), mode="edge")
    idx = np.arange(win)[:, None] + np.arange(T)[None, :]
    windows = pad[idx]                          # (win, T, S)
    med = np.median(windows, axis=0)
    mad = np.median(np.abs(windows - med), axis=0)
    thresh = n_sigma * 1.4826 * mad
    mask = np.abs(x - med) > thresh
    x[mask] = med[mask]
    return x


def lowpass(x: np.ndarray, cutoff_hz: float = 10.0, order: int = 4) -> np.ndarray:
    """Low-pass filter each subcarrier (axis 0 = time)."""
    fs = config.SAMPLE_RATE_HZ
    if _HAVE_SCIPY and x.shape[0] > 3 * order:
        b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
        return filtfilt(b, a, x, axis=0)
    # fallback: short moving average
    w = max(3, int(fs / cutoff_hz))
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, x)


def denoise_amplitude(csi_stream: np.ndarray) -> np.ndarray:
    """Full amplitude clean-up: |CSI| → Hampel → low-pass → drop null subcarriers."""
    amp = csi_parser.amplitude(csi_stream).astype(np.float32)
    amp = hampel(amp)
    amp = lowpass(amp)
    return csi_parser.select_subcarriers(amp)


def standardise(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-subcarrier zero-mean/unit-var over the whole stream."""
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + eps)


def window_stream(
    x: np.ndarray,
    labels: np.ndarray | None = None,
    size: int = config.WINDOW_SIZE,
    step: int = config.WINDOW_STEP,
):
    """Slide a window over (T, S). Returns (windows, [window_labels]).

    A window's label is the majority label of its samples.
    """
    T = x.shape[0]
    starts = range(0, max(1, T - size + 1), step)
    wins = np.stack([x[s : s + size] for s in starts]) if T >= size else x[None, :]
    if labels is None:
        return wins
    wlabels = np.array(
        [np.bincount(labels[s : s + size]).argmax() for s in starts]
        if T >= size
        else [np.bincount(labels).argmax()]
    )
    return wins, wlabels
