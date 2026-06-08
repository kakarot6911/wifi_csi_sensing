"""Parse raw ESP32 CSI into amplitude / phase, and sanitize phase.

The ESP32 (`esp-csi` firmware) prints one line per received packet, e.g.:

    CSI_DATA,STA,aa:bb:cc:dd:ee:ff,-42,11,1,...,[3 -1 4 0 -2 5 ...]

The trailing bracketed field is a flat list of **int8 I/Q pairs**, two per
subcarrier (imag, real interleaved in the ESP order). This module turns that
into a complex array and the usual amplitude/phase representations.

Everything here is pure numpy so it runs the same on captured CSV, the live
serial stream, or synthetic data.
"""

from __future__ import annotations

import re
import numpy as np

from . import config

_BRACKET = re.compile(r"\[([-\d\s]+)\]")


def parse_esp_line(line: str) -> np.ndarray | None:
    """Parse one `CSI_DATA,...` serial line → complex CSI vector.

    Returns a complex array of length NUM_SUBCARRIERS, or None if the line is
    not a CSI record / is malformed.
    """
    if "CSI_DATA" not in line:
        return None
    m = _BRACKET.search(line)
    if not m:
        return None
    try:
        raw = np.fromstring(m.group(1), sep=" ", dtype=np.float32)
    except Exception:
        return None
    if raw.size < 2:
        return None
    raw = raw[: (raw.size // 2) * 2]          # drop a dangling byte if any
    imag, real = raw[0::2], raw[1::2]
    csi = real + 1j * imag
    # Pad/truncate to the expected subcarrier count so downstream shapes hold.
    out = np.zeros(config.NUM_SUBCARRIERS, dtype=np.complex64)
    n = min(csi.size, config.NUM_SUBCARRIERS)
    out[:n] = csi[:n]
    return out


def amplitude(csi: np.ndarray) -> np.ndarray:
    """|CSI| per subcarrier. Accepts a (T, S) stream or a single (S,) vector."""
    return np.abs(csi)


def phase(csi: np.ndarray) -> np.ndarray:
    """Raw wrapped phase per subcarrier."""
    return np.angle(csi)


def sanitize_phase(phase_stream: np.ndarray) -> np.ndarray:
    """Remove the linear phase ramp (CFO/STO) that makes raw ESP32 phase unusable.

    Per sample we unwrap across subcarriers and subtract the best-fit line, a
    standard CSI phase-calibration step. Input/Output: (T, S).
    """
    phase_stream = np.unwrap(phase_stream, axis=-1)
    s = np.arange(phase_stream.shape[-1])
    # least-squares slope/intercept per row, vectorised
    s_mean = s.mean()
    p_mean = phase_stream.mean(axis=-1, keepdims=True)
    slope = ((phase_stream - p_mean) * (s - s_mean)).sum(-1, keepdims=True) / (
        ((s - s_mean) ** 2).sum() + 1e-9
    )
    intercept = p_mean - slope * s_mean
    return phase_stream - (slope * s + intercept)


def select_subcarriers(stream: np.ndarray) -> np.ndarray:
    """Keep only non-null subcarriers. Input (T, NUM_SUBCARRIERS) → (T, 52)."""
    return stream[..., config.VALID_SUBCARRIERS]
