"""Streamlit dashboard for the WiFi CSI sensing project.

Tabs:
  • Overview     – what the system does + headline metrics
  • Live Replay  – stream the held-out CSI, show the CSI heatmap + live prediction
  • Respiration  – estimate breaths/min on a stationary window
  • Model        – confusion matrices + SHAP explainability

Run:  streamlit run dashboard/app.py
(Run ./run.sh first so the dataset + model artefacts exist.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, preprocess, features  # noqa: E402

st.set_page_config(page_title="WiFi CSI Sensing", page_icon="📡", layout="wide")


@st.cache_resource
def load_model():
    if not config.MODEL_PATH.exists():
        return None, None
    return joblib.load(config.MODEL_PATH), joblib.load(config.SCALER_PATH)


@st.cache_data
def load_replay():
    if not config.REPLAY_CSV.exists():
        return None
    return pd.read_csv(config.REPLAY_CSV)


@st.cache_data
def load_metrics():
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text())


def _heatmap(arr: np.ndarray):
    """Normalise a 2D array to an RGB image for st.image (no matplotlib needed)."""
    a = arr - arr.min()
    a = a / (a.max() + 1e-6)
    r = np.clip(1.5 * a - 0.2, 0, 1)
    g = np.clip(a, 0, 1)
    b = np.clip(1.2 * (1 - a), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def predict_window(model, scaler, amp_window):
    amp = preprocess.standardise(amp_window)
    amp = preprocess.csi_parser.select_subcarriers(amp) \
        if amp.shape[1] == config.NUM_SUBCARRIERS else amp
    f = features.window_features(amp)[None, :]
    idx = int(model.predict(scaler.transform(f))[0])
    return config.CLASSES[idx]


model, scaler = load_model()
replay = load_replay()
metrics = load_metrics()

st.title("📡 WiFi CSI Human Sensing")
st.caption("Device-free presence · activity · fall · respiration — from ESP32 channel state information")

if model is None:
    st.warning("No trained model found. Run `./run.sh` (or `python -m src.train`) first.")

tab_over, tab_live, tab_resp, tab_model = st.tabs(
    ["Overview", "Live Replay", "Respiration", "Model"])

# --------------------------------------------------------------------------- #
with tab_over:
    c1, c2, c3 = st.columns(3)
    if metrics:
        best = metrics["best_model"]
        f1 = metrics["models"][best]["macro_f1"]
        c1.metric("Best model", best)
        c2.metric("Macro-F1", f"{f1:.2f}")
        c3.metric("Train windows", f"{metrics['n_windows']:,}")
    st.markdown(
        """
This system infers **what a person is doing** from how their body distorts
ambient WiFi — no camera, works in the dark and through smoke.

* **Tier 1 – Presence:** empty vs occupied
* **Tier 2 – Activity:** sit / stand / walk / **fall detection**
* **Tier 3 – Respiration:** breaths-per-minute on a still subject

Data source is pluggable: synthetic (default), a public dataset (UT-HAR), or a
live ESP32 capture — see the **Live Replay** and **Respiration** tabs.
        """)

# --------------------------------------------------------------------------- #
with tab_live:
    st.subheader("Live CSI replay + activity prediction")
    if replay is None:
        st.info("Run `python -m src.generate_synthetic` to create the replay stream.")
    else:
        classes = replay["label"].unique().tolist()
        pick = st.selectbox("Replay a held-out class", classes,
                            index=min(3, len(classes) - 1))
        speed = st.slider("Playback speed", 1, 20, 8)
        seg = replay[replay["label"] == pick].reset_index(drop=True)
        amp_cols = [c for c in seg.columns if c.startswith("sc")]
        amp = seg[amp_cols].to_numpy(dtype=np.float32)

        heat = st.empty()
        pred_box = st.empty()
        prog = st.progress(0.0)
        W = config.WINDOW_SIZE
        if st.button("▶ Play"):
            for i in range(0, max(1, len(amp) - W), W // 2):
                win = amp[i:i + W]
                if win.shape[0] < W:
                    break
                heat.image(_heatmap(win.T), caption="CSI amplitude (subcarrier × time)",
                           use_container_width=True)
                if model is not None:
                    pred = predict_window(model, scaler, win)
                    ok = "✅" if pred == pick else "↪"
                    pred_box.metric("Predicted activity", f"{pred} {ok}",
                                    delta=f"true: {pick}")
                prog.progress(min(1.0, i / max(1, len(amp) - W)))
                time.sleep(1.0 / speed)
            st.success("Replay complete.")

# --------------------------------------------------------------------------- #
with tab_resp:
    st.subheader("Respiration-rate estimation")
    st.caption("Stationary subject only. Uses a long window + spectral peak in the breathing band.")
    if replay is None:
        st.info("Generate the dataset first.")
    else:
        seg = replay[replay["label"] == "breathe"]
        amp_cols = [c for c in seg.columns if c.startswith("sc")]
        amp = seg[amp_cols].to_numpy(dtype=np.float32)
        amp = preprocess.csi_parser.select_subcarriers(amp)
        W = config.RESP_WINDOW_SIZE
        if len(amp) >= W:
            bpm = features.respiration_rate(amp[:W])
            st.metric("Estimated respiration", f"{bpm:.1f} breaths/min")
            st.caption("(synthetic breathe sessions are generated at 12–24 bpm)")
        else:
            st.info("Not enough samples for a respiration window.")

# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Evaluation")
    if metrics:
        rows = [{"model": k, "macro_f1": v["macro_f1"]}
                for k, v in metrics["models"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cols = st.columns(2)
    imgs = sorted(config.REPORTS_DIR.glob("confusion_*.png"))
    for i, p in enumerate(imgs):
        cols[i % 2].image(str(p), caption=p.stem, use_container_width=True)
    shap_png = config.REPORTS_DIR / "shap_summary.png"
    if shap_png.exists():
        st.image(str(shap_png), caption="Feature importance (SHAP)",
                 use_container_width=True)
