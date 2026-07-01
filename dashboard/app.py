"""Streamlit dashboard for the WiFi CSI sensing project — premium UI.

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

st.set_page_config(page_title="WiFi CSI Sensing", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
ACCENT = "linear-gradient(135deg, #22d3ee 0%, #818cf8 55%, #c084fc 100%)"
CLASS_META = {
    "empty":   ("🌑", "#64748b", "No occupant"),
    "sit":     ("🪑", "#38bdf8", "Seated"),
    "stand":   ("🧍", "#34d399", "Standing"),
    "walk":    ("🚶", "#a78bfa", "Walking"),
    "fall":    ("🚨", "#fb7185", "Fall detected"),
    "breathe": ("🫁", "#f0abfc", "Breathing"),
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(129,140,248,0.14), transparent 60%),
    radial-gradient(1000px 500px at 100% 0%, rgba(34,211,238,0.10), transparent 55%),
    #0a0e1a;
}
/* strip default chrome for a cleaner canvas */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

/* ---------- hero ---------- */
.hero { margin: 0 0 1.6rem 0; }
.hero .eyebrow {
  display:inline-flex; align-items:center; gap:.5rem;
  font-size:.72rem; letter-spacing:.18em; text-transform:uppercase;
  color:#a5b4fc; font-weight:600;
  padding:.35rem .8rem; border-radius:999px;
  background:rgba(129,140,248,0.10); border:1px solid rgba(129,140,248,0.28);
}
.hero h1 {
  font-size:2.9rem; font-weight:800; line-height:1.05; margin:.8rem 0 .3rem 0;
  background:%ACCENT%; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; letter-spacing:-.02em;
}
.hero p { color:#94a3b8; font-size:1.05rem; max-width:640px; margin:.2rem 0 0 0; }
.pulse { width:9px; height:9px; border-radius:50%; background:#34d399;
  box-shadow:0 0 0 0 rgba(52,211,153,.7); animation:pulse 2s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(52,211,153,.6);}
  70%{box-shadow:0 0 0 8px rgba(52,211,153,0);} 100%{box-shadow:0 0 0 0 rgba(52,211,153,0);} }

/* ---------- cards ---------- */
.grid { display:grid; gap:1rem; }
.g3 { grid-template-columns:repeat(3,1fr); }
.g2 { grid-template-columns:repeat(2,1fr); }
@media (max-width:820px){ .g3,.g2{ grid-template-columns:1fr; } }

.card {
  background:rgba(255,255,255,0.035);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:18px; padding:1.25rem 1.35rem;
  backdrop-filter:blur(8px);
  transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.card:hover { transform:translateY(-3px); border-color:rgba(129,140,248,0.4);
  box-shadow:0 12px 40px -12px rgba(129,140,248,0.35); }

.stat .label { color:#94a3b8; font-size:.78rem; letter-spacing:.08em;
  text-transform:uppercase; font-weight:600; }
.stat .value { font-size:2.2rem; font-weight:800; color:#f1f5f9; margin-top:.25rem;
  letter-spacing:-.02em; }
.stat .sub { color:#64748b; font-size:.82rem; margin-top:.15rem; }
.stat .value.grad { background:%ACCENT%; -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }

.cap { display:flex; gap:.9rem; align-items:flex-start; }
.cap .ic { font-size:1.5rem; line-height:1; }
.cap h4 { margin:0; font-size:1rem; color:#e2e8f0; font-weight:700; }
.cap p { margin:.25rem 0 0 0; color:#94a3b8; font-size:.86rem; line-height:1.45; }
.tier { font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:#a5b4fc; }

/* ---------- prediction panel ---------- */
.pred {
  border-radius:18px; padding:1.4rem 1.5rem;
  background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);
}
.pred.danger { border-color:rgba(251,113,133,0.55);
  box-shadow:0 0 0 1px rgba(251,113,133,0.25), 0 16px 50px -18px rgba(251,113,133,0.5); }
.pred .top { display:flex; align-items:center; gap:1rem; }
.pred .emoji { font-size:2.6rem; line-height:1; }
.pred .name { font-size:1.9rem; font-weight:800; color:#f1f5f9; letter-spacing:-.01em;
  text-transform:capitalize; }
.pred .verdict { margin-left:auto; font-size:.8rem; font-weight:700; padding:.35rem .7rem;
  border-radius:999px; }
.ok { color:#6ee7b7; background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.35); }
.miss { color:#fca5a5; background:rgba(251,113,133,0.12); border:1px solid rgba(251,113,133,0.35); }

.conf { margin-top:1.1rem; display:flex; flex-direction:column; gap:.55rem; }
.conf .row { display:grid; grid-template-columns:78px 1fr 46px; align-items:center; gap:.7rem; }
.conf .cls { color:#cbd5e1; font-size:.85rem; font-weight:600; text-transform:capitalize; }
.conf .track { height:9px; border-radius:999px; background:rgba(255,255,255,0.07); overflow:hidden; }
.conf .fill { height:100%; border-radius:999px; background:%ACCENT%; }
.conf .fill.dim { background:rgba(148,163,184,0.4); }
.conf .pct { color:#94a3b8; font-size:.8rem; text-align:right; font-variant-numeric:tabular-nums; }

/* ---------- respiration gauge ---------- */
.gauge { text-align:center; padding:1.8rem 1rem; }
.gauge .num { font-size:4rem; font-weight:800; letter-spacing:-.03em;
  background:%ACCENT%; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; }
.gauge .unit { color:#94a3b8; font-size:1rem; font-weight:600; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid rgba(255,255,255,0.07); }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.55rem 1.1rem;
  color:#94a3b8; font-weight:600; }
.stTabs [aria-selected="true"] { color:#f1f5f9 !important;
  background:rgba(129,140,248,0.10); }
.stTabs [data-baseweb="tab-highlight"] { background:%ACCENT%; height:3px; border-radius:3px; }

/* buttons */
.stButton>button { border-radius:12px; font-weight:700; border:1px solid rgba(129,140,248,0.4);
  background:rgba(129,140,248,0.12); color:#e2e8f0; padding:.5rem 1.3rem; transition:.15s; }
.stButton>button:hover { background:rgba(129,140,248,0.25); border-color:#818cf8;
  transform:translateY(-1px); }

.sect { color:#e2e8f0; font-size:1.15rem; font-weight:700; margin:.4rem 0 .9rem 0; }
.foot { color:#475569; font-size:.78rem; text-align:center; margin-top:2.5rem;
  padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.06); }
hr { border-color:rgba(255,255,255,0.06); }
</style>
""".replace("%ACCENT%", ACCENT)

st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data / model loaders
# --------------------------------------------------------------------------- #
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


@st.cache_resource
def _cmap():
    import matplotlib
    return matplotlib.colormaps["turbo"]


def _heatmap(arr: np.ndarray):
    """Map a 2D array to a premium RGB image via the 'turbo' colormap."""
    a = arr - arr.min()
    a = a / (a.max() + 1e-6)
    rgba = _cmap()(a)                      # (H, W, 4) float
    return (rgba[..., :3] * 255).astype(np.uint8)


def predict_window(model, scaler, amp_window):
    amp = preprocess.standardise(amp_window)
    amp = preprocess.csi_parser.select_subcarriers(amp) \
        if amp.shape[1] == config.NUM_SUBCARRIERS else amp
    f = features.window_features(amp)[None, :]
    Xs = scaler.transform(f)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[0]
        idx = int(np.argmax(proba))
    else:
        idx = int(model.predict(Xs)[0])
        proba = None
    return config.CLASSES[idx], proba


def _confidence_html(proba, predicted):
    order = np.argsort(proba)[::-1]
    rows = ""
    for rank, i in enumerate(order[:4]):
        cls = config.CLASSES[i]
        pct = float(proba[i]) * 100
        fill = "fill" if rank == 0 else "fill dim"
        rows += (
            f'<div class="row"><div class="cls">{cls}</div>'
            f'<div class="track"><div class="{fill}" style="width:{pct:.0f}%"></div></div>'
            f'<div class="pct">{pct:.0f}%</div></div>'
        )
    return f'<div class="conf">{rows}</div>'


def _prediction_html(pred, true_label, proba):
    emoji, _, _ = CLASS_META.get(pred, ("📶", "#818cf8", ""))
    hit = pred == true_label
    verdict = ('<span class="verdict ok">✓ correct</span>' if hit
               else '<span class="verdict miss">✗ true: ' + str(true_label) + '</span>')
    danger = " danger" if pred == "fall" else ""
    conf = _confidence_html(proba, pred) if proba is not None else ""
    return (
        f'<div class="pred{danger}"><div class="top">'
        f'<span class="emoji">{emoji}</span>'
        f'<span class="name">{pred}</span>{verdict}</div>{conf}</div>'
    )


model, scaler = load_model()
replay = load_replay()
metrics = load_metrics()

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
live_dot = ('<span class="pulse"></span> Model loaded' if model is not None
            else "⚠ No model")
st.markdown(f"""
<div class="hero">
  <span class="eyebrow">{live_dot} · ESP32 · Channel State Information</span>
  <h1>WiFi CSI Human Sensing</h1>
  <p>Device-free presence, activity, fall &amp; respiration sensing — inferred from how a
     body distorts ambient WiFi. No camera. Works in the dark, through smoke, around corners.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("No trained model found. Run `./run.sh` (or `python -m src.train`) first.")

tab_over, tab_live, tab_resp, tab_model = st.tabs(
    ["  Overview  ", "  Live Replay  ", "  Respiration  ", "  Model  "])

# --------------------------------------------------------------------------- #
with tab_over:
    if metrics:
        best = metrics["best_model"]
        f1 = metrics["models"][best]["macro_f1"]
        n_models = len(metrics["models"])
        st.markdown(f"""
        <div class="grid g3">
          <div class="card stat"><div class="label">Best model</div>
            <div class="value">{best}</div><div class="sub">selected by macro-F1</div></div>
          <div class="card stat"><div class="label">Macro-F1</div>
            <div class="value grad">{f1:.2f}</div><div class="sub">balanced across 6 classes</div></div>
          <div class="card stat"><div class="label">Training windows</div>
            <div class="value">{metrics['n_windows']:,}</div>
            <div class="sub">{n_models} models · {metrics.get('source','synthetic')} source</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sect" style="margin-top:1.6rem">Sensing tiers</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="grid g3">
      <div class="card cap"><div class="ic">🚦</div><div>
        <div class="tier">Tier 1 · Presence</div>
        <h4>Empty vs occupied</h4>
        <p>Detect whether anyone is in the space at all — the coarsest, most robust signal.</p></div></div>
      <div class="card cap"><div class="ic">🏃</div><div>
        <div class="tier">Tier 2 · Activity</div>
        <h4>Sit · stand · walk · fall</h4>
        <p>Classify motion state, including <b>fall detection</b> from the jerk signature.</p></div></div>
      <div class="card cap"><div class="ic">🫁</div><div>
        <div class="tier">Tier 3 · Respiration</div>
        <h4>Breaths per minute</h4>
        <p>Spectral peak in the breathing band recovers respiration rate on a still subject.</p></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sect" style="margin-top:1.6rem">Signal pipeline</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <p style="color:#cbd5e1; margin:0; line-height:1.9; font-size:.92rem;">
      <b style="color:#22d3ee">ESP32 TX → RX</b> &nbsp;·&nbsp; packets @ 100&nbsp;Hz &nbsp;→&nbsp;
      <b style="color:#818cf8">csi_parser</b> amplitude / phase sanitise &nbsp;→&nbsp;
      <b style="color:#818cf8">preprocess</b> Hampel · low-pass · window &nbsp;→&nbsp;
      <b style="color:#818cf8">features</b> motion · variance · entropy &nbsp;→&nbsp;
      <b style="color:#c084fc">model</b> RandomForest + MLP + CNN-LSTM &nbsp;→&nbsp;
      <b style="color:#f0abfc">prediction</b> · heatmap · bpm</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
with tab_live:
    st.markdown('<div class="sect">Live CSI replay &amp; activity prediction</div>',
                unsafe_allow_html=True)
    if replay is None:
        st.info("Run `python -m src.generate_synthetic` to create the replay stream.")
    else:
        classes = replay["label"].unique().tolist()
        c1, c2 = st.columns([2, 1])
        with c1:
            pick = st.selectbox("Held-out class to replay", classes,
                                index=min(3, len(classes) - 1))
        with c2:
            speed = st.slider("Playback speed", 1, 20, 10)

        seg = replay[replay["label"] == pick].reset_index(drop=True)
        amp_cols = [c for c in seg.columns if c.startswith("sc")]
        amp = seg[amp_cols].to_numpy(dtype=np.float32)

        play = st.button("▶  Play stream", width="stretch")

        left, right = st.columns([3, 2])
        heat = left.empty()
        pred_box = right.empty()
        prog = st.progress(0.0)
        W = config.WINDOW_SIZE

        # idle preview so the panel is never empty
        if amp.shape[0] >= W:
            heat.image(_heatmap(amp[:W].T),
                       caption="CSI amplitude  (subcarrier × time)",
                       width="stretch")
        if model is not None and amp.shape[0] >= W:
            p0, pr0 = predict_window(model, scaler, amp[:W])
            pred_box.markdown(_prediction_html(p0, pick, pr0), unsafe_allow_html=True)

        if play:
            for i in range(0, max(1, len(amp) - W), W // 2):
                win = amp[i:i + W]
                if win.shape[0] < W:
                    break
                heat.image(_heatmap(win.T),
                           caption="CSI amplitude  (subcarrier × time)",
                           width="stretch")
                if model is not None:
                    pred, proba = predict_window(model, scaler, win)
                    pred_box.markdown(_prediction_html(pred, pick, proba),
                                      unsafe_allow_html=True)
                prog.progress(min(1.0, i / max(1, len(amp) - W)))
                time.sleep(1.0 / speed)
            prog.progress(1.0)
            st.success("Replay complete.")

# --------------------------------------------------------------------------- #
with tab_resp:
    st.markdown('<div class="sect">Respiration-rate estimation</div>',
                unsafe_allow_html=True)
    st.caption("Stationary subject only · long window + spectral peak in the breathing band.")
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
            st.markdown(f"""
            <div class="card gauge">
              <div class="num">{bpm:.1f}</div>
              <div class="unit">breaths / minute</div>
              <p style="color:#64748b; font-size:.82rem; margin-top:.8rem;">
                synthetic breathe sessions are generated at 12–24 bpm</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough samples for a respiration window.")

# --------------------------------------------------------------------------- #
with tab_model:
    st.markdown('<div class="sect">Model comparison</div>', unsafe_allow_html=True)
    if metrics:
        items = sorted(metrics["models"].items(),
                       key=lambda kv: kv[1]["macro_f1"], reverse=True)
        best = metrics["best_model"]
        rows = ""
        for name, v in items:
            pct = v["macro_f1"] * 100
            fill = "fill" if name == best else "fill dim"
            tag = ' <span class="tier">best</span>' if name == best else ""
            rows += (
                f'<div class="row"><div class="cls" style="width:auto">{name}{tag}</div>'
                f'<div class="track"><div class="{fill}" style="width:{pct:.0f}%"></div></div>'
                f'<div class="pct">{v["macro_f1"]:.3f}</div></div>'
            )
        st.markdown(f'<div class="card"><div class="conf" '
                    f'style="margin-top:0">{rows}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sect" style="margin-top:1.4rem">Confusion matrices</div>',
                unsafe_allow_html=True)
    imgs = sorted(config.REPORTS_DIR.glob("confusion_*.png"))
    if imgs:
        cols = st.columns(2)
        for i, p in enumerate(imgs):
            cols[i % 2].image(str(p), caption=p.stem.replace("confusion_", ""),
                              width="stretch")

    shap_png = config.REPORTS_DIR / "shap_summary.png"
    if shap_png.exists():
        st.markdown('<div class="sect" style="margin-top:1.4rem">Explainability (SHAP)</div>',
                    unsafe_allow_html=True)
        st.image(str(shap_png), width="stretch")

st.markdown('<div class="foot">WiFi CSI Human Sensing · device-free RF sensing on ESP32 · '
            'synthetic · UT-HAR · live capture</div>', unsafe_allow_html=True)
