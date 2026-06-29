# 📡 WiFi CSI Human Sensing (ESP32)

Device-free human sensing from **WiFi Channel State Information** — infer what a
person is doing from how their body distorts the wireless channel. No camera,
works in the dark / through smoke / around corners.

Built to run **three ways** so you can start today and add hardware later:

| Source | Command | Needs |
|---|---|---|
| **Synthetic** (default) | `./run.sh` | nothing — physics-based CSI generator |
| **Public dataset** (UT-HAR) | `./run.sh --source ut_har` | dataset in `data/raw/ut_har/` |
| **Live ESP32** | `python -m src.live_capture …` | 2× ESP32 + `esp-csi` firmware |
| **Your captures** | `./run.sh --source captured` | recorded CSVs in `data/raw/` |

## Capabilities (tiered)
- **Tier 1 — Presence:** empty vs occupied
- **Tier 2 — Activity:** sit / stand / walk / **fall detection**
- **Tier 3 — Respiration:** breaths-per-minute on a stationary subject

## Quickstart (no hardware)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh                 # generate → train → explain → dashboard
```
Then open the Streamlit dashboard: **Live Replay** streams held-out CSI and shows
the live activity prediction + CSI heatmap; **Respiration** estimates breaths/min;
**Model** shows confusion matrices and SHAP.

## How it works
```
ESP32 TX ──packets@100Hz──▶ ESP32 RX (esp-csi) ──CSI──▶ host
                                                          │
   csi_parser ─ amplitude/phase, sanitize, drop nulls     │
   preprocess ─ Hampel → low-pass → standardise → window  │
   features   ─ motion/variance/entropy  +  respiration   │
   train      ─ RandomForest + MLP  [+ optional CNN-LSTM] │
   dashboard  ─ live prediction · heatmap · bpm · SHAP ◀──┘
```

## Results (synthetic, 1,512 windows · 6 classes)

| Model | Input | Macro-F1 | walk / fall / breathe | empty / sit / stand |
|---|---|--:|---|---|
| **RandomForest** | engineered features | **0.82** | 1.00 / 1.00 / 0.98 | 0.78 / 0.56 / 0.56 |
| CNN-LSTM | raw CSI windows | 0.76 | 1.00 / 1.00 / 0.99 | 0.47 / 0.42 / 0.69 |
| MLP | engineered features | 0.67 | — | — |

The **CNN-LSTM learns the dynamic activities (walk/fall/breathe ≈ 1.0) straight from
raw CSI** with zero hand-crafted features — the architecture that scales to real
multi-session captures. On this *small synthetic* set it overfits the near-static
classes (`empty`/`sit`/`stand`), so the feature-based **RandomForest still wins
overall**. Deep models are data-hungry; the gap is expected to close with real,
varied captures. Numbers are pipeline validation, not field performance.

## Project layout
```
src/
  config.py            paths, CSI geometry, classes, hyper-params
  csi_parser.py        decode ESP32 CSI, amplitude/phase, phase sanitisation
  preprocess.py        Hampel + low-pass denoise, subcarrier select, windowing
  features.py          statistical features + respiration-rate (FFT/PCA)
  generate_synthetic.py physics-based CSI generator (6 classes)
  datasets.py          synthetic / UT-HAR loaders, label normalisation
  train.py             train+eval, save model/scaler/metrics/confusion
  cnn_lstm.py          optional 1D-CNN-LSTM (PyTorch)
  explain.py           SHAP (falls back to permutation importance)
  live_capture.py      serial/UDP collector from a real ESP32
dashboard/app.py       Streamlit UI
firmware/README.md     flashing the two ESP32 boards (esp-csi)
```

## Hardware (≈ $20–35)
2× ESP32-S3 dev boards (one TX, one RX) + USB cables. See **`firmware/README.md`**.

## Honest limitations
- Single antenna (no MIMO) → great for coarse sensing, weak for fine gestures.
- Models **do not transfer across rooms** — train per environment.
- Respiration needs a still subject and careful setup.
- Synthetic data validates the *pipeline*; real-world numbers come from your captures.

> Dependencies degrade gracefully: scipy/matplotlib/shap/torch/pyserial are all
> optional — the core pipeline runs on numpy + scikit-learn + streamlit alone.
