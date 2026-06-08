"""Train & evaluate CSI activity models, then persist artefacts + metrics.

Flow:  load (synthetic|ut_har) → denoise per session → window → features →
       train RandomForest + MLP (sklearn) [+ optional CNN-LSTM if torch] →
       pick best by macro-F1 → save model/scaler/metrics/confusion-matrices.

Run:   python -m src.train                 # synthetic
       python -m src.train --source ut_har # real public dataset
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config, datasets, preprocess, features

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


def _windows_from_stream(csi, y, sessions=None):
    """Denoise + window, respecting session boundaries when available."""
    Xw, yw = [], []
    if sessions is not None:
        for sid in np.unique(sessions):
            m = sessions == sid
            amp = preprocess.denoise_amplitude(csi[m])
            amp = preprocess.standardise(amp)
            w, lw = preprocess.window_stream(amp, y[m])
            Xw.append(w); yw.append(lw)
    else:
        amp = preprocess.standardise(preprocess.denoise_amplitude(csi))
        w, lw = preprocess.window_stream(amp, y)
        Xw.append(w); yw.append(lw)
    return np.concatenate(Xw), np.concatenate(yw)


def _confusion(name, y_true, y_pred):
    if not _HAVE_PLT:
        return
    cm = confusion_matrix(y_true, y_pred, labels=range(len(config.CLASSES)))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASSES))); ax.set_yticks(range(len(config.CLASSES)))
    ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASSES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion — {name}"); ax.set_xlabel("pred"); ax.set_ylabel("true")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def _maybe_train_cnn(Xtr, ytr, Xte, yte):
    """Train a 1D-CNN-LSTM on raw windows if PyTorch is installed; else skip."""
    try:
        import torch
        from .cnn_lstm import train_cnn_lstm
    except Exception:
        print("   (CNN-LSTM skipped — install torch to enable the deep model)")
        return None
    return train_cnn_lstm(Xtr, ytr, Xte, yte)


def main(source="synthetic"):
    print(f"==> Loading data  (source={source})")
    csi, y, info = datasets.load(source)
    sessions = info.get("session") if isinstance(info, dict) else None

    print("==> Denoising + windowing")
    Xw, yw = _windows_from_stream(csi, y, sessions)
    print(f"   {Xw.shape[0]:,} windows of shape {Xw.shape[1:]}  "
          f"(classes present: {sorted(set(yw))})")

    print("==> Extracting features")
    X = features.features_for_windows(Xw)

    Xtr, Xte, ytr, yte, Wtr, Wte = train_test_split(
        X, yw, Xw, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=yw)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    results = {}
    models = {
        "RandomForest": RandomForestClassifier(**config.RF_PARAMS),
        "MLP": MLPClassifier(**config.MLP_PARAMS),
    }
    best_name, best_model, best_f1 = None, None, -1.0
    for name, clf in models.items():
        print(f"==> Training {name}")
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xte_s)
        f1 = f1_score(yte, pred, average="macro")
        results[name] = {
            "macro_f1": round(float(f1), 4),
            "report": classification_report(
                yte, pred, labels=range(len(config.CLASSES)),
                target_names=config.CLASSES, output_dict=True, zero_division=0),
        }
        _confusion(name, yte, pred)
        print(f"   {name} macro-F1 = {f1:.3f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, clf, f1

    cnn = _maybe_train_cnn(Wtr, ytr, Wte, yte)
    if cnn is not None:
        results["CNN_LSTM"] = {"macro_f1": round(float(cnn["macro_f1"]), 4)}
        _confusion("CNN_LSTM", yte, cnn["pred"])
        print(f"   CNN-LSTM macro-F1 = {cnn['macro_f1']:.3f}")

    print(f"==> Best feature model: {best_name} (macro-F1={best_f1:.3f}) — saving")
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump({"classes": config.CLASSES}, config.LABEL_ENCODER_PATH)

    metrics = {
        "source": source,
        "n_windows": int(Xw.shape[0]),
        "best_model": best_name,
        "feature_names": features.FEATURE_NAMES,
        "models": results,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"==> Wrote {config.METRICS_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "ut_har"])
    main(ap.parse_args().source)
