"""Explainability for the CSI activity model.

Runs SHAP on the saved feature model (which features drive each activity), and
falls back to sklearn permutation importance if SHAP isn't installed — same
pattern as the NIDS project. Output: reports/shap_summary.png.
"""

from __future__ import annotations

import numpy as np
import joblib

from . import config, datasets, preprocess, features, train

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


def _rebuild_features(source="synthetic"):
    csi, y, info = datasets.load(source)
    sessions = info.get("session") if isinstance(info, dict) else None
    Xw, yw = train._windows_from_stream(csi, y, sessions)
    return features.features_for_windows(Xw), yw


def main(source="synthetic"):
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    X, y = _rebuild_features(source)
    Xs = scaler.transform(X)
    names = features.FEATURE_NAMES

    try:
        import shap
        print("==> Computing SHAP values")
        bg = shap.sample(Xs, min(100, len(Xs)), random_state=config.RANDOM_STATE)
        explainer = shap.Explainer(model.predict, bg)
        sv = explainer(Xs[:300])
        if _HAVE_PLT:
            shap.summary_plot(sv, features=Xs[:300], feature_names=names, show=False)
            plt.tight_layout()
            plt.savefig(config.REPORTS_DIR / "shap_summary.png", dpi=130,
                        bbox_inches="tight")
            plt.close()
        print("   wrote reports/shap_summary.png")
        return
    except Exception as e:
        print(f"   SHAP unavailable ({e}); falling back to permutation importance")

    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, Xs, y, n_repeats=8,
                               random_state=config.RANDOM_STATE)
    order = np.argsort(r.importances_mean)
    if _HAVE_PLT:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([names[i] for i in order], r.importances_mean[order])
        ax.set_title("Permutation importance"); fig.tight_layout()
        fig.savefig(config.REPORTS_DIR / "shap_summary.png", dpi=130)
        plt.close(fig)
    print("   wrote reports/shap_summary.png (permutation importance)")


if __name__ == "__main__":
    main()
