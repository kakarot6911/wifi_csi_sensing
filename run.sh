#!/usr/bin/env bash
# One-shot pipeline: generate data -> train+evaluate -> explain -> dashboard.
# Usage: ./run.sh                  (synthetic, full run, then dashboard)
#        ./run.sh --source ut_har  (train on the public dataset in data/raw/ut_har)
#        ./run.sh --no-dash        (everything except the dashboard)
set -euo pipefail
cd "$(dirname "$0")"

SOURCE="synthetic"
DASH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --no-dash) DASH=0; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$SOURCE" == "synthetic" ]]; then
  echo "==> [1/4] Generating synthetic ESP32-style CSI"
  python -m src.generate_synthetic
else
  echo "==> [1/4] Using real dataset: $SOURCE"
fi

echo "==> [2/4] Training & evaluating models"
python -m src.train --source "$SOURCE"

echo "==> [3/4] Building explainability report"
python -m src.explain || echo "   (explain step skipped — optional deps missing)"

if [[ "$DASH" -eq 0 ]]; then
  echo "==> Done. Launch later with: streamlit run dashboard/app.py"
  exit 0
fi

echo "==> [4/4] Launching dashboard"
streamlit run dashboard/app.py
