#!/usr/bin/env bash
# Reproduce everything: data -> analysis -> figures -> presentation
set -euo pipefail
cd "$(dirname "$0")"
python src/download_data.py
python src/prepare.py
python src/analysis.py
(cd src && python figures.py)
python src/build_deck.py
echo "Done. Open presentation/index.html"
