#!/usr/bin/env bash
# Reproduces every file in submissions/ (CPU only, ~3-4 h on 4 cores / 16 GB RAM). Run from the repo root.
set -euo pipefail
export WORK=${WORK:-work}
mkdir -p "$WORK/feat" "$WORK/share" "$WORK/logs"
[ -e "$WORK/data" ] || ln -s "$(pwd)/data" "$WORK/data"
PY="python3 -I"
# 1) timeline reconstruction + test-like out-of-fold features (sim world = 20 % hold-out, final world = real test)
$PY scripts/build_features.py "$WORK/data" "$WORK/feat" sim 10
$PY scripts/build_features.py "$WORK/data" "$WORK/feat" final 10
# 2) v1: single LightGBM                                   -> share/submission_lgb_quick.csv
$PY scripts/v1_quick_lgb.py "$WORK"
# 3) soft-alignment features for the sim world (+ hold-out check)
$PY experiments/soft_exp.py "$WORK" 1.0
# 4) v2a / v2 ensemble / rate-matched hedge                -> share/submission_v2*.csv
$PY scripts/v2_final.py "$WORK"
# 5) hold-out pseudo-label study (also saves the sim model used by v3)
$PY experiments/pseudolabel_variants.py "$WORK"
# 6) v3: soft pseudo-labels (gated by a week-block bootstrap) -> share/submission_v3_pseudolabel.csv
$PY scripts/v3_pseudolabel.py "$WORK"
# 7) v4: blend of v2 and v3                                -> share/submission_v4_blend_v2_v3.csv
$PY scripts/v4_blend.py "$WORK"
# 8) v6 / v7: alignment bagging (+ v3)
$PY scripts/v6_v7_alignbag.py
# 9) v8a-e: decision variants of v6 / the 0.7-0.3 blend
$PY scripts/v8_variants.py "$WORK"
# 10) public/private split robustness check on the hold-out
$PY scripts/private_split_sim.py "$WORK"
