# Sensor Anomaly Detection: Energy Manufacturing Plant

This project predicts anomalies (`target` = 1) from five sensor readings (`X1…X5`) captured every minute in an energy
manufacturing plant. It is scored on **F1 and accuracy for each class**.

Everything lives in **[`anomaly_detection.ipynb`](anomaly_detection.ipynb)**. Running it writes `submission.csv` / `submission.parquet`,
which follow the `sample_submission` format with `target` as the string `'0'`/`'1'`.

## Key findings

1. **The sensors were exp/log-encoded.** `log(X1)`, `exp(X2)`, `log(X3)`, `log(X4)` and `exp(X5)` give clean values:
   a ~240 V voltage, integer counters and a level from 1 to 32. The "outliers" of up to 1e38 are just `exp(88)`, so we
   decode the values instead of clipping them.
2. **`Date` has day and month swapped for days ≤ 12.** The rows are stored in true time order, so the calendar can be repaired.
3. **There is one reading per minute, and the test set is a random 20 % of the same minutes as train.** Rows inside each
   file are in time order: lag-1 autocorrelation is 0.91–0.98, and anomalies come in episodes with P(y=1 | previous y=1) = 0.74.
4. **We rebuild the full timeline.** An exact dynamic-programming merge interleaves train and test minutes for each day by
   minimising sensor jumps. A held-out minute is put back within about 1 row of its true position (median error).

## Approach

| Stage | Details |
|---|---|
| Features | Lags and leads; centred, past and future rolling statistics (5–241 min); deviations; daily z-scores and ranks; voltage-drop and counter run-length features; calendar features. Also leave-self-out **neighbour-label** features: the labels of nearby training minutes, the distance to the nearest anomaly and the day's anomaly rate. |
| Leakage control | A row's own label is never used. Training rows get *test-like* features out-of-fold: each of 10 chunks is hidden, re-inserted by the DP, and featurised from the other labels. |
| Validation | 20 % of train is pushed through exactly the same pipeline as the test set (honest hold-out). The models also get a rolling-origin backtest by year. |
| Classical models | Logistic Regression, Linear SVM, RBF-SVM, KNN and Decision Tree, each tuned with `GridSearchCV` + `GroupKFold` by week. |
| Advanced models | Random Forest, XGBoost, LightGBM (random-search tuned), CatBoost, a PyTorch MLP and a dilated 1-D CNN over ±32-minute windows. |
| Ensemble | Weighted blend with weights and threshold cross-fitted on two halves of the hold-out. The threshold is chosen to maximise anomaly-class F1. |
| Robustness | Threshold sensitivity, per-year backtest, reliability diagram, residuals by month and hour, error analysis and SHAP contributions. |

## Results (honest hold-out, 327 852 rows, 2 680 anomalies)

These results come from development runs of the same pipeline (LightGBM, threshold tuned for anomaly-class F1). The notebook
prints the full table for every model, the ensemble, and per-class precision, recall/accuracy and F1.

| Features | PR-AUC | F1 (anomaly) |
|---|---|---|
| raw decoded sensors only | 0.70 | 0.64 |
| + temporal context (sensor-only model) | 0.93 | 0.86 |
| + neighbour labels | 0.94 | 0.87 |
| + test-like (out-of-fold) training features | **0.949** | **0.883** |
| **+ soft (posterior) alignment, LightGBM ×3 seeds** | **0.952** | **0.886** |
| upper bound with perfect timeline positions | 0.956 | 0.897 |

Class-0 F1 and accuracy are ≈ 0.999 for every reasonable model.

### Experiment log (all measured on the honest hold-out; scripts in `experiments/`)

| Idea | Hold-out effect | Decision |
|---|---|---|
| Test-like (out-of-fold re-placed) training features | F1 0.877 → 0.884 | ✅ kept |
| Soft (posterior) alignment features, T = 1 | F1 0.8836 → 0.8866, PR-AUC 0.9491 → 0.9508 | ✅ kept |
| Seed bagging (LightGBM ×3) | lower variance | ✅ kept |
| Soft pseudo-labels of unlabelled minutes | +0.0023 F1, gain in 93 % of week-block bootstrap resamples | ✅ kept (v3) |
| Stage-2 stacking on neighbouring predictions | −2 pts | ❌ |
| Post-hoc episode smoothing | no gain | ❌ |
| 20 instead of 10 out-of-fold chunks | −0.5 pt | ❌ |
| Context-specific thresholds (edge / isolated) | no gain (0.8835 vs 0.8835) | ❌ |
| Leave-out anomaly rate of identical sensor vectors | −0.6 pt cross-fitted | ❌ |
| XGBoost / CatBoost in the blend | blend ≈ 87 % LightGBM, no gain | kept only as a diversity pick |
| Learned alignment cost (−log P(Δ) per sensor) | worse exact placement at episode edges (0.538 → 0.44–0.51) | ❌ |

**Error anatomy:** ~60 % of the remaining errors sit at episode boundaries (±1-minute ambiguity), ~20 % are isolated anomalies far
from any labelled one. With perfect minute positions a single LightGBM reaches 0.897 on the hold-out, so the practical ceiling of
this approach on the leaderboard is ≈ 0.89–0.90. **Kept:** seed bagging of the boosted models, plus a
second, rate-matched submission as a hedge against calibration shift.

## Leaderboard

Public leaderboard metric = F1 of the anomaly class (verified: hold-out F1₁ predicted the public score). Public = ~50 % of test.

| Submission (`submissions/`) | What it is | Public F1 |
|---|---|---|
| `v4_blend_v2_v3.csv` | average of the v2 ensemble and v3 | **0.89205** |
| `v2_hedge_rate_matched.csv` | v2 ensemble, rate-matched threshold | 0.89193 |
| `v2_ensemble.csv` | 0.87 LightGBM×3 + 0.12 XGBoost + CatBoost, soft alignment | 0.89191 |
| `v2a_lgb3_soft_PRIMARY.csv` | LightGBM ×3 seeds + soft alignment | 0.89133 |
| `v3_pseudolabel.csv` | v2a + soft pseudo-labels of test minutes (+0.0023 on hold-out) | 0.89036 |
| `v1_lgb_public_0.8897.csv` | single LightGBM, hard alignment | 0.88966 |

All six are selected for the private leaderboard (Kaggle keeps the best). A simulated public/private split on the hold-out
(`submissions/private_sim.py`) shows the private−public difference is ±0.015 (90 % range) purely from sampling noise, with no
sign of over-fitting: every choice above was made on the hold-out, never on the public board.
`submissions/*.py` are the standalone scripts that produced the files.

## How to run

**On Kaggle (recommended, GPU):**
1. Create a notebook and choose *File → Import Notebook* with `anomaly_detection.ipynb`.
2. Use *Add Input* to attach the dataset. The notebook finds `train.parquet` anywhere under `/kaggle/input/`.
3. Set *Settings → Accelerator → GPU*, then *Run All*. Submissions are written to `/kaggle/working/`.

**Locally:**

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute anomaly_detection.ipynb --ExecutePreprocessor.timeout=12000
```

The data goes in `data/` (`train.parquet`, `test.parquet`, `sample_submission.parquet`). Expensive feature tables are cached in
`cache/`. A full-strength run (`FAST_MODE = False`, the default) takes about 2.5–3 h on Kaggle (T4 GPU + 4 vCPU) — use *Save Version → Save & Run All* so it runs in the background. `FAST_MODE = True` cuts it to about 1.5 h. On a Kaggle GPU, XGBoost, CatBoost and the PyTorch models run on CUDA. Set `SMOKE=1` for a 4-minute end-to-end check on every
25th day.

## Repository layout

```
anomaly_detection.ipynb       # the full analysis (run it to produce outputs + submissions)
submission.csv / .parquet     # written by the notebook: final test predictions (ID, target)
submission_probabilities.csv  # written by the notebook: ensemble anomaly probability per test row
submission_rate_matched.csv   # written by the notebook: hedge submission (rate-matched threshold)
submissions/                  # submitted prediction files + the scripts that produced them
experiments/                  # scripts of the hold-out experiments in the log above
data/                         # competition data (CC0)
requirements.txt
```
