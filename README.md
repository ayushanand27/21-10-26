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
| upper bound with perfect timeline positions | 0.956 | 0.897 |

Class-0 F1 and accuracy are ≈ 0.999 for every reasonable model.

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
`cache/`. A full CPU-only run takes about 2 h on 4 cores with 16 GB of RAM. On a Kaggle GPU, XGBoost, CatBoost and the PyTorch models run on CUDA. Set `SMOKE=1` for a 4-minute end-to-end check on every
25th day.

## Repository layout

```
anomaly_detection.ipynb       # the full analysis (run it to produce outputs + submissions)
submission.csv / .parquet     # written by the notebook: final test predictions (ID, target)
submission_probabilities.csv  # written by the notebook: ensemble anomaly probability per test row
data/                         # competition data (CC0)
requirements.txt
```
