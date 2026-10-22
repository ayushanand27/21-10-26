"""v8a-e: decision variants of the v6 alignment bag (and of the 0.7/0.3 blend with v3), all inside the range supported
by the hold-out. Uses the test probabilities saved by scripts/v6_v7_alignbag.py and scripts/v3_pseudolabel.py.
usage: python scripts/v8_variants.py WORK"""
import sys, numpy as np, pandas as pd
S = sys.argv[1]; F = f'{S}/feat'
P = np.load(f'{F}/v6_test_preds.npy', allow_pickle=True).item()               # LightGBM x3 under base / equal / mix alignments
ids = pd.read_parquet(f'{S}/data/test.parquet', columns=['ID']).ID.values
bag3 = np.mean([P[k] for k in ('base', 'eq', 'mix')], 0)                         # = v6 probabilities (v6 = bag3 >= 0.465)
v3 = pd.read_csv(f'{F}/v3_test_probs.csv').set_index('ID')['p_v3'].reindex(ids).values
cands = {'v8a_bag3_thr0.44': (bag3, 0.44), 'v8b_bag3_thr0.49': (bag3, 0.49), 'v8c_bag3_thr0.52': (bag3, 0.52),
         'v8d_0.7bag3_0.3v3_thr0.52': (0.7 * bag3 + 0.3 * v3, 0.52), 'v8e_0.7bag3_0.3v3_thr0.47': (0.7 * bag3 + 0.3 * v3, 0.47)}
for name, (p, t) in cands.items():
    pred = (p >= t).astype(int)
    pd.DataFrame({'ID': ids, 'target': pred.astype(str)}).to_csv(f'{S}/share/{name}.csv', index=False)
    print(f'{name:28s} predicted anomaly rate {pred.mean() * 100:.4f}%')
