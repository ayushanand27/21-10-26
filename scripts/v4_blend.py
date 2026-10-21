"""v4: average of the v2 ensemble and v3 pseudo-label probabilities; threshold set so that the predicted anomaly rate
equals the mean of the two parents' rates (both thresholds were chosen on the hold-out).
usage: python scripts/v4_blend.py WORK"""
import sys, numpy as np, pandas as pd
S = sys.argv[1]
a = pd.read_csv(f'{S}/share/submission_v2_probabilities.csv').set_index('ID')['p_anomaly']
b = pd.read_csv(f'{S}/feat/v3_test_probs.csv').set_index('ID')['p_v3'].reindex(a.index)
p = (a + b) / 2
rate = ((a >= 0.515).mean() + (b >= 0.46).mean()) / 2          # v2 hold-out threshold 0.515, v3 hold-out threshold 0.46
t = float(np.quantile(p, 1 - rate))
sub = pd.DataFrame({'ID': a.index, 'target': (p.values >= t).astype(int).astype(str)})
sub.to_csv(f'{S}/share/submission_v4_blend_v2_v3.csv', index=False)
print('threshold', round(t, 3), 'predicted rate %.4f%%' % ((p >= t).mean() * 100), sub.target.value_counts().to_dict())
