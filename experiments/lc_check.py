import sys, os, time, numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
from pipeline import load, decode, fix_dates, sensor_matrix, standardise, _merge_two, MERGE_W
from learned_cost import int_codes, fit_tables, merge_learned, merge_mixed
S=sys.argv[1]
tr,te=load(f'{S}/data'); d=decode(tr); d['day'],_=fix_dates(tr); te_d=decode(te); te_d['day'],_=fix_dates(te)
Str,_=standardise(sensor_matrix(d),sensor_matrix(te_d)); I=int_codes(d); day=d.day.values; y=tr.target.values
hide=np.random.default_rng(42).random(len(d))<0.2
T=fit_tables(I[~hide], day[~hide])          # fitted on visible rows only (their file order, gaps included)
r=np.random.default_rng(1); days=r.choice(day.max()+1,150,replace=False)
res={}
variants={'hand-made L1 (current)':None, 'mixed lam=0.05':0.05, 'mixed lam=0.15':0.15, 'mixed lam=0.4':0.4}
for name,alpha in variants.items():
    e=[]; bnd=[]; t=time.time()
    for dd in days:
        idx=np.flatnonzero(day==dd); hid=hide[idx]
        if hid.sum()==0 or (~hid).sum()==0: continue
        if alpha is None: src=_merge_two(Str[idx[~hid]],Str[idx[hid]],MERGE_W)
        else: src=merge_mixed(Str[idx[~hid]],Str[idx[hid]],MERGE_W,I[idx[~hid]],I[idx[hid]],T,np.array([2,2,1,1,1.]),alpha)
        err=np.abs(np.flatnonzero(src==1)-np.flatnonzero(hid)); e.append(err)
        yy=y[idx]; edge=np.zeros(len(idx),bool); ch=np.flatnonzero(yy[1:]!=yy[:-1]); edge[ch]=True; edge[ch+1]=True
        bnd.append(err[edge[hid]])
    e=np.concatenate(e); b=np.concatenate(bnd)
    print(f'{name:26s} exact {np.mean(e==0):.3f}  within±1 {np.mean(e<=1):.3f}  mean|err| {e.mean():.2f}   | at episode edges: exact {np.mean(b==0):.3f} ±1 {np.mean(b<=1):.3f} (n={len(b)})  {time.time()-t:.0f}s',flush=True)
