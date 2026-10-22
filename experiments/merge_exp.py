import sys, numpy as np, pandas as pd, numba, time
D=sys.argv[1]
tr=pd.read_parquet(D+'/train.parquet'); tr['target']=tr.target.astype(int)
def feats(df):
    return np.column_stack([np.log(df.X1), df.X2, np.log(df.X3.clip(lower=1)), np.log(df.X4.clip(lower=1)), df.X5]).astype(np.float64)
F=feats(tr)
F=(F-F.mean(0))/F.std(0)
tr['blk']=(tr.Date!=tr.Date.shift()).cumsum()

@numba.njit(cache=True)
def merge(A,B,w):
    n=A.shape[0]; m=B.shape[0]; INF=1e18
    # cost[i,j,k]: placed i of A, j of B, last from k (0=A,1=B)
    C=np.full((n+1,m+1,2),INF); P=np.zeros((n+1,m+1,2),np.int8)
    C[1,0,0]=0.0
    if m>0: C[0,1,1]=0.0
    def d(x,y):
        s=0.0
        for t in range(x.shape[0]): s+=w[t]*abs(x[t]-y[t])
        return s
    for i in range(n+1):
        for j in range(m+1):
            for k in range(2):
                c=C[i,j,k]
                if c>=INF: continue
                last = A[i-1] if k==0 else B[j-1]
                if i<n:
                    v=c+d(last,A[i])
                    if v<C[i+1,j,0]: C[i+1,j,0]=v; P[i+1,j,0]=k
                if j<m:
                    v=c+d(last,B[j])
                    if v<C[i,j+1,1]: C[i,j+1,1]=v; P[i,j+1,1]=k
    # backtrack -> order: sequence of (src,idx)
    k=0 if C[n,m,0]<=C[n,m,1] else 1
    i=n; j=m; src=np.empty(n+m,np.int8); L=n+m
    while i+j>0:
        L-=1; src[L]=k; pk=P[i,j,k]
        if k==0: i-=1
        else: j-=1
        k=pk
    return src
w=np.array([1.,2.,.5,.5,1.])
rng=np.random.default_rng(0)
errs_dp=[];errs_rk=[]; t0=time.time()
for b,g in list(tr.groupby('blk'))[:200]:
    idx=g.index.values; N=len(idx)
    mask=rng.random(N)<0.2
    A=F[idx[~mask]]; B=F[idx[mask]]
    src=merge(A,B,w)
    truepos=np.arange(N)
    # estimated positions of B rows in merged sequence
    posB=np.nonzero(src==1)[0]; tB=truepos[mask]
    errs_dp.append(np.abs(posB-tB))
    m=mask.sum(); est=(np.arange(m)+0.5)/m*N
    errs_rk.append(np.abs(est-tB))
e1=np.concatenate(errs_dp); e2=np.concatenate(errs_rk)
print('time',time.time()-t0)
for nm,e in [('dp',e1),('rank',e2)]: print(nm,'mean',e.mean(),'median',np.median(e),'exact',(e==0).mean(),'<=2',(e<=2).mean())
