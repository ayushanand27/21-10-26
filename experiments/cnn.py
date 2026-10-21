import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
S=sys.argv[1]; W=32
d=pd.read_parquet(f'{S}/feat/feat_sim.parquet',columns=['src','is_lab','y','x1','x2','x3','x4','x5','x2_m241','minute'])
X=d[['x1','x2','x3','x4','x5']].to_numpy(np.float32).copy()
X[:,1]=X[:,1]-d.x2_m241.to_numpy(np.float32)          # detrended voltage
X=(X-X.mean(0))/X.std(0)
lab=(d.src.eq('train')&d.is_lab).to_numpy()
Y=d.y.fillna(0).to_numpy(np.float32)
C=np.column_stack([X, np.where(lab,Y,0), lab.astype(np.float32)]).astype(np.float32)   # 7 channels
Cp=np.pad(C,((W,W),(0,0)))
fit=np.flatnonzero(lab); ho=np.flatnonzero(d.src.eq('train').to_numpy()&~lab)
def batch(idx):
    w=np.stack([Cp[i:i+2*W+1] for i in idx])          # (B, 2W+1, 7), centre = row i
    w[:,W,5:]=0                                        # never see own label
    return torch.from_numpy(w).permute(0,2,1)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(nn.Conv1d(7,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),
                          nn.Conv1d(64,64,5,padding=4,dilation=2),nn.BatchNorm1d(64),nn.ReLU(),
                          nn.Conv1d(64,64,5,padding=8,dilation=4),nn.BatchNorm1d(64),nn.ReLU())
        s.h=nn.Sequential(nn.Linear(64*2,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
    def forward(s,x):
        z=s.c(x); return s.h(torch.cat([z[:,:,W],z.mean(2)],1)).squeeze(1)
net=Net(); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4)
pos=fit[Y[fit]==1]; neg=fit[Y[fit]==0]
lossf=nn.BCEWithLogitsLoss()
def predict(idx):
    net.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(idx),8192): out.append(torch.sigmoid(net(batch(idx[i:i+8192]))).numpy())
    net.train(); return np.concatenate(out)
t0=time.time()
for ep in range(8):
    idx=np.r_[pos, np.random.choice(neg,len(pos)*15,replace=False)]; np.random.shuffle(idx)
    for i in range(0,len(idx),512):
        b=idx[i:i+512]; opt.zero_grad(); l=lossf(net(batch(b)),torch.from_numpy(Y[b])); l.backward(); opt.step()
    p=predict(ho); ts=np.linspace(0.05,0.99,95); fs=[f1_score(Y[ho],p>t) for t in ts]; j=int(np.argmax(fs))
    print(ep,f'AP={average_precision_score(Y[ho],p):.4f} F1={fs[j]:.4f}@{ts[j]:.2f} {time.time()-t0:.0f}s',flush=True)
