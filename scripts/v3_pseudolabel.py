import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
import build_lib as B
from pipeline import label_features, merge_timeline, MERGE_W
from softalign import soft_features, SOFT_NAMES
from sklearn.metrics import average_precision_score, f1_score
S=sys.argv[1]; F=f'{S}/feat'; OUT=f'{S}/share'; t0=time.time()
def log(*a): print(*a, f'[{(time.time()-t0)/60:.1f} min]', flush=True)
y_all=B.dtr['y'].values.astype(float); pool,H=B.pool,B.H
LABC=None
def soft_relabel(bst, cols, X_unl, pos_unl, day_unl, lab_pos, lab_y, lab_day, tgt):
    """X_unl: features of ALL unlabelled rows (targets + others). Soft pseudo-labels = stage-1 p for every unlabelled row."""
    p=bst.predict(X_unl[cols])
    lp=np.r_[lab_pos,pos_unl]; ly=np.r_[lab_y,p]; ld=np.r_[lab_day,day_unl]
    LF=label_features(pos_unl[tgt],lp,ly,day_unl[tgt],ld); LFh=label_features(pos_unl[tgt],lp,(ly>=0.5).astype(float),day_unl[tgt],ld)
    for c in ('dist_prev_anom','dist_next_anom','dist_anom'): LF[c]=LFh[c].values
    X=X_unl.loc[tgt,cols].copy()
    for c in LABC: X[c]=LF[c].values.astype(np.float32)
    return p[tgt], bst.predict(X)
# ================= 1) sim world: paired comparison + week-block bootstrap
ev2=B.view(pool,[H],H,with_test=True).reset_index(drop=True)
seq=np.sort(pool); g=np.sort(H)
pa,pb=merge_timeline(B.Str[seq],B.day_tr[seq],B.Str[g],B.day_tr[g]); allr=np.r_[seq,g]; seq=allr[np.argsort(np.r_[pa,pb],kind='stable')]
p1,_=merge_timeline(B.Str[seq],B.day_tr[seq],B.Ste,B.day_te); pos_tr=np.empty(len(B.dtr),np.int64); pos_tr[seq]=p1
sH=pd.read_parquet(f'{F}/soft_sim_eval_T1.0.parquet'); sp=np.sort(pool)
sT=pd.DataFrame(soft_features(B.Str[sp],B.day_tr[sp],y_all[sp],B.Ste,B.day_te,MERGE_W,1.0),columns=SOFT_NAMES,index=B.dte['ID'].values)
isH=(ev2.src=='train').values
ev2[SOFT_NAMES]=np.where(isH[:,None],sH.reindex(ev2.row.values).values,sT.reindex(ev2.ID.values).values).astype(np.float32)
bst=lgb.Booster(model_file=f'{F}/lgb_sim_soft.txt'); cols=bst.feature_name()
LABC=[c for c in cols if c.startswith(('yprev','ynext','yboth','gap_','dist_','day_rate'))]
p0,pP=soft_relabel(bst,cols,ev2,ev2.pos.values,ev2.day.values,pos_tr[pool],y_all[pool],B.day_tr[pool],isH)
yH=ev2.y.values[isH].astype(int); wk=ev2.day.values[isH]//7
grid=np.linspace(0.05,0.95,181)
def bestt(yy,pp): return grid[np.argmax([f1_score(yy,pp>=t) for t in grid])]
t0_,tP=bestt(yH,p0),bestt(yH,pP)
log('sim: base F1 %.4f @%.3f | pseudo F1 %.4f @%.3f'%(f1_score(yH,p0>=t0_),t0_,f1_score(yH,pP>=tP),tP))
r=np.random.default_rng(0); W=np.unique(wk); wins=[]; diffs=[]
for b in range(200):
    sw=r.choice(W,len(W),replace=True); idx=np.concatenate([np.flatnonzero(wk==w) for w in sw])
    d=f1_score(yH[idx],pP[idx]>=tP)-f1_score(yH[idx],p0[idx]>=t0_); diffs.append(d)
diffs=np.array(diffs); log('bootstrap diff mean %.4f, 90%% CI [%.4f, %.4f], P(gain>0)=%.2f'%(diffs.mean(),*np.quantile(diffs,[.05,.95]),(diffs>0).mean()))
if (diffs>0).mean()<0.8: log('NOT significant -> stop, no v3'); sys.exit(0)
# ================= 2) final world: retrain on all train (test-like features), pseudo-label test, write v3
trF=pd.read_parquet(f'{F}/tl_final_train.parquet').join(pd.read_parquet(f'{F}/soft_final_train_T1.0.parquet'),on='row')
teF=pd.read_parquet(f'{F}/tl_final_eval.parquet').join(pd.read_parquet(f'{F}/soft_final_test_T1.0.parquet'),on='ID')
_,pos_te=merge_timeline(B.Str,B.day_tr,B.Ste,B.day_te)
pos_trF,_=merge_timeline(B.Str,B.day_tr,B.Ste,B.day_te)
idmap=pd.Series(pos_te,index=B.dte['ID'].values); assert np.all(idmap.reindex(teF.ID.values).values==teF.pos.values)
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4)
p0s,pPs=[],[]
tgt=np.ones(len(teF),bool)
for s in range(3):
    b=lgb.train({**P,'seed':42+s},lgb.Dataset(trF[cols],trF.y),1300)
    a,c=soft_relabel(b,cols,teF,teF.pos.values,teF.day.values,pos_trF,y_all,B.day_tr,tgt); p0s.append(a); pPs.append(c); log('final seed',s)
p0T,pPT=np.mean(p0s,0),np.mean(pPs,0)
test_ids=B.dte['ID'].values
o=pd.DataFrame({'ID':teF.ID.values,'p':pPT}).set_index('ID').reindex(test_ids)
sub=pd.DataFrame({'ID':test_ids,'target':(o.p.values>=tP).astype(int).astype(str)}); sub.to_csv(f'{OUT}/submission_v3_pseudolabel.csv',index=False)
pd.DataFrame({'ID':test_ids,'p_stage1':pd.Series(p0T,index=teF.ID.values).reindex(test_ids).values,'p_v3':o.p.values}).to_csv(f'{F}/v3_test_probs.csv',index=False)
log('v3 written: thr %.3f pred rate %.4f%%'%(tP,(o.p.values>=tP).mean()*100), sub.target.value_counts().to_dict(),
    '| agreement with stage-1@%.3f: %.4f'%(t0_,((o.p.values>=tP)==(pd.Series(p0T,index=teF.ID.values).reindex(test_ids).values>=t0_)).mean()))
