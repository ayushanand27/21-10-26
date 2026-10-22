"""Density-matched pseudo-labels: training features ALSO built with (out-of-fold) soft pseudo-labels."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
import build_lib as B
from pipeline import label_features, merge_timeline, MERGE_W
from softalign import soft_features, SOFT_NAMES
from eval_utils import load_sim, baseline_pred, report, P, ROUNDS, F
t0=time.time()
def log(*a): print(*a, f'[{(time.time()-t0)/60:.1f} min]', flush=True)
y_all=B.dtr['y'].values.astype(float); pool,H=B.pool,B.H
trn, ev, cols = load_sim()
LABC=[c for c in cols if c.startswith(('yprev','ynext','yboth','gap_','dist_','day_rate'))]
chunk_of=pd.Series(trn.chunk.values,index=trn.row.values)
def positions(lab, groups):
    seq=np.sort(lab)
    for g in groups:
        g=np.sort(g); pa,pb=merge_timeline(B.Str[seq],B.day_tr[seq],B.Str[g],B.day_tr[g]); allr=np.r_[seq,g]; seq=allr[np.argsort(np.r_[pa,pb],kind='stable')]
    p1,_=merge_timeline(B.Str[seq],B.day_tr[seq],B.Ste,B.day_te); pos=np.empty(len(B.dtr),np.int64); pos[seq]=p1; return pos
def add_soft(V, lab):
    L=np.sort(lab); isT=(V.src=='test').values
    trr=np.sort(V.row.values[~isT])
    st=pd.DataFrame(soft_features(B.Str[L],B.day_tr[L],y_all[L],B.Str[trr],B.day_tr[trr],MERGE_W,1.0),columns=SOFT_NAMES,index=trr)
    sT=pd.DataFrame(soft_features(B.Str[L],B.day_tr[L],y_all[L],B.Ste,B.day_te,MERGE_W,1.0),columns=SOFT_NAMES,index=B.dte['ID'].values)
    V[SOFT_NAMES]=np.where(isT[:,None],sT.reindex(V.ID.values).to_numpy(),st.reindex(V.row.values).to_numpy()).astype(np.float32)
    return V
def relabel(V, p, lab, pos_tr, tgt_mask):
    lp=np.r_[pos_tr[lab],V.pos.values]; ly=np.r_[y_all[lab],p]; ld=np.r_[B.day_tr[lab],V.day.values]
    LF=label_features(V.pos.values[tgt_mask],lp,ly,V.day.values[tgt_mask],ld)
    LFh=label_features(V.pos.values[tgt_mask],lp,(ly>=0.5).astype(float),V.day.values[tgt_mask],ld)
    for c in ('dist_prev_anom','dist_next_anom','dist_anom'): LF[c]=LFh[c].values
    return LF[LABC].astype(np.float32)
# ---- 1) five fold models (each blind to a pair of chunks)
fold_models={}
for f in range(5):
    ks=(2*f,2*f+1); m=~trn.chunk.isin(ks).values
    fold_models[f]=lgb.train(P,lgb.Dataset(trn.loc[m,cols],trn.y[m]),ROUNDS); log('fold model',f)
# ---- 2) per chunk view: predict all unlabelled rows with the blind fold model, relabel the chunk rows
new_tr=[]
for k in range(10):
    ck=np.sort(trn.row.values[trn.chunk.values==k]); lab=np.setdiff1d(pool,ck)
    V=B.view(lab,[ck,H],np.r_[ck,H],with_test=True).reset_index(drop=True); V=add_soft(V,lab)
    pos_tr=positions(lab,[ck,H]); assert np.all(pos_tr[V.row.values[V.src.values=='train']]==V.pos.values[V.src.values=='train'])
    p=fold_models[k//2].predict(V[cols])
    tgt=np.isin(V.row.values,ck)&(V.src.values=='train')
    LF=relabel(V,p,lab,pos_tr,tgt); LF.index=V.row.values[tgt]; new_tr.append(LF); log('chunk view',k)
    del V
NT=pd.concat(new_tr)
trn2=trn.copy(); trn2[LABC]=NT.reindex(trn2.row.values).values
# ---- 3) hold-out: relabel H with the stage-1 full model (exactly like v3)
V=B.view(pool,[H],H,with_test=True).reset_index(drop=True); V=add_soft(V,pool)
pos_tr=positions(pool,[H]); st1=lgb.Booster(model_file=f'{F}/lgb_sim_soft.txt')
p=st1.predict(V[cols]); isH=(V.src.values=='train')
LFH=relabel(V,p,pool,pos_tr,isH); LFH.index=V.row.values[isH]
ev2=ev.copy(); ev2[LABC]=LFH.reindex(ev2.row.values).values
p_v3=st1.predict(ev2[cols])                      # v3-style: stage-1 model on relabelled hold-out
log('stage-2 training')
b2=lgb.train(P,lgb.Dataset(trn2[cols],trn2.y),ROUNDS); b2.save_model(f'{F}/lgb_sim_v5.txt')
p_v5=b2.predict(ev2[cols])
pb=baseline_pred(ev); np.save(f'{F}/hold_p_v3.npy',p_v3); np.save(f'{F}/hold_p_v5.npy',p_v5)
report('v3 (pseudo at test time only) vs baseline',ev,p_v3,pb)
report('v5 (density-matched training) vs baseline',ev,p_v5,pb)
report('v5 vs v3',ev,p_v5,p_v3)
report('avg(v3,v5) vs v3',ev,(p_v3+p_v5)/2,p_v3)
log('DONE')
