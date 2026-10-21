import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, catboost as cb, warnings; warnings.filterwarnings('ignore')
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), p) for p in ('.', '../src', '../scripts')]
from pipeline import load, decode, fix_dates, sensor_matrix, standardise, MERGE_W
from softalign import soft_features, SOFT_NAMES
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from scipy.optimize import minimize
S=sys.argv[1]; F=f'{S}/feat'; OUT=f'{S}/share'; t0=time.time(); T=1.0
def log(*a): print(*a, f'[{(time.time()-t0)/60:.1f} min]', flush=True)
tr,te=load(f'{S}/data'); dtr,dte=decode(tr),decode(te); dtr['day'],_=fix_dates(tr); dte['day'],_=fix_dates(te)
Str,Ste=standardise(sensor_matrix(dtr),sensor_matrix(dte)); y=tr.target.values; day=dtr.day.values; dayt=dte.day.values
# ---------------- soft features for the final world
p_tr=f'{F}/soft_final_train_T{T}.parquet'; p_te=f'{F}/soft_final_test_T{T}.parquet'
trF=pd.read_parquet(f'{F}/tl_final_train.parquet'); teF=pd.read_parquet(f'{F}/tl_final_eval.parquet')
if not os.path.exists(p_tr):
    parts=[]
    rows=trF.row.values; ch=trF.chunk.values
    for k in range(10):
        ck=np.sort(rows[ch==k]); lab=np.sort(rows[ch!=k])
        parts.append(pd.DataFrame(soft_features(Str[lab],day[lab],y[lab],Str[ck],day[ck],MERGE_W,T),columns=SOFT_NAMES).assign(row=ck))
    pd.concat(parts).set_index('row').to_parquet(p_tr)
    allr=np.arange(len(dtr))
    pd.DataFrame(soft_features(Str,day,y,Ste,dayt,MERGE_W,T),columns=SOFT_NAMES).assign(ID=te.ID.values).set_index('ID').to_parquet(p_te)
log('final soft features ready')
trF=trF.join(pd.read_parquet(p_tr),on='row'); teF=teF.join(pd.read_parquet(p_te),on='ID')
trS=pd.read_parquet(f'{F}/tl_sim_train.parquet').join(pd.read_parquet(f'{F}/soft_sim_train_T{T}.parquet'),on='row')
evS=pd.read_parquet(f'{F}/tl_sim_eval.parquet').join(pd.read_parquet(f'{F}/soft_sim_eval_T{T}.parquet'),on='row')
meta=['row','pos','day','src','ID','y','chunk']; cols=[c for c in trS.columns if c not in meta]
assert all(c in teF.columns for c in cols)
XS,yS,XE,yE=trS[cols].to_numpy(np.float32),trS.y.to_numpy().astype(int),evS[cols].to_numpy(np.float32),evS.y.to_numpy().astype(int)
XF,yF,XT=trF[cols].to_numpy(np.float32),trF.y.to_numpy().astype(int),teF[cols].to_numpy(np.float32)
del trS,trF
test_ids=teF.ID.values
def best(yy,pp):
    ts=np.linspace(0.05,0.95,181); fs=[f1_score(yy,pp>=t) for t in ts]; i=int(np.argmax(fs)); return fs[i],ts[i]
def write(name,p_test,thr):
    o=pd.DataFrame({'ID':test_ids,'p':p_test}).set_index('ID').reindex(te.ID.values)
    assert o.p.notna().all()
    sub=pd.DataFrame({'ID':te.ID.values,'target':(o.p.values>=thr).astype(int).astype(str)}); sub.to_csv(f'{OUT}/{name}.csv',index=False)
    log(name,'thr',round(thr,3),'pred rate %.4f%%'%((o.p.values>=thr).mean()*100), sub.target.value_counts().to_dict())
    return o.p.values
HP,TP={},{}
# ---------------- LightGBM, 3 seeds
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_child_samples=50,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=4)
hs,ts_=[],[]
for s in range(3):
    hs.append(lgb.train({**P,'seed':42+s},lgb.Dataset(XS,yS),1300).predict(XE))
    ts_.append(lgb.train({**P,'seed':42+s},lgb.Dataset(XF,yF),1300).predict(XT)); log('lgb seed',s,'holdout F1 so far',round(best(yE,np.mean(hs,0))[0],4))
HP['lgb'],TP['lgb']=np.mean(hs,0),np.mean(ts_,0)
f,t=best(yE,HP['lgb']); log('LGB x3 holdout AP %.4f F1 %.4f @%.3f'%(average_precision_score(yE,HP['lgb']),f,t))
write('submission_v2a_lgb_soft',TP['lgb'],t)
np.save(f'{F}/v2_hold.npy',HP); np.save(f'{F}/v2_test.npy',TP)
# ---------------- XGBoost
XP=dict(n_estimators=700,learning_rate=0.05,max_depth=8,min_child_weight=5,subsample=0.8,colsample_bytree=0.6,reg_lambda=2.0,tree_method='hist',n_jobs=4,random_state=42)
HP['xgb']=xgb.XGBClassifier(**XP).fit(XS,yS).predict_proba(XE)[:,1]; TP['xgb']=xgb.XGBClassifier(**XP).fit(XF,yF).predict_proba(XT)[:,1]
f,t=best(yE,HP['xgb']); log('XGB holdout AP %.4f F1 %.4f'%(average_precision_score(yE,HP['xgb']),f)); np.save(f'{F}/v2_hold.npy',HP); np.save(f'{F}/v2_test.npy',TP)
# ---------------- CatBoost
CP=dict(iterations=1200,learning_rate=0.1,depth=8,l2_leaf_reg=3,thread_count=4,random_seed=42,verbose=0)
HP['cb']=cb.CatBoostClassifier(**CP).fit(XS,yS).predict_proba(XE)[:,1]; TP['cb']=cb.CatBoostClassifier(**CP).fit(XF,yF).predict_proba(XT)[:,1]
f,t=best(yE,HP['cb']); log('CB holdout AP %.4f F1 %.4f'%(average_precision_score(yE,HP['cb']),f)); np.save(f'{F}/v2_hold.npy',HP); np.save(f'{F}/v2_test.npy',TP)
# ---------------- blend: weights by AP, cross-fitted check, threshold by F1
names=list(HP); PH=np.column_stack([HP[n] for n in names]); PT=np.column_stack([TP[n] for n in names])
def fitw(P_,y_):
    fn=lambda w: -average_precision_score(y_,P_@(np.abs(w)/np.abs(w).sum()))
    w=minimize(fn,np.ones(P_.shape[1])/P_.shape[1],method='Nelder-Mead',options={'maxiter':300}).x; return np.abs(w)/np.abs(w).sum()
half=(evS.day.values//7)%2; cf=[]
for a in (0,1):
    A,B=half==a,half!=a; w=fitw(PH[A],yE[A]); _,t=best(yE[A],PH[A]@w); cf.append(f1_score(yE[B],PH[B]@w>=t))
W=fitw(PH,yE); pe=PH@W; f,t=best(yE,pe)
log('weights',dict(zip(names,W.round(3))),'cross-fitted F1 %.4f'%np.mean(cf),'| full holdout F1 %.4f @%.3f AP %.4f'%(f,t,average_precision_score(yE,pe)))
log('per-class on holdout: P1 %.4f R1 %.4f F1_0 %.5f'%(precision_score(yE,pe>=t),recall_score(yE,pe>=t),f1_score(yE,pe>=t,pos_label=0)))
p_test=write('submission_v2_ensemble',PT@W,t)
rate=(pe>=t).mean(); t_rate=float(np.quantile(PT@W,1-rate)); write('submission_v2_hedge_rate_matched',PT@W,t_rate)
pd.DataFrame({'ID':test_ids,'p_anomaly':PT@W}).to_csv(f'{OUT}/submission_v2_probabilities.csv',index=False)
log('DONE')
