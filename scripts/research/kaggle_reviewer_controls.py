#!/usr/bin/env python3
"""Kaggle reviewer controls — requires archived datasets. No fake results.

Run:
  python scripts/research/kaggle_reviewer_controls.py --root /kaggle/working --matched-ablation

Archived checkpoints are not independently authenticated. V20 is a second look.
"""
from pathlib import Path
import argparse, json, subprocess, shutil
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
FEATURE_CONTEXT=['feat_ret_1d','feat_ret_5d','feat_ret_20d',
 'feat_rolling_vol_20d','feat_rsi_14','feat_relative_volume']
def restore(folder,slug,names):
    folder.mkdir(parents=True,exist_ok=True)
    if not all((folder/n).exists() for n in names):
        if not shutil.which('kaggle'):raise FileNotFoundError(str(folder))
        subprocess.run(['kaggle','datasets','download','-d',slug,'-p',str(folder),'--unzip'],check=True)
    for name in names:
        if not (folder/name).exists():raise FileNotFoundError(str(folder/name))
def data(path):
    d=pd.read_parquet(path)
    for col in ('date','target_date'):d[col]=pd.to_datetime(d[col]).dt.normalize()
    assert d.session_id.is_unique and (d.target_date>d.date).all()
    return d
def labels(d):
    y=d.reaction_label.map({-1:0,0:1,1:2})
    assert y.notna().all()
    return y.to_numpy(int)
def price(d,fields):
    x=d[fields].reset_index(drop=True).copy()
    x['feat_rolling_avg_vol_20d']=np.log1p(x.feat_rolling_avg_vol_20d.clip(lower=0))
    assert np.isfinite(x.to_numpy(float)).all()
    return x.astype('float32')
def context(d,fields):
    x=price(d,fields)
    m=d.groupby('date')[FEATURE_CONTEXT].median().reindex(d.date.to_numpy()).reset_index(drop=True)
    assert m.notna().all().all()
    for c in FEATURE_CONTEXT:x['market_'+c[5:]]=m[c].to_numpy()
    for days in (1,5,20):
        x[f'relative_return_{days}d']=x[f'feat_ret_{days}d']-x[f'market_ret_{days}d']
    x['relative_volatility']=x.feat_rolling_vol_20d/(x.market_rolling_vol_20d+.003)
    x['relative_rsi']=x.feat_rsi_14-x.market_rsi_14
    x['volume_surprise']=np.log1p(x.feat_relative_volume.clip(lower=0))
    x['vol_adjusted_momentum']=x.relative_return_5d/(x.feat_rolling_vol_20d+.005)
    x['market_direction']=np.sign(x.market_ret_5d)
    assert x.shape[1]==30 and np.isfinite(x.to_numpy(float)).all()
    return x.astype('float32')
def f1(y,p):return float(f1_score(y,p,average='macro',labels=[0,1,2],zero_division=0))
def score(y,p):return {'macro_f1':f1(y,p),'accuracy':float(accuracy_score(y,p)),
 'neutral_share':float(np.mean(p==1))}
def adjust(prob,w):
    pr=prob.copy()
    pr[:,1]*=w
    return pr.argmax(axis=1)
def date_ci(d,y,a,b,reps=10000):
    groups=list(d.groupby('date').indices.values())
    cma=np.array([confusion_matrix(y[i],a[i],labels=[0,1,2]) for i in groups])
    cmb=np.array([confusion_matrix(y[i],b[i],labels=[0,1,2]) for i in groups])
    rng=np.random.default_rng(20261021); diffs=[]
    def mf(cm):
        den=cm.sum(0)+cm.sum(1)
        return np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0).mean()
    for _ in range(reps):
        ids=rng.integers(len(groups),size=len(groups))
        diffs.append(float(mf(cmb[ids].sum(0))-mf(cma[ids].sum(0))))
    return list(map(float,np.quantile(diffs,[.025,.975])))
def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/kaggle/working'))
    ap.add_argument('--out',type=Path,default=Path('/kaggle/working/marketreactionlab_reviewer_controls'))
    ap.add_argument('--matched-ablation',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    p9=a.root/'marketreactionlab_v9_2026'
    p13=a.root/'marketreactionlab_v13_context'
    restore(p9,'abhijeetnayak63/marketreactionlab-v9-prices-2026',
      ['report.json','v9_price_only_xgboost.joblib','train.parquet','validation_2024.parquet','sealed_2025_2026.parquet'])
    restore(p13,'abhijeetnayak63/marketreactionlab-v13-context-20261019',
      ['v13_model_pre2025.joblib','report.json'])
    cols=json.loads((p9/'report.json').read_text())['features']
    train=data(p9/'train.parquet');dev=data(p9/'validation_2024.parquet');test=data(p9/'sealed_2025_2026.parquet')
    assert train.target_date.max()<dev.date.min() and dev.target_date.max()<test.date.min()
    assert len(dev)==33885 and len(test)==60480
    m9=joblib.load(p9/'v9_price_only_xgboost.joblib')
    m13=joblib.load(p13/'v13_model_pre2025.joblib')
    for m in (m9,m13):
        assert np.array_equal(m.classes_,[0,1,2])
    qd={'v9':m9.predict_proba(price(dev,cols)),'v13':m13.predict_proba(context(dev,cols))}
    qt={'v9':m9.predict_proba(price(test,cols)),'v13':m13.predict_proba(context(test,cols))}
    yd=labels(dev);yt=labels(test)
    grid=np.round(np.arange(.6,1.401,.025),3)
    # Freeze neutral weights using DEVELOPMENT 2024 only.
    best={name:max(grid,key=lambda w:(f1(yd,adjust(qd[name],w)),-abs(w-1))) for name in qd}
    report={'research_status':'EXPLORATORY; both 2024 and 2025-26 previously inspected',
     'weights_chosen_on_2024':{k:float(v) for k,v in best.items()},
     'train_window_provenance':'replay scripts say 2020-2023; checkpoint-level metadata unverified',
     '2024':{},'2025_2026':{},'newly_fit_controls':{}}
    for year,yy,qq,key in [('2024',yd,qd,'2024'),('2025_2026',yt,qt,'2025_2026')]:
        for name in ('v9','v13'):
            report[key][name]={'default':score(yy,adjust(qq[name],1.)),
              '2024_frozen_weight':score(yy,adjust(qq[name],best[name]))}
    report['2025_2026']['ci95_V13minusV9_2024selected']=date_ci(
      test,yt,adjust(qt['v9'],best['v9']),adjust(qt['v13'],best['v13']))
    volatility=LogisticRegression(max_iter=500,class_weight='balanced',random_state=42)
    x=lambda df:np.log1p(df.feat_rolling_vol_20d.clip(lower=0)).to_numpy().reshape(-1,1)
    volatility.fit(x(train),labels(train))
    report['newly_fit_controls']['volatility_only_logistic']={'2024':score(yd,volatility.predict(x(dev))),
     '2025_2026':score(yt,volatility.predict(x(test)))}
    if a.matched_ablation:
        for feature_set,extract in [('price_only',price),('context_plus_price',context)]:
            clf=LGBMClassifier(objective='multiclass',num_class=3,n_estimators=450,num_leaves=15,
              learning_rate=.025,min_child_samples=80,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=42)
            ly=labels(train)
            clf.fit(extract(train,cols),ly,sample_weight=compute_sample_weight('balanced',ly))
            report['newly_fit_controls'][feature_set]={'2024':score(yd,clf.predict(extract(dev,cols))),
               '2025_2026':score(yt,clf.predict(extract(test,cols)))}
            joblib.dump(clf,a.out/f'matched_lgbm_{feature_set}.joblib')
    (a.out/'reviewer_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
