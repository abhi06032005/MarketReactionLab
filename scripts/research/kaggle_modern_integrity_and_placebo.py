#!/usr/bin/env python3
"""Kaggle: recompute modern return-label arithmetic and optional timing/placebo
experiments from archived Yahoo-derived price Parquet files. Never claims an
upstream Yahoo point-in-time audit. Run with --train-controls for XGB re-fits.
"""
import argparse,json,subprocess,shutil
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
def load(p):
 d=pd.read_parquet(p)
 for c in ['date','target_date']:d[c]=pd.to_datetime(d[c]).dt.normalize()
 assert d.session_id.is_unique and (d.target_date>d.date).all()
 return d
def verify(d):
 required=['p_ref','p_tgt','spy_ref','spy_tgt','stock_return','spy_return','excess_return']
 assert set(required).issubset(d)
 assert (d[['p_ref','p_tgt','spy_ref','spy_tgt']]>0).all().all()
 v=(d.p_tgt/d.p_ref-1)-(d.spy_tgt/d.spy_ref-1)
 err=float((v-d.excess_return).abs().max())
 assert err<1e-9
 y=np.select([v<-.005,v>.005],[-1,1],default=0)
 assert np.array_equal(y,d.reaction_label.to_numpy())
 return {'rows':len(d),'max_error':err,'bad_labels':0}
def yy(d):return d.reaction_label.map({-1:0,0:1,1:2}).to_numpy(int)
def make_x(d,features,lag=False):
 cols=['lag1_'+f for f in features] if lag else features
 x=d[cols].copy();x.columns=features
 x['feat_rolling_avg_vol_20d']=np.log1p(x.feat_rolling_avg_vol_20d.clip(lower=0))
 assert np.isfinite(x.to_numpy(float)).all()
 return x.to_numpy('float32')
def clf():
 return XGBClassifier(objective='multi:softprob',num_class=3,n_estimators=450,
 max_depth=4,learning_rate=.035,min_child_weight=15,subsample=.85,
 colsample_bytree=.85,reg_lambda=10,tree_method='hist',
 eval_metric='mlogloss',n_jobs=4,random_state=42)
def metrics(y,p):
 return {'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),
 'accuracy':float(accuracy_score(y,p))}
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--root',type=Path,default=Path('/kaggle/working'))
 parser.add_argument('--output',type=Path,default=Path('/kaggle/working/mrl_integrity'))
 parser.add_argument('--train-controls',action='store_true')
 a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 folder=a.root/'marketreactionlab_v9_2026'
 names=['train.parquet','validation_2024.parquet','sealed_2025_2026.parquet','report.json']
 if not all((folder/n).exists() for n in names):
  folder.mkdir(parents=True,exist_ok=True)
  if not shutil.which('kaggle'):raise RuntimeError('Missing Kaggle dataset files')
  subprocess.run(['kaggle','datasets','download','-d','abhijeetnayak63/marketreactionlab-v9-prices-2026',
   '-p',str(folder),'--unzip'],check=True)
 train=load(folder/'train.parquet')
 dev=load(folder/'validation_2024.parquet')
 hold=load(folder/'sealed_2025_2026.parquet')
 assert train.target_date.max()<dev.date.min() and dev.target_date.max()<hold.date.min()
 result={'status':'saved-price arithmetic audit; Yahoo upstream and execution timing NOT audited',
 'train':verify(train),'validation_2024':verify(dev),'opened_2025_2026':verify(hold)}
 if a.train_controls:
  features=json.loads((folder/'report.json').read_text())['features']
  pool=pd.concat([train,dev,hold],ignore_index=True).sort_values(['ticker','date']).reset_index(drop=True)
  for field in features:pool['lag1_'+field]=pool.groupby('ticker')[field].shift(1)
  lagfields=['lag1_'+f for f in features]
  t=pool[(pool.date.dt.year<=2023)&(pool.target_date.dt.year<=2023)].dropna(subset=lagfields).copy()
  d=pool[(pool.date.dt.year==2024)&(pool.target_date.dt.year==2024)].dropna(subset=lagfields).copy()
  y=yy(t);yd=yy(d)
  result['controls_n_validation']=len(d)
  for name,use_lag in [('original',False),('one_session_lag',True)]:
   m=clf();m.fit(make_x(t,features,use_lag),y,sample_weight=compute_sample_weight('balanced',y))
   result[name]=metrics(yd,m.predict(make_x(d,features,use_lag)))
  rng=np.random.default_rng(20261021);shuffled=y.copy()
  for idx in t.reset_index(drop=True).groupby('date').indices.values():
   shuffled[idx]=rng.permutation(shuffled[idx])
  m=clf();m.fit(make_x(t,features),shuffled,sample_weight=compute_sample_weight('balanced',shuffled))
  result['within_date_label_permutation']=metrics(yd,m.predict(make_x(d,features)))
  result['controls_warning']='Retrospective 2024; only one permutation seed, NOT a formal permutation p-value'
 else:result['controls_status']='NOT RUN: pass --train-controls'
 (a.output/'modern_integrity_and_placebo.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
