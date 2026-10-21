#!/usr/bin/env python3
"""V20 audit from archived paired predictions; requires pandas numpy sklearn.

python scripts/research/v20_extended_prediction_audit.py --csv second_look_predictions.csv
No retraining or raw-price target validation. Weight sweeps are post-hoc.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
L=[0,1,2]
C=["negative","neutral","positive"]
def macro(y,p):return float(f1_score(y,p,labels=L,average="macro",zero_division=0))
def cmf1(m):
    den=m.sum(0)+m.sum(1)
    return np.divide(2*np.diag(m),den,out=np.zeros(3),where=den!=0).mean()
def boot(df,y,a,b,by="date",n=10000,seed=20261021):
    groups=list(df.groupby(by).indices.values())
    aa=np.array([confusion_matrix(y[i],a[i],labels=L) for i in groups])
    bb=np.array([confusion_matrix(y[i],b[i],labels=L) for i in groups])
    rng=np.random.default_rng(seed)
    results=[]
    for _ in range(n):
        ix=rng.integers(0,len(groups),size=len(groups))
        results.append(float(cmf1(bb[ix].sum(0))-cmf1(aa[ix].sum(0))))
    return list(map(float,np.quantile(results,[.025,.975])))
def run(csv,out):
    df=pd.read_csv(csv);df["date"]=pd.to_datetime(df.date).dt.normalize()
    df["week"]=df.date.dt.to_period("W-FRI").astype(str)
    assert df.session_id.is_unique and len(df)>0
    y=df.actual.to_numpy(int)
    assert set(y).issubset(L)
    pred={m:df[m].to_numpy(int) for m in ("v9","v13")}
    probs={}
    for m in pred:
        a=df[[f"{m}_prob_{c}" for c in C]].to_numpy(float)
        assert np.isfinite(a).all() and (a>=0).all()
        assert np.max(np.abs(a.sum(1)-1))<.002 and np.array_equal(a.argmax(1),pred[m])
        probs[m]=a/a.sum(1,keepdims=True)
    summary={"evidence":"SAVED-PREDICTION recomputation; not independent model or target audit",
      "n":len(df),"date_count":df.date.nunique(),"classes":dict(zip(C,map(int,np.bincount(y,minlength=3))))}
    for m in pred:
        summary[m]={"macro_f1":macro(y,pred[m]),"accuracy":float(accuracy_score(y,pred[m]))}
    summary["delta_macro_f1"]=summary["v13"]["macro_f1"]-summary["v9"]["macro_f1"]
    summary["ci95_date"]=boot(df,y,pred["v9"],pred["v13"])
    summary["ci95_week"]=boot(df,y,pred["v9"],pred["v13"],"week")
    summary["always_non_neutral_f1"]=float(f1_score(y!=1,np.ones(len(y),dtype=int)))
    sweep=[]
    for m in pred:
        for w in np.round(np.arange(.5,1.501,.025),3):
            prob=probs[m].copy();prob[:,1]*=w
            pp=prob.argmax(1)
            sweep.append({"model":m,"neutral_weight":float(w),"macro_f1":macro(y,pp),
             "accuracy":float(accuracy_score(y,pp)),"neutral_pred_share":float((pp==1).mean())})
    pd.DataFrame(sweep).to_csv(out/"v20_symmetric_POSTHOC_sweep.csv",index=False)
    annual=[]
    for year,part in df.groupby(df.date.dt.year):
        for m in pred:
            ix=part.index.to_numpy()
            from sklearn.metrics import precision_recall_fscore_support
            pr,rec,f1,_=precision_recall_fscore_support(y[ix],pred[m][ix],labels=L,zero_division=0)
            annual.append(dict(year=int(year),model=m,macro_f1=macro(y[ix],pred[m][ix]),
              **{f"f1_{C[k]}":float(f1[k]) for k in L},
              **{f"recall_{C[k]}":float(rec[k]) for k in L}))
    pd.DataFrame(annual).to_csv(out/"v20_class_by_year.csv",index=False)
    # Always-negative constant sign baseline on ALL truly non-neutral cases.
    sign={}
    mask=y!=1
    selected=df.loc[mask,["date"]].copy()
    for m in pred:
        sign_pred=np.where(probs[m][:,0]>=probs[m][:,2],0,2)
        diff=(sign_pred[mask]==y[mask]).astype(float)-(y[mask]==0).astype(float)
        selected["diff"]=diff
        byday=selected.groupby("date")["diff"].agg(["sum","count"])
        total=byday.to_numpy();rng=np.random.default_rng(20261021)
        estimates=[]
        for _ in range(10000):
            ix=rng.integers(len(total),size=len(total))
            estimates.append(total[ix,0].sum()/total[ix,1].sum())
        sign[m]={"n":int(mask.sum()),"accuracy":float(np.mean(sign_pred[mask]==y[mask])),
        "always_negative_accuracy":float(np.mean(y[mask]==0)),
        "delta":float(diff.mean()),"ci95_date":list(map(float,np.quantile(estimates,[.025,.975])))}
    summary["sign_on_all_true_tails"]=sign
    summary["caveat"]="Weight selection on 2025-2026 is post-hoc; no trained volatility-only, SPY or ablation result."
    (out/"v20_extended_audit.json").write_text(json.dumps(summary,indent=2,default=int))
    print(json.dumps(summary,indent=2,default=int))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",required=True,type=Path)
    parser.add_argument("--out",default=Path("results"),type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    run(args.csv,args.out)
