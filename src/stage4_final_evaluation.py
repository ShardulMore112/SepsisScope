"""
X-FedSepsis — Stage 4.0 Final Evaluation
=========================================
Frozen candidate:
    TR1 F0+F1+F2+F3 equal-weight probability ensemble.

This script:
    1. Loads frozen validation/test predictions.
    2. Reconstructs patient boundaries from the frozen split + patient_splits.csv.
    3. Selects the ensemble threshold on VALIDATION only (max F1 over 1001 thresholds).
    4. Applies that threshold unchanged to TEST.
    5. Produces timestep, patient-level, early-warning, calibration,
       bootstrap, CSV/JSON, and publication figures.

No model is trained. No test-set tuning is performed.
"""
from pathlib import Path
import io, json, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "final_evaluation"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def load_npz(path):
    return np.load(path, allow_pickle=True)

def extract_probs_targets(path):
    a = load_npz(path)
    p = a["probabilities"] if "probabilities" in a else a["probs"]
    y = a["labels"] if "labels" in a else a["targets"]
    return p.astype(float), y.astype(int)

def best_threshold(y, p):
    """Exact project rule: max F1 over 1001 thresholds [0,1]."""
    th = np.linspace(0.0, 1.0, 1001)
    order = np.argsort(p)
    sp = p[order]
    sy = y[order].astype(np.int64)
    cpos = np.cumsum(sy)
    total = int(sy.sum())
    counts = np.searchsorted(sp, th, side="left")
    pred_pos = len(y) - counts
    tp = np.where(counts == 0, total,
                  total - cpos[np.minimum(counts - 1, len(y)-1)])
    fp = pred_pos - tp
    fn = total - tp
    den = 2*tp + fp + fn
    f1 = np.divide(2*tp, den, out=np.zeros_like(den,dtype=float), where=den>0)
    i = int(np.argmax(f1))
    return float(th[i]), float(f1[i])

def binary_metrics(y,p,t):
    pred=(p>=t).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,pred,average="binary",zero_division=0
    )
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "AUPRC": float(average_precision_score(y,p)),
        "AUROC": float(roc_auc_score(y,p)),
        "threshold": float(t),
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": float(tn/(tn+fp)) if tn+fp else 0.0,
        "F1": float(f1),
        "accuracy": float((tp+tn)/(tp+tn+fp+fn)),
        "TN": int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)
    }

# ------------------------------------------------------------
# Input paths
# ------------------------------------------------------------
val_paths = [
    RESULTS/"model_result"/"feature_ablation"/f"TR1-F{i}"/"validation_predictions.npz"
    for i in range(4)
]
test_paths = [
    RESULTS/"baseline_result"/"baseline_transformer_f0_test_predictions.npz",
    RESULTS/"model_result"/"feature_ablation"/"TR1-F1"/"test_predictions.npz",
    RESULTS/"model_result"/"feature_ablation"/"TR1-F2"/"test_predictions.npz",
    RESULTS/"model_result"/"feature_ablation"/"TR1-F3"/"test_predictions.npz",
]

val_probs=[]; test_probs=[]
for path in val_paths:
    p,y=extract_probs_targets(path)
    val_probs.append(p)
    if "y_val" not in locals(): y_val=y
    elif not np.array_equal(y_val,y): raise ValueError("Validation targets differ across representations.")
for path in test_paths:
    p,y=extract_probs_targets(path)
    test_probs.append(p)
    if "y_test" not in locals(): y_test=y
    elif not np.array_equal(y_test,y): raise ValueError("Test targets differ across representations.")

p_val=np.mean(val_probs,axis=0)
p_test=np.mean(test_probs,axis=0)

threshold, _ = best_threshold(y_val,p_val)
val_m=binary_metrics(y_val,p_val,threshold)
test_m=binary_metrics(y_test,p_test,threshold)

# ------------------------------------------------------------
# Reconstruct test patient boundaries
# ------------------------------------------------------------
split=np.load(ROOT/"data"/"splits"/"split_indices.npz",allow_pickle=True)
test_ids=split["test_patient_ids"].astype(str)

patient_splits=pd.read_csv(RESULTS/"dataset_audit_result"/"patient_splits.csv")
test_meta=patient_splits[patient_splits["split"]=="test"].set_index("patient_id")
test_meta=test_meta.loc[test_ids]
lengths=test_meta["hours"].to_numpy(int)

if lengths.sum()!=len(y_test):
    raise ValueError(f"Patient-hour total {lengths.sum()} != prediction length {len(y_test)}.")

patient_labels=test_meta["sepsis"].to_numpy(int)
offsets=np.r_[0,np.cumsum(lengths)]
pred=(p_test>=threshold).astype(np.int8)

rows=[]
for i,(pid,l,cls) in enumerate(zip(test_ids,lengths,patient_labels)):
    s,e=offsets[i],offsets[i+1]
    yy=y_test[s:e]; pp=p_test[s:e]; aa=pred[s:e]
    alerts=np.flatnonzero(aa); positives=np.flatnonzero(yy)
    first_alert=int(alerts[0]) if len(alerts) else np.nan
    first_pos=int(positives[0]) if len(positives) else np.nan

    if cls==1:
        if not len(alerts): category="MISSED"; lead=np.nan; rel=np.nan
        elif first_alert < first_pos: category="EARLY"; lead=float(first_pos-first_alert); rel=float(first_alert-first_pos)
        elif first_alert == first_pos: category="AT_ONSET"; lead=0.0; rel=0.0
        else: category="LATE"; lead=np.nan; rel=float(first_alert-first_pos)
    else:
        category="FALSE_ALARM" if len(alerts) else "CLEAN"
        lead=np.nan; rel=np.nan

    rows.append({
        "patient_id":pid,"sepsis":int(cls),"hours":int(l),
        "first_positive_hour":first_pos,"first_alert_hour":first_alert,
        "n_alerts":len(alerts),"category":category,
        "lead_time_hours":lead,"first_alert_offset_hours":rel,
        "max_risk":float(np.max(pp))
    })
patient_df=pd.DataFrame(rows)

tp=int(((patient_df.sepsis==1)&(patient_df.category!="MISSED")).sum())
fn=int(((patient_df.sepsis==1)&(patient_df.category=="MISSED")).sum())
fp=int(((patient_df.sepsis==0)&(patient_df.category=="FALSE_ALARM")).sum())
tn=int(((patient_df.sepsis==0)&(patient_df.category=="CLEAN")).sum())
psens=tp/(tp+fn); pspec=tn/(tn+fp); pprec=tp/(tp+fp); pf1=2*pprec*psens/(pprec+psens)

patient_scores=patient_df.max_risk.to_numpy()
patient_y=patient_df.sepsis.to_numpy()
patient_auc=roc_auc_score(patient_y,patient_scores)
patient_ap=average_precision_score(patient_y,patient_scores)

# ------------------------------------------------------------
# Save primary outputs
# ------------------------------------------------------------
patient_df.to_csv(OUT/"patient_level_results.csv",index=False)
pd.DataFrame(confusion_matrix(y_test,pred,labels=[0,1]),
             index=["Actual 0","Actual 1"],columns=["Pred 0","Pred 1"]).to_csv(OUT/"timestep_confusion_matrix.csv")
pd.DataFrame([[tn,fp],[fn,tp]],
             index=["Actual non-sepsis","Actual sepsis"],
             columns=["Pred negative","Pred positive"]).to_csv(OUT/"patient_confusion_matrix.csv")

np.savez_compressed(OUT/"final_validation_ensemble.npz",
                    probabilities=p_val.astype(np.float32),
                    labels=y_val.astype(np.int8),
                    threshold=np.array(threshold,dtype=np.float32))
np.savez_compressed(OUT/"final_test_ensemble.npz",
                    probabilities=p_test.astype(np.float32),
                    labels=y_test.astype(np.int8),
                    threshold=np.array(threshold,dtype=np.float32))

categories=["EARLY","AT_ONSET","LATE","MISSED","FALSE_ALARM","CLEAN"]
patient_df.category.value_counts().reindex(categories,fill_value=0).rename("count").to_csv(OUT/"alert_category_counts.csv")
early=patient_df.loc[patient_df.category=="EARLY","lead_time_hours"].dropna()
pd.Series({
    "n_early":len(early),
    "mean_lead_hours":early.mean(),
    "median_lead_hours":early.median(),
    "p25_lead_hours":early.quantile(.25),
    "p75_lead_hours":early.quantile(.75),
    "min_lead_hours":early.min(),
    "max_lead_hours":early.max()
}).to_csv(OUT/"early_warning_statistics.csv")

# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------
brier=brier_score_loss(y_test,p_test)
bins=np.linspace(0,1,11); cal=[]
for lo,hi in zip(bins[:-1],bins[1:]):
    mask=(p_test>=lo)&((p_test<hi) if hi<1 else (p_test<=hi))
    if mask.any():
        cal.append({"bin_lower":lo,"bin_upper":hi,"count":int(mask.sum()),
                    "mean_pred":float(p_test[mask].mean()),
                    "fraction_positive":float(y_test[mask].mean())})
cal_df=pd.DataFrame(cal)
cal_df.to_csv(OUT/"calibration_bins.csv",index=False)
ece=float(sum((r["count"]/len(y_test))*abs(r["mean_pred"]-r["fraction_positive"])
              for _,r in cal_df.iterrows()))

# ------------------------------------------------------------
# Component test comparison
# ------------------------------------------------------------
component=[]
for i,(vp,tpred) in enumerate(zip(val_probs,test_probs)):
    # use the already stored validation threshold when present
    a=np.load(val_paths[i],allow_pickle=True)
    t=float(np.asarray(a["threshold"]).reshape(-1)[0]) if "threshold" in a else best_threshold(y_val,vp)[0]
    m=binary_metrics(y_test,tpred,t)
    component.append({"model":f"TR1-F{i}",**m})
component.append({"model":"TR1-ENSEMBLE (F0+F1+F2+F3)",**test_m})
pd.DataFrame(component).to_csv(OUT/"component_and_ensemble_test_metrics.csv",index=False)

# ------------------------------------------------------------
# Validation feature combinations (TR1)
# ------------------------------------------------------------
comb=[]
for r in range(1,5):
    for ids in itertools.combinations(range(4),r):
        pp=np.mean([val_probs[i] for i in ids],axis=0)
        t,f1=best_threshold(y_val,pp)
        pr,rec,_,_=precision_recall_fscore_support(y_val,(pp>=t).astype(int),average="binary",zero_division=0)
        comb.append({"features":"+".join(f"F{i}" for i in ids),"n_features":r,
                     "AUPRC":average_precision_score(y_val,pp),
                     "AUROC":roc_auc_score(y_val,pp),"F1":f1,"threshold":t,
                     "precision":pr,"sensitivity":rec})
comb_df=pd.DataFrame(comb).sort_values("AUPRC",ascending=False)
comb_df.to_csv(OUT/"validation_tr1_feature_combination_results.csv",index=False)

# ------------------------------------------------------------
# Patient-cluster bootstrap CIs
# ------------------------------------------------------------
rng=np.random.default_rng(2026); boot=[]
cats=patient_df.category.to_numpy()
for _ in range(200):
    idx=rng.integers(0,len(patient_df),len(patient_df))
    yy=patient_y[idx]; ss=patient_scores[idx]; cc=cats[idx]
    if np.unique(yy).size==2:
        ba=roc_auc_score(yy,ss); bp=average_precision_score(yy,ss)
    else:
        ba=np.nan; bp=np.nan
    btp=((yy==1)&(cc!="MISSED")).sum(); bfn=((yy==1)&(cc=="MISSED")).sum()
    bfp=((yy==0)&(cc=="FALSE_ALARM")).sum(); btn=((yy==0)&(cc=="CLEAN")).sum()
    boot.append([ba,bp,btp/(btp+bfn),btn/(btn+bfp)])
boot=np.asarray(boot)
ests=[patient_auc,patient_ap,psens,pspec]
names=["patient_AUROC","patient_AUPRC","patient_sensitivity","patient_specificity"]
ci=pd.DataFrame([{"metric":n,"estimate":float(e),
                  "ci95_lower":float(np.nanpercentile(boot[:,j],2.5)),
                  "ci95_upper":float(np.nanpercentile(boot[:,j],97.5))}
                 for j,(n,e) in enumerate(zip(names,ests))])
ci.to_csv(OUT/"patient_bootstrap_95ci.csv",index=False)

# ------------------------------------------------------------
# Final JSON
# ------------------------------------------------------------
final={
 "protocol":{
   "primary_candidate":"TR1 F0+F1+F2+F3 equal-weight probability ensemble",
   "selection_metric":"validation AUPRC",
   "threshold_rule":"maximum validation F1 over 1001 thresholds",
   "frozen_threshold":threshold,
   "test_used_for_selection":False,
   "prior_test_exposure_caveat":True,
   "feature_ablation_training_budget_caveat":"F1/F2/F3 ablations used 5 epochs while the earlier F0 family protocol used 15 epochs."
 },
 "validation":{"n_timesteps":len(y_val),"positive_timesteps":int(y_val.sum()),
               "prevalence":float(y_val.mean()),**val_m},
 "test_timestep":test_m,
 "test_patient":{"n_patients":len(patient_df),"positive_patients":int(patient_y.sum()),
                 "patient_prevalence":float(patient_y.mean()),
                 "TP":tp,"TN":tn,"FP":fp,"FN":fn,
                 "sensitivity":psens,"specificity":pspec,"precision":pprec,"F1":pf1,
                 "AUROC_max_probability":patient_auc,"AUPRC_max_probability":patient_ap},
 "alert_categories":patient_df.category.value_counts().reindex(categories,fill_value=0).to_dict(),
 "early_warning":{"n_early":len(early),"mean_hours":float(early.mean()),
                  "median_hours":float(early.median()),"p25_hours":float(early.quantile(.25)),
                  "p75_hours":float(early.quantile(.75)),"min_hours":float(early.min()),
                  "max_hours":float(early.max())},
 "calibration":{"Brier":float(brier),"ECE_10_equal_width":ece}
}
with open(OUT/"final_metrics.json","w") as f:
    json.dump(final,f,indent=2)

# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG/name,dpi=300,bbox_inches="tight")
    plt.close()

fpr,tpr,_=roc_curve(y_test,p_test)
plt.figure(figsize=(7,6)); plt.plot(fpr,tpr,label=f"AUROC={test_m['AUROC']:.3f}"); plt.plot([0,1],[0,1],"--",label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Stage 4.0 — Test ROC Curve"); plt.legend(); savefig("final_roc_curve.png")

prec,rec,_=precision_recall_curve(y_test,p_test)
plt.figure(figsize=(7,6)); plt.plot(rec,prec,label=f"AUPRC={test_m['AUPRC']:.3f}"); plt.axhline(y_test.mean(),ls="--",label=f"Prevalence={y_test.mean():.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Stage 4.0 — Test Precision–Recall Curve"); plt.legend(); savefig("final_precision_recall_curve.png")

cm=confusion_matrix(y_test,pred,labels=[0,1])
plt.figure(figsize=(6,5)); plt.imshow(cm,aspect="auto"); plt.colorbar()
for i in range(2):
    for j in range(2): plt.text(j,i,f"{cm[i,j]:,}",ha="center",va="center")
plt.xticks([0,1],["Predicted 0","Predicted 1"]); plt.yticks([0,1],["Actual 0","Actual 1"]); plt.title("Test Timestep Confusion Matrix"); savefig("final_timestep_confusion_matrix.png")

pcm=np.array([[tn,fp],[fn,tp]])
plt.figure(figsize=(6,5)); plt.imshow(pcm,aspect="auto"); plt.colorbar()
for i in range(2):
    for j in range(2): plt.text(j,i,f"{pcm[i,j]:,}",ha="center",va="center")
plt.xticks([0,1],["Predicted negative","Predicted positive"]); plt.yticks([0,1],["Actual non-sepsis","Actual sepsis"]); plt.title("Test Patient-Level Confusion Matrix"); savefig("final_patient_confusion_matrix.png")

ths=np.linspace(0,1,501); curves={k:[] for k in ["F1","Sensitivity","Specificity","Precision"]}
for t in ths:
    pp=(p_test>=t).astype(int); pr,rc,f1,_=precision_recall_fscore_support(y_test,pp,average="binary",zero_division=0)
    tn0,fp0,fn0,tp0=confusion_matrix(y_test,pp,labels=[0,1]).ravel()
    curves["F1"].append(f1); curves["Sensitivity"].append(rc); curves["Specificity"].append(tn0/(tn0+fp0)); curves["Precision"].append(pr)
plt.figure(figsize=(8,6))
for k,v in curves.items(): plt.plot(ths,v,label=k)
plt.axvline(threshold,ls="--",label=f"Frozen threshold={threshold:.3f}"); plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.ylim(0,1.02); plt.title("Test Threshold Sensitivity Analysis"); plt.legend(); savefig("final_threshold_analysis.png")

counts=patient_df.category.value_counts().reindex(categories,fill_value=0)
plt.figure(figsize=(9,5)); plt.bar(categories,counts.values); plt.ylabel("Patients"); plt.title("Patient-Level Alert Outcome Categories"); plt.xticks(rotation=25,ha="right"); savefig("final_alert_categories.png")

plt.figure(figsize=(8,5)); plt.hist(early,bins=20); plt.xlabel("Lead time before first positive SepsisLabel (hours)"); plt.ylabel("Early-alert patients"); plt.title("Early-Warning Lead-Time Distribution"); savefig("final_lead_time_distribution.png")

offsets=patient_df.loc[patient_df.sepsis==1,"first_alert_offset_hours"].dropna()
plt.figure(figsize=(8,5)); plt.hist(offsets,bins=20); plt.axvline(0,ls="--"); plt.xlabel("First-alert offset relative to first positive SepsisLabel (hours)"); plt.ylabel("Sepsis patients"); plt.title("First Alert Timing Relative to First Positive Label"); savefig("final_alert_timing_relative_to_label.png")

comp=pd.DataFrame(component)
x=np.arange(len(comp)); w=.38
fig,ax=plt.subplots(figsize=(10,6)); ax.bar(x-w/2,comp.AUPRC,width=w,label="AUPRC"); ax.bar(x+w/2,comp.AUROC,width=w,label="AUROC")
ax.set_xticks(x); ax.set_xticklabels(comp.model,rotation=25,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Test Performance: Transformer Representations and Frozen Ensemble"); ax.legend(); savefig("final_model_comparison.png")

c=comb_df.sort_values("AUPRC")
plt.figure(figsize=(9,7)); plt.barh(c.features,c.AUPRC); plt.xlabel("Validation AUPRC"); plt.ylabel("TR1 feature combination"); plt.title("Validation TR1 Feature-Combination Analysis"); savefig("validation_tr1_feature_combinations.png")

plt.figure(figsize=(6,6)); plt.plot(cal_df.mean_pred,cal_df.fraction_positive,marker="o",label=f"Brier={brier:.4f}"); plt.plot([0,1],[0,1],"--",label="Perfect calibration")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed positive fraction"); plt.title("Test Reliability Diagram"); plt.legend(); savefig("final_calibration_reliability.png")

pfpr,ptpr,_=roc_curve(patient_y,patient_scores)
plt.figure(figsize=(8,6)); plt.plot(pfpr,ptpr,label=f"AUROC={patient_auc:.3f}"); plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Patient-Level ROC (Maximum Trajectory Risk)"); plt.legend(); savefig("final_patient_roc_curve.png")

ppr,prr,_=precision_recall_curve(patient_y,patient_scores)
plt.figure(figsize=(8,6)); plt.plot(prr,ppr,label=f"AUPRC={patient_ap:.3f}"); plt.axhline(patient_y.mean(),ls="--",label=f"Prevalence={patient_y.mean():.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Patient-Level Precision–Recall Curve"); plt.legend(); savefig("final_patient_precision_recall_curve.png")

print("Stage 4.0 complete.")
print("Frozen threshold:", threshold)
print("Test timestep:", test_m)
print("Patient:", {"TP":tp,"TN":tn,"FP":fp,"FN":fn,"sensitivity":psens,"specificity":pspec,"precision":pprec,"F1":pf1})
print("Figures:", len(list(FIG.glob("final_*.png")))+len(list(FIG.glob("validation_*.png"))))
