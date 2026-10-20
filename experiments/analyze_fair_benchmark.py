"""Audit and publish the strict capacity/budget-matched experiment."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_fair_benchmark as fit
import analyze_multievent_results as stats

def main():
    torch.set_num_threads(4)
    protocol=json.loads((fit.OUT/"manifest.json").read_text())
    for path in (Path(fit.__file__),Path(fit.base.__file__)):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["source_sha256"][path.name]
    assert len(list((fit.OUT/"runs").glob("*.json")))==144
    rows=[]; manifest=[]; cache={}; audit=[]
    index=stats.ExperimentIndex(tuple(fit.base.EVENTS),fit.MODELS,(11,22,33),
                                {e:(1,2,3) for e in fit.base.EVENTS})
    for event in index.events:
        for fold,frame,split,core,h in fit.base.load_event_folds(event):
            train,val,n=map(int,split)
            actual=frame[["rumor_activity","debunk_activity"]].to_numpy(float)
            for model in index.models:
                for seed in index.seeds:
                    stem=fit.OUT/"runs"/f"{event}_fold{fold}_{model}_seed{seed}"
                    payload=json.loads(stem.with_suffix(".json").read_text())
                    run=payload["run"]
                    assert run["epochs_completed"]==260 and run["parameter_count"]==18619
                    assert payload["gradient_connected"]
                    expected_config=dict(protocol["config"])
                    if model=="Data-Transformer":
                        expected_config.update(physics_weight=0.,history_weight=0.)
                    assert payload["config"]==expected_config
                    pred=pd.read_csv(stem.with_suffix(".csv"))
                    np.testing.assert_allclose(pred[["actual_rumor","actual_debunk"]],actual)
                    assert pred["split"].tolist()==["train"]*train+["validation"]*(val-train)+["test"]*(n-val)
                    fitted=fit.factory(model,16)
                    fitted.load_state_dict(torch.load(stem.with_suffix(".pt"),weights_only=True))
                    with torch.no_grad():
                        normalized=fitted.observation((torch.arange(1,n+1,dtype=torch.float32)/n).view(-1,1)).numpy()
                    reproduced=normalized*np.maximum(actual[:train].sum(0),1)
                    np.testing.assert_allclose(reproduced,pred[["predicted_rumor","predicted_debunk"]],rtol=2e-6,atol=1e-6)
                    raw32=actual.astype(np.float32)
                    scale32=np.maximum(raw32[:train].sum(0),1).astype(np.float32)
                    score=fit.base.validation_score(
                        fitted,(torch.arange(1,n+1,dtype=torch.float32)/n).view(-1,1),
                        torch.tensor(raw32/scale32),torch.tensor(np.cumsum(raw32,axis=0)/scale32),
                        int(train),int(val))
                    np.testing.assert_allclose(score,run["validation_score"],rtol=2e-6,atol=1e-9)
                    for k,channel in enumerate(stats.CHANNELS):
                        recorded=payload["rows"][k]
                        recomputed=fit.base.regression_metrics(actual[val:,k],reproduced[val:,k],actual[:train,k])
                        for metric in ("nrmse","nmae"):
                            np.testing.assert_allclose(recorded[metric],recomputed[metric],rtol=2e-6)
                        rows.append(recorded)
                    cache[(event,fold,model,seed)]=pred
                    manifest.append(run)
                    audit.append({"run":stem.name,"sha256":{ext:hashlib.sha256(stem.with_suffix(ext).read_bytes()).hexdigest() for ext in (".pt",".csv",".json")}})
    df=pd.DataFrame(rows); runs=pd.DataFrame(manifest)
    df.to_csv(fit.OUT/"metrics.csv",index=False)
    runs.to_csv(fit.OUT/"run_manifest.csv",index=False)
    selected=runs.sort_values(["validation_score","seed"]).groupby(["event","fold","model"],sort=False).head(1)
    selected.to_csv(fit.OUT/"selected_initializations.csv",index=False)
    chosen={(r.event,r.fold,r.model):int(r.seed) for r in selected.itertuples()}
    chosen_rows=df.merge(selected[["event","fold","model","seed"]],on=["event","fold","model","seed"])
    # Balanced design: equal channel, fold and event weighting.
    seed_macro=df.groupby(["model","seed"])[["nrmse","nmae"]].mean()
    summary=seed_macro.groupby("model").agg(["mean","std"])
    summary.columns=[f"{a}_{b}" for a,b in summary.columns]
    summary=summary.join(chosen_rows.groupby("model")[["nrmse","nmae"]].mean().add_prefix("selected_"))
    summary=summary.join(runs.groupby("model")["runtime_seconds"].agg(["mean","std"]).add_prefix("runtime_"))
    summary=summary.loc[list(fit.MODELS)]
    summary.to_csv(fit.OUT/"summary.csv")
    df.groupby(["event","model","seed"])[["nrmse","nmae"]].mean().to_csv(fit.OUT/"event_seed_metrics.csv")
    chosen_rows.groupby(["event","model"])[["nrmse","nmae"]].mean().to_csv(fit.OUT/"event_selected_metrics.csv")
    selected_ci=stats.paired_validation_selected_bootstrap(cache,chosen,index,5000,20260918)
    selected_ci.to_csv(fit.OUT/"selected_bootstrap.csv",index=False)
    all_ci=stats.paired_hierarchical_block_bootstrap(cache,index,5000,20260918)
    all_ci.to_csv(fit.OUT/"all_seed_bootstrap.csv",index=False)
    data_hashes={e:hashlib.sha256((fit.base.DATA_DIR/f"{e}.csv").read_bytes()).hexdigest() for e in index.events}
    (fit.OUT/"audit.json").write_text(json.dumps({"fits":len(runs),"checkpoint_replays":len(runs),
        "validation_scores_recomputed":len(runs),"data_sha256":data_hashes,"runs":audit},indent=2))
    lines=[r"\begin{table*}[t]",r"\centering",r"\caption{Strict capacity- and update-matched benchmark: 18,619 trainable parameters and 260 updates per fit. All-seed errors are macro mean $\pm$ sample SD across three seeds. Selected errors use validation-only seed selection. Elapsed training time is mean $\pm$ SD over 36 fits per method on the same CPU (four threads).}",r"\label{tab:fair_benchmark}",r"\small",r"\setlength{\tabcolsep}{5pt}",r"\begin{tabular}{lccccc}",r"\toprule",r"Method & All-seed NRMSE & All-seed NMAE & Selected NRMSE & Selected NMAE & Time (s)/fit\\",r"\midrule"]
    for name,r in summary.iterrows():
        lines.append(f"{name} & ${r.nrmse_mean:.4f}\\pm{r.nrmse_std:.4f}$ & ${r.nmae_mean:.4f}\\pm{r.nmae_std:.4f}$ & {r.selected_nrmse:.4f} & {r.selected_nmae:.4f} & ${r.runtime_mean:.1f}\\pm{r.runtime_std:.1f}$\\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",r"\begin{table*}[t]",r"\centering",r"\caption{Paired error differences in the strict matched benchmark. $\Delta={\rm NRMSE}_{\rm baseline}-{\rm NRMSE}_{\rm TransPINN}$; positive values favor TransPINN. Intervals use 5000 paired event/fold circular-block bootstrap replicates, with paired seed resampling for the all-seed comparison.}",r"\label{tab:fair_bootstrap}",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",r"Baseline & All-seed $\Delta$ & 95\% interval & Selected $\Delta$ & 95\% interval\\",r"\midrule"]
    for r in selected_ci.itertuples():
        a=all_ci.loc[all_ci.baseline.eq(r.baseline)].iloc[0]
        lines.append(f"{r.baseline} & ${a.baseline_minus_transpinn_nrmse:+.4f}$ & $[{a.ci95_low:.4f},\\ {a.ci95_high:.4f}]$ & ${r.baseline_minus_transpinn_nrmse:+.4f}$ & $[{r.ci95_low:.4f},\\ {r.ci95_high:.4f}]$\\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (fit.ROOT/"experiments/fair_benchmark_tables.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(summary.to_string()); print(selected_ci[["baseline","baseline_minus_transpinn_nrmse","ci95_low","ci95_high"]].to_string(index=False))

if __name__=="__main__":
    main()
