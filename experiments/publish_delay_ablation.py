"""Audit predictions and publish the delay comparison without test selection."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/python-deps'))
import numpy as np
import pandas as pd
import torch
import run_multievent_baselines as base
import analyze_multievent_results as analysis
OUT=ROOT/'output/delay_ablation'
NAMES={'full':'TransPINN','tau_zero':'TransPINN tau=0'}

def main():
    torch.set_num_threads(4)
    rows=[]; infos=[]; cache={}; hashes={}
    for event in base.EVENTS:
        for fold,original,split,_,_ in base.load_event_folds(event):
            actual=original[['rumor_activity','debunk_activity']].to_numpy(float)
            for variant,name in NAMES.items():
                for seed in (11,22,33):
                    stem=f'{event}_fold{fold}_{variant}_seed{seed}'
                    record=json.loads((OUT/(stem+'.json')).read_text())
                    assert record['config']==base.asdict(base.Config())
                    f=pd.read_csv(OUT/(stem+'.csv'))
                    np.testing.assert_allclose(f[['actual_rumor','actual_debunk']],actual)
                    np.testing.assert_array_equal(f['split'],np.where(np.arange(len(f))<split[0],'train',np.where(np.arange(len(f))<split[1],'validation','test')))
                    model=base.RevisedRumorPINN('TransPINN',16,fixed_tau=0. if variant=='tau_zero' else None)
                    model.load_state_dict(torch.load(OUT/(stem+'.pt'),weights_only=True))
                    times=(torch.arange(1,len(f)+1,dtype=torch.float32)/len(f)).view(-1,1)
                    with torch.no_grad(): pred=model.observation(times).numpy()*np.maximum(actual[:split[0]].sum(0),1.)
                    np.testing.assert_allclose(pred,f[['predicted_rumor','predicted_debunk']],rtol=2e-5,atol=1e-6)
                    for r in record['rows']:
                        c=('rumor','debunk').index(r['channel'])
                        m=base.regression_metrics(actual[split[1]:,c],pred[split[1]:,c],actual[:split[0],c])
                        np.testing.assert_allclose([r['nrmse'],r['nmae']],[m['nrmse'],m['nmae']],rtol=2e-5,atol=1e-7)
                        rows.append(dict(r,model=name))
                    infos.append(dict(record['run'],model=name))
                    cache[event,fold,name,seed]=f
                    for ext in ('.json','.csv','.pt'):
                        p=OUT/(stem+ext); hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    runs=pd.DataFrame(rows); info=pd.DataFrame(infos)
    selected=info.sort_values(['validation_score','seed']).drop_duplicates(['event','fold','model'])
    chosen=runs.merge(selected[['event','fold','model','seed']],on=['event','fold','model','seed'])
    selection={(r.event,r.fold,r.model):r.seed for r in selected.itertuples()}
    index=analysis.ExperimentIndex(events=base.EVENTS,models=tuple(NAMES.values()),seeds=(11,22,33),folds_by_event={e:(1,2,3) for e in base.EVENTS})
    effects=analysis.paired_validation_selected_bootstrap(cache,selection,index,5000,20260928)
    summaries=[]
    for name in NAMES.values():
        allseed=runs[runs.model==name].groupby('seed').nrmse.mean()
        sub=chosen[chosen.model==name]; fits=info[info.model==name]
        summaries.append(dict(model=name,all_mean=allseed.mean(),all_sd=allseed.std(),selected_nrmse=sub.nrmse.mean(),selected_nmae=sub.nmae.mean(),runtime_mean=fits.runtime_seconds.mean(),runtime_sd=fits.runtime_seconds.std(),parameter_count=int(fits.parameter_count.iloc[0])))
    summary=pd.DataFrame(summaries)
    for name,frame in [('summary',summary),('runs',runs),('fits',info),('selection',selected),('paired_bootstrap',effects),('event_summary',chosen.groupby(['model','event'])[['nrmse','nmae']].mean().reset_index())]: frame.to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'audit.json').write_text(json.dumps({'fits_verified':len(info),'all_predictions_reloaded':True,'input_sha256':hashes},indent=2))
    full,zero=summary.iloc[0],summary.iloc[1]; ef=effects.iloc[0]
    diff=ef.baseline_minus_transpinn_nrmse
    conclusion=('The interval includes zero, so this experiment does not establish a clear forecasting benefit from a positive delay.' if ef.ci95_low<=0<=ef.ci95_high else ('The interval favors the full model under this protocol.' if ef.ci95_low>0 else 'The interval favors the model with zero delay under this protocol.'))
    section=r'''\subsection{Ablation of the delay}
\label{sec:delay_ablation}
\begingroup\color{red}
We compare the full TransPINN with a version that fixes $\tau=0$.
In this version, $X(t-\tau)=X(t)$ and $e^{-a_E\tau}=1$.
The cohort transfer becomes $Q_0(t)=p(t)\mathcal I(t)$.
Thus $\dot E=(1-p(t))\mathcal I(t)-a_EE$, while $H$ receives $p(t)\mathcal I(t)$.
The transfer pathway remains active. This comparison removes the waiting time.
The attention window remains $[\max(0,t-0.18),t]$.

Both variants use the same four events, three chronological folds, and seeds 11, 22, and 33.
We retrain both variants, giving 72 fits on the same CPU with four threads.
The network, losses, initial history, optimizer, and maximum budget of 260 epochs remain the same.
Each fit estimates the remaining parameters again.
The full model estimates $\tau$ within $[0.01,0.22]$.
The ablation fixes it at zero, reducing the trainable parameter count from 18619 to 18618.
Only training observations enter the data loss.
Validation loss selects each checkpoint and the seed for deployment.
Test observations do not enter selection.

\begin{table}[t]
\centering\scriptsize
\caption{Delay ablation. NRMSE across all seeds is the mean $\pm$ sample SD of the three seed averages. Selected errors use validation alone. Runtime is mean CPU seconds per fit.}
\label{tab:delay_ablation}
\begin{tabular}{@{}lcc@{}}
\toprule
Metric & Full TransPINN & $\tau=0$\\
\midrule
'''
    section+=f'NRMSE across seeds & ${full.all_mean:.6f}\\pm{full.all_sd:.6f}$ & ${zero.all_mean:.6f}\\pm{zero.all_sd:.6f}$\\\\\n'
    section+=f'Selected NRMSE & {full.selected_nrmse:.6f} & {zero.selected_nrmse:.6f}\\\\\nSelected NMAE & {full.selected_nmae:.6f} & {zero.selected_nmae:.6f}\\\\\nSeconds per fit & {full.runtime_mean:.2f} & {zero.runtime_mean:.2f}\\\\\n'
    section+=r'''\bottomrule
\end{tabular}
\end{table}
'''
    section+=f'Table~\\ref{{tab:delay_ablation}} reports test errors. The selected NRMSE difference, $\\Delta={{\\rm NRMSE}}_{{\\tau=0}}-{{\\rm NRMSE}}_{{\\rm full}}$, is ${diff:+.6f}$. Its 95\\% interval is $[{ef.ci95_low:.6f},{ef.ci95_high:.6f}]$ from 5000 paired hierarchical bootstrap replicates with blocks of consecutive test bins. {conclusion} This is a forecast comparison on observed events. It does not measure parameter recovery or identify a causal effect of delay.\n\\endgroup\n'
    (ROOT/'experiments/delay_ablation_section.tex').write_text(section,encoding='utf-8')
    response=f'We add a direct delay ablation in the subsection "Ablation of the delay" and its table. We fix tau at zero and retrain all remaining parameters. The cohort transfer becomes Q_0(t)=p(t)I(t), so the pathway remains active without a waiting time. The attention window remains unchanged. Both variants use the same training, validation, and test splits, four events, three folds, and three seeds. We retrain both variants for a total of 72 fits with the same maximum budget of 260 epochs. Validation alone selects checkpoints and deployment seeds. The selected test NRMSE is {full.selected_nrmse:.6f} for the full model and {zero.selected_nrmse:.6f} for tau=0. The difference (tau=0 minus full) is {diff:+.6f}, with a paired 95% bootstrap interval of [{ef.ci95_low:.6f}, {ef.ci95_high:.6f}]. {conclusion} The table also reports variation across seeds, NMAE, and runtime. This added experiment concerns forecasting on the observed events. It does not supply a separate synthetic parameter recovery or noise and sparsity study for the delay ablation.'
    (OUT/'response_addition.txt').write_text(response,encoding='utf-8')
    print(summary.to_string(index=False)); print(effects.to_string(index=False))

if __name__=='__main__': main()
