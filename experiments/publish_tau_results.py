"""Audit all fixed-delay fits and produce the manuscript's two-protocol figures."""
from pathlib import Path
import json, itertools, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/tau_sweep'
T=BASE/'tables'; F=BASE/'figures'; F.mkdir(exist_ok=True)
d=pd.read_csv(T/'tau_sweep_runs.csv')
manifest=json.loads((T/'tau_sweep_manifest.json').read_text())
m=pd.DataFrame(manifest['runs'])
keys=['fixed_tau_normalized','event','fold','model','seed']
taus=[.02,.05,.07,.10,.15];events=['harvey','csdc','ottawa','sydney'];models=['PINN','TransPINN']
expected=set(itertools.product(taus,events,[1,2,3],models,[11,22,33]))
assert set(map(tuple,m[keys].to_numpy()))==expected and len(m)==360
assert len(d)==720 and not d.duplicated(keys+['channel']).any()
assert d.groupby(keys).channel.apply(set).eq({'rumor','debunk'}).all()
assert np.allclose(d.tau_normalized,d.fixed_tau_normalized) and d.tau_is_fixed.all()
assert manifest['config']['epochs']==260
observations={}
for row in m.to_dict('records'):
    tau,event,fold,model,seed=(row[k] for k in keys)
    p=BASE/'predictions'/f"tau_{tau:.3f}_{event}_fold{fold}_{model.lower()}_seed{seed}.csv".replace(f'{tau:.3f}',f'{tau:.3f}'.replace('.','p'))
    pred=pd.read_csv(p)
    ident=(event,fold)
    obs=pred[['timestamp','split','actual_rumor','actual_debunk']]
    if ident in observations:pd.testing.assert_frame_equal(obs,observations[ident])
    else:observations[ident]=obs
    subset=d
    for k in keys:subset=subset[subset[k]==row[k]]
    for channel in ['rumor','debunk']:
        test=pred['split'].eq('test');train=pred['split'].eq('train')
        actual=pred[f'actual_{channel}'];forecast=pred[f'predicted_{channel}']
        error=np.sqrt(np.mean((actual[test]-forecast[test])**2))/max(1.,np.sqrt(np.mean(actual[train]**2)))
        assert np.isclose(error,subset.loc[subset.channel.eq(channel),'nrmse'].iloc[0],rtol=1e-5)

# Identical prespecified seed candidates; no test metric enters selection.
selected=m.sort_values(keys[:-1]+['validation_score','seed']).drop_duplicates(keys[:-1])
v=d.merge(selected[keys],on=keys,validate='many_to_one')
selected[keys+['validation_score','best_epoch']].to_csv(T/'tau_validation_selection.csv',index=False)
macro=d.groupby(['fixed_tau_normalized','model','seed']).nrmse.mean().reset_index()
summary=macro.groupby(['fixed_tau_normalized','model']).nrmse.agg(['mean','std']).reset_index()
deploy=v.groupby(['fixed_tau_normalized','model']).nrmse.mean().unstack()
table=summary.pivot(index='fixed_tau_normalized',columns='model')
result=pd.DataFrame(index=taus)
for model in models:
    result[model+'_all_mean']=table['mean'][model]
    result[model+'_all_sd']=table['std'][model]
    result[model+'_selected']=deploy[model]
result['relative_reduction_percent']=100*(deploy.PINN-deploy.TransPINN)/deploy.PINN
result.index.name='tau_s'
result.to_csv(T/'tau_publication_summary.csv')

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                     'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
styles={'PINN':dict(color='#235A88',marker='s',linestyle='--'),
        'TransPINN':dict(color='#B12A34',marker='o',linestyle='-')}
fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
for model in models:
    axes[0].errorbar(taus,result[model+'_all_mean'],yerr=result[model+'_all_sd'],
                     capsize=2,markersize=3,linewidth=1.2,label=model,**styles[model])
    axes[1].plot(taus,result[model+'_selected'],markersize=3,linewidth=1.2,label=model,**styles[model])
for ax,title in zip(axes,['(a) All initializations: mean and SD','(b) Validation-selected prediction']):
    ax.set(title=title,xlabel=r'Fixed normalized delay $\tau_s$',ylabel='Macro test NRMSE')
    ax.set_xticks(taus);ax.grid(alpha=.2);ax.legend(frameon=False)
for ext in ['pdf','png']:fig.savefig(F/f'tau_protocol_comparison.{ext}',dpi=300)
plt.close(fig)
eventmeans=v.groupby(['fixed_tau_normalized','event','model']).nrmse.mean()
fig,axes=plt.subplots(2,2,figsize=(7.1,4.4),layout='constrained')
for ax,event,title in zip(axes.flat,events,['(a) Hurricane Harvey','(b) CSDC-Rumor','(c) Ottawa shooting','(d) Sydney cafe siege']):
    for model in models:ax.plot(taus,[eventmeans.loc[(tau,event,model)] for tau in taus],markersize=3,linewidth=1.2,label=model,**styles[model])
    ax.set(title=title,xlabel=r'Fixed normalized delay $\tau_s$',ylabel='Test NRMSE')
    ax.set_xticks(taus);ax.grid(alpha=.2);ax.legend(frameon=False)
for ext in ['pdf','png']:fig.savefig(F/f'tau_event_comparison.{ext}',dpi=300)
plt.close(fig)
rows=[]
for tau,r in result.iterrows():
    rows.append(f"{tau:.2f} & ${r.PINN_all_mean:.3f}\\pm{r.PINN_all_sd:.3f}$ & ${r.TransPINN_all_mean:.3f}\\pm{r.TransPINN_all_sd:.3f}$ & {r.PINN_selected:.3f} & {r.TransPINN_selected:.3f} & {r.relative_reduction_percent:+.2f}\\\\")
(T/'tau_table_rows.tex').write_text('\n'.join(rows)+'\n')
(T/'tau_publication_audit.json').write_text(json.dumps({'fits':360,'channels':720,'prediction_files_checked':360,
 'selection':'minimum validation score; seed breaks ties','runs_sha256':hashlib.sha256((T/'tau_sweep_runs.csv').read_bytes()).hexdigest()},indent=2))
print(result.to_string())
