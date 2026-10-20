"""Validate the complete supplemental grid and publish honest diagnostics."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/rigorous_revision'
TAB=OUT/'tables';FIG=OUT/'figures'
EVENTS=('harvey','csdc','ottawa','sydney')
VARIANTS=('full','no_fatigue','no_inhibition','no_physics')
LABELS={'full':'Full TransPINN','no_fatigue':'No fatigue','no_inhibition':'No inhibition',
        'no_physics':'No physics'}


def main():
    manifest=[];rows=[];observations={}
    for event,fold,variant,seed in itertools.product(EVENTS,(1,2,3),VARIANTS,(11,22,33)):
        stem=f'{event}_fold{fold}_{variant}_seed{seed}'
        record=json.loads((OUT/'full'/f'{stem}.json').read_text())
        assert record['config']['epochs']==260
        assert record['variant']==variant
        frame=pd.read_csv(OUT/'full'/f'{stem}.csv')
        assert (OUT/'full'/f'{stem}.pt').exists()
        cols=['split','actual_rumor','actual_debunk']
        key=(event,fold)
        if key in observations:pd.testing.assert_frame_equal(frame[cols],observations[key])
        else:observations[key]=frame[cols]
        for row in record['rows']:
            channel=row['channel'];train=frame.split=='train';test=frame.split=='test'
            y=frame.loc[test,'actual_'+channel].to_numpy()
            pred=frame.loc[test,'predicted_'+channel].to_numpy()
            den=max(1.,np.sqrt(np.mean(frame.loc[train,'actual_'+channel].to_numpy()**2)))
            np.testing.assert_allclose(np.sqrt(np.mean((pred-y)**2))/den,row['nrmse'],rtol=1e-6)
            rows.append(dict(**row,variant=variant))
        manifest.append(dict(**record['run'],variant=variant))
    runs=pd.DataFrame(rows);info=pd.DataFrame(manifest)
    assert len(runs)==288 and len(info)==144
    selected=info.sort_values(['validation_score','seed']).drop_duplicates(['event','fold','variant'])
    chosen=runs.merge(selected[['event','fold','variant','seed']],on=['event','fold','variant','seed'])
    runs.to_csv(TAB/'mechanism_all_runs.csv',index=False)
    selected.to_csv(TAB/'mechanism_selection.csv',index=False)
    chosen.to_csv(TAB/'mechanism_selected_runs.csv',index=False)
    summaries=[]
    for variant in VARIANTS:
        seed_macro=runs[runs.variant==variant].groupby('seed').nrmse.mean()
        select=chosen[chosen.variant==variant]
        summaries.append(dict(variant=variant,all_mean=seed_macro.mean(),all_sd=seed_macro.std(),
            selected_nrmse=select.nrmse.mean(),negative_r2=int((select.r2<0).sum()),
            median_r2=select.r2.median(),mean_residual_rms=select.physics_residual_rms.mean()))
    summary=pd.DataFrame(summaries)
    summary.to_csv(TAB/'mechanism_summary.csv',index=False)
    (TAB/'mechanism_rows.tex').write_text('\n'.join(
        f'{LABELS[r.variant]} & ${r.all_mean:.3f}\\pm {r.all_sd:.3f}$ & '
        f'{r.selected_nrmse:.3f} & {r.negative_r2}/24 \\\\' for r in summary.itertuples()),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    # Confirm the full rerun reproduces the archived comparison before reusing it.
    old=pd.read_csv(ROOT/'output/tables/baseline_runs.csv')
    compare=runs[runs.variant=='full'].merge(old[old.model=='TransPINN'],
        on=['event','fold','seed','channel'],suffixes=('_new','_old'))
    diff=float(np.max(np.abs(compare.nrmse_new-compare.nrmse_old)))
    print('Full-model rerun maximum NRMSE difference',diff,flush=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    x=np.arange(4)
    for j,variant in enumerate(VARIANTS):
        ev=chosen[chosen.variant==variant].groupby('event').nrmse.mean()
        axes[0].bar(x+(j-1.5)*.18,[ev[e] for e in EVENTS],.18,label=LABELS[variant])
    axes[0].set_xticks(x,['Harvey','CSDC','Ottawa','Sydney'])
    axes[0].set_ylabel('Validation-selected test NRMSE')
    axes[0].legend(fontsize=7);axes[0].set_title('(a) Mechanism ablations')
    syn=pd.read_csv(TAB/'synthetic_numerical_recovery.csv')
    for n,marker in [(12,'o'),(48,'s')]:
        group=syn[syn.samples==n].groupby('noise_fraction')
        for metric,style,label in [('parameter_relative_l2','-','Parameter error'),
                                    ('prediction_relative_l2','--','Observation error')]:
            vals=group[metric].mean()
            axes[1].plot(vals.index*100,vals.values,marker+style,label=f'{label}, n={n}')
    axes[1].set_yscale('log');axes[1].set_xlabel('Synthetic multiplicative noise (%)')
    axes[1].set_ylabel('Mean relative $L^2$ error')
    axes[1].set_title('(b) Numerical inverse diagnostic')
    axes[1].legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(FIG/f'mechanism_and_recovery.{ext}',dpi=300)
    plt.close(fig)
    synsum=syn.groupby(['samples','noise_fraction']).agg(
        parameter_error=('parameter_relative_l2','mean'),observation_error=('prediction_relative_l2','mean'),
        converged=('converged','sum')).reset_index()
    synsum.to_csv(TAB/'synthetic_recovery_summary.csv',index=False)
    (TAB/'synthetic_rows.tex').write_text('\n'.join(
        f'{r.samples} & {r.noise_fraction*100:.0f} & {r.parameter_error:.3f} & '
        f'{r.observation_error:.5f} & {r.converged}/3 \\\\' for r in synsum.itertuples()),encoding='utf-8')
    replay=pd.read_csv(TAB/'dde_replay_diagnostics.csv')
    (TAB/'replay_rows.tex').write_text('\n'.join(
        f'{e.title()} & {g.head_nrmse.mean():.3f} & {g.dde_nrmse.mean():.3f} & '
        f'{g.replay_head_gap.mean():.3f} \\\\' for e,g in replay.groupby('event')),encoding='utf-8')
    (TAB/'supplemental_audit.json').write_text(json.dumps(dict(fits=144,scored_rows=288,
        selection='minimum validation objective, ties lower seed; no test selection',
        full_rerun_maximum_nrmse_difference=diff,
        synthetic_converged=int(syn.converged.sum()),synthetic_fits=len(syn)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
