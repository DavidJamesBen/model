"""Audit absolute forecasting accuracy and regenerate accurately named figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import analyze_multievent_results as audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/rigorous_revision'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
for folder in (FIG, TAB):
    folder.mkdir(parents=True, exist_ok=True)


def metrics(y, pred, train):
    err = pred-y
    rmse = np.sqrt(np.mean(err**2))
    sst = np.sum((y-y.mean())**2)
    return dict(rmse=rmse, nrmse=rmse/max(1., np.sqrt(np.mean(train**2))),
                r2=1-np.sum(err**2)/sst if sst>0 else np.nan)


def main():
    runs, manifest, index = audit.load_inputs()
    verified, cache = audit.validate_predictions_and_metrics(runs, index)
    selection, selection_table = audit.validation_selected_seeds(manifest, index)
    chosen, ech, ev, macro = audit.validation_selected_summaries(verified, selection, index)
    _, _, _, _, _, resources = audit.summaries(verified, index)
    simple = []
    for event in index.events:
        for fold in index.folds_by_event[event]:
            frame = cache[(event, fold, 'TransPINN', selection[(event, fold, 'TransPINN')])]
            for channel in ('rumor', 'debunk'):
                train = frame.loc[frame.split=='train', 'actual_'+channel].to_numpy()
                y = frame.loc[frame.split=='test', 'actual_'+channel].to_numpy()
                for name, value in [('Last value', train[-1]), ('Training mean', train.mean()),
                                    ('Recent mean', train[-min(5,len(train)):].mean())]:
                    simple.append(dict(event=event, fold=fold, channel=channel, model=name,
                                       **metrics(y, np.full_like(y, value), train)))
    combined = pd.concat([chosen[['event','fold','channel','model','rmse','nrmse','r2']],
                          pd.DataFrame(simple)], ignore_index=True)
    combined.to_csv(TAB/'absolute_test_metrics.csv', index=False)
    summary = combined.groupby('model').agg(nrmse=('nrmse','mean'),
        median_r2=('r2','median'), negative_r2=('r2',lambda x:int((x<0).sum())),
        units=('r2','count')).reset_index()
    summary.to_csv(TAB/'absolute_summary.csv',index=False)
    # Raw RMSE is not averaged across events with different units.
    event_channel = combined.groupby(['event','channel','model']).agg(
        rmse=('rmse','mean'), nrmse=('nrmse','mean'), median_r2=('r2','median')).reset_index()
    event_channel.to_csv(TAB/'event_channel_absolute_metrics.csv',index=False)
    trans = combined[combined.model=='TransPINN']
    detail = []
    for event in index.events:
        for channel in ('rumor','debunk'):
            part=trans[(trans.event==event)&(trans.channel==channel)]
            detail.append(f'{audit.EVENT_LABELS[event]} & {channel} & {part.rmse.mean():.3f} & '
                          f'{part.r2.median():.3f} & {int((part.r2<0).sum())}/3 \\\\')
    (TAB/'rmse_r2_rows.tex').write_text('\n'.join(detail),encoding='utf-8')
    (TAB/'absolute_rows.tex').write_text('\n'.join(
        f'{r.model.replace("TimePINN","Causal PINN")} & {r.nrmse:.3f} & '
        f'{r.median_r2:.3f} & {r.negative_r2}/{r.units} \\\\'
        for r in summary.itertuples()),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
    methods=['TransPINN','Last value','Recent mean','Training mean']
    fig, axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    means=combined.groupby(['event','model']).nrmse.mean()
    x=np.arange(4)
    for j,name in enumerate(methods):
        axes[0].bar(x+(j-1.5)*.18,[means.loc[(e,name)] for e in index.events],.18,label=name)
    axes[0].set_xticks(x,['Harvey','CSDC','Ottawa','Sydney'])
    axes[0].set_ylabel('Test NRMSE'); axes[0].legend(fontsize=7)
    axes[0].set_title('(a) Simple forecasting references')
    matrix=np.array([[int(((trans.event==e)&(trans.channel==c)&(trans.r2<0)).sum())
                     for c in ['rumor','debunk']] for e in index.events])
    axes[1].imshow(matrix,vmin=0,vmax=3,cmap='Oranges',aspect='auto')
    axes[1].set_xticks([0,1],['Rumor','Corrective'])
    axes[1].set_yticks(range(4),['Harvey','CSDC','Ottawa','Sydney'])
    for i in range(4):
        for j in range(2): axes[1].text(j,i,f'{matrix[i,j]}/3',ha='center',va='center')
    axes[1].set_title('(b) TransPINN test windows with $R^2<0$')
    for ext in ['pdf','png']: fig.savefig(FIG/f'absolute_accuracy.{ext}',dpi=300)
    plt.close(fig)
    # Same audited data and uncertainty tables; only visible baseline names change.
    original_save = Figure.savefig
    def named_save(self,*args,**kwargs):
        from matplotlib.ticker import FixedFormatter, FixedLocator
        self.canvas.draw()
        for ax in self.axes:
            for axis in (ax.xaxis, ax.yaxis):
                labels = [label.get_text() for label in axis.get_majorticklabels()]
                if any('TimePINN' in label for label in labels):
                    axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
                    axis.set_major_formatter(FixedFormatter(
                        [label.replace('TimePINN','Causal PINN') for label in labels]))
        for text in self.findobj(Text):
            text.set_text(text.get_text().replace('TimePINN','Causal PINN'))
        return original_save(self,*args,**kwargs)
    Figure.savefig=named_save
    audit.FIGURES=FIG
    bs=pd.read_csv(ROOT/'output/tables/validation_selected_paired_block_bootstrap.csv')
    audit.plot_validation_selected_overview(ev,macro,bs,resources,index)
    audit.plot_validation_selected_heatmap(ech,index)
    audit.plot_all_seed_error_distribution(verified,index)
    audit.plot_validation_selected_forecasts(cache,selection,index)
    Figure.savefig=original_save
    (TAB/'metric_audit.json').write_text(json.dumps({'verified_rows':len(verified),
        'selected_units_per_model':24,'simple_baselines_use':'training prefix only',
        'source_manifest':str(audit.RUN_MANIFEST_PATH)},indent=2),encoding='utf-8')


if __name__=='__main__': main()
