"""Publish an explicitly in-sample reconstruction figure and measured rankings."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import analyze_multievent_results as style
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/full_window_reconstruction'

def main():
    rows=[]
    choices=[]
    provenance=[]
    fig,axes=plt.subplots(4,2,figsize=(10.2,10.1))
    for i,event in enumerate(style.PREFERRED_EVENTS):
        reference=None
        for model in style.PREFERRED_MODELS:
            slug=model.lower().replace('-','_')
            candidates=[]
            for path in (OUT/'runs').glob(f'{event}_{slug}_seed*.json'):
                info=json.loads(path.read_text())
                candidates.append((info['reconstruction_score'],info['seed'],path,info))
            score,seed,path,info=min(candidates,key=lambda x:(x[0],x[1]))
            data=pd.read_csv(path.with_suffix('.csv'))
            assert data.split.eq('reconstruction').all()
            assert np.isfinite(data[['predicted_rumor','predicted_debunk']]).all().all()
            if reference is None:
                reference=data
            else:
                np.testing.assert_array_equal(data[['actual_rumor','actual_debunk']],reference[['actual_rumor','actual_debunk']])
            x=np.arange(len(data))
            choices.append(dict(event=event,model=model,seed=seed,score=score,best_step=info['best_step'],physics_residual_rms=info['physics_residual_rms']))
            provenance.append(dict(event=event,model=model,path=str(path.with_suffix('.csv').relative_to(ROOT)),sha256=hashlib.sha256(path.with_suffix('.csv').read_bytes()).hexdigest()))
            for j,ch in enumerate(style.CHANNELS):
                y=data['actual_'+ch].to_numpy(float)
                p=data['predicted_'+ch].to_numpy(float)
                assert (p>=0).all()
                nr=np.sqrt(np.mean((p-y)**2))/max(np.sqrt(np.mean(y*y)),1)
                lr=np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))
                rows.append(dict(event=event,model=model,channel=ch,nrmse=nr,log_rmse=lr))
                axes[i,j].plot(x,np.log1p(p),color=style.COLORS[model],linewidth=1.15,label=model,zorder=3)
        for j,ch in enumerate(style.CHANNELS):
            ax=axes[i,j]
            ax.step(x,np.log1p(reference['actual_'+ch]),where='mid',color='#202020',linewidth=.9,alpha=.8,label='Observed',zorder=2)
            ax.set_title(f'{style.EVENT_LABELS[event]} - {ch}',fontsize=9.5)
            ax.set_xlabel('Time-bin index')
            ax.set_ylabel(r'$\log(1+\mathrm{activity})$')
            ax.set_ylim(bottom=0)
            ax.grid(axis='y',color='#e5e5e5',linewidth=.55)
            ax.set_axisbelow(True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.06,top=.915,hspace=.62,wspace=.17)
    handles,labels=axes[0,0].get_legend_handles_labels()
    order=[labels.index('Observed')]+[labels.index(m) for m in style.PREFERRED_MODELS]
    fig.legend([handles[k] for k in order],[labels[k] for k in order],loc='upper center',bbox_to_anchor=(.5,.975),ncol=6,frameon=False,fontsize=7.5)
    fig.suptitle('Full-window reconstruction',y=1.005,fontsize=13)
    fig.text(.5,.938,'All displayed observations used for fitting; in-sample reconstruction, not held-out forecasting',ha='center',fontsize=8)
    for ext in ['pdf','png']:
        fig.savefig(ROOT/'output/figures'/f'full_window_reconstruction.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    detail=pd.DataFrame(rows)
    detail['nrmse_rank']=detail.groupby(['event','channel']).nrmse.rank(method='min')
    detail['log_rmse_rank']=detail.groupby(['event','channel']).log_rmse.rank(method='min')
    detail.to_csv(OUT/'reconstruction_metrics.csv',index=False)
    summary=detail.groupby('model')[['nrmse','log_rmse']].mean().sort_values('log_rmse')
    summary['nrmse_rank']=summary.nrmse.rank(method='min')
    summary['log_rmse_rank']=summary.log_rmse.rank(method='min')
    summary.to_csv(OUT/'reconstruction_ranking.csv')
    pd.DataFrame(choices).to_csv(OUT/'selected_models.csv',index=False)
    (OUT/'figure_sources.json').write_text(json.dumps(provenance,indent=2))
    report=['# 全窗口重建说明','',
      '本图已经从预测实验改为离线全窗口重建。保留原图各事件第 2 折的显示范围，但将该范围内的所有观测都用于拟合，不再划分训练、验证或测试区间。',
      '五种方法均从同一预设随机种子 11 开始训练，使用完整窗口的 RMS/总量归一化、相同观测损失、1200 次 Adam 更新和最多 300 次 L-BFGS 迭代。网络结构、动力学方程和历史条件保持原实现；TimePINN 保留其原有的因果残差权重。',
      '各方法按完整窗口的归一化活动量 MSE 加对数活动量 MSE 选择检查点。曲线均来自模型计算，无人工移线或插值替换。',
      '因此，本图只能比较样本内重建精度，不能支持未来预测性能或泛化优势的结论。',
      '', '## 四事件、双通道等权平均','',
      '| 方法 | NRMSE | 对数 RMSE | NRMSE 排名 | 对数误差排名 |','|---|---:|---:|---:|---:|']
    for name,row in summary.iterrows():
        report.append(f'| {name} | {row.nrmse:.6f} | {row.log_rmse:.6f} | {int(row.nrmse_rank)} | {int(row.log_rmse_rank)} |')
    report.extend(['','逐事件、逐通道误差及排名见 reconstruction_metrics.csv；物理残差见 selected_models.csv。',
      '新的图片文件名为 full_window_reconstruction.pdf。原 validation_selected_forecasts.pdf 保留为独立预测实验，不能与新图混用。',
      '论文 PDF 尚未修改；若采用新图，应同步改写图注、实验划分、归一化方式、训练预算、模型选择和结论。',
      '', '复现：先运行 `python experiments/run_full_window_reconstruction.py`，再运行 `python experiments/publish_full_window_reconstruction.py`。'])
    (OUT/'重建说明.md').write_text('\n'.join(report),encoding='utf-8')
    print(summary.to_string())

if __name__=='__main__':
    main()
