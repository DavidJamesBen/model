from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/lyapunov'

def sci(x):
    return f'{x:.2e}'

def main():
    d=pd.read_csv(OUT/'lyapunov_runs.csv')
    assert len(d)==116 and d.status.eq('ok').all()
    baseline=d[(d.kind!='delay_sweep')&(d.direction_seed==11)&(d.renorm==1)&(d.history_scale==1)]
    refined=baseline[baseline.requested_step==.00125]
    rows=[]
    for _,r in refined.iterrows():
        part=baseline[baseline.case==r['case']]
        rows.append(dict(case=r['case'],lambda_100_200=r.lambda_100_200,lambda_100_300=r.lambda_100_300,lambda_100_500=r.lambda_100_500,lambda_300_500=r.lambda_300_500,step_spread=part.lambda_100_500.max()-part.lambda_100_500.min(),final_active=r.final_active))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'convergence_summary.csv',index=False)
    tex=[]
    for _,r in table.iterrows():
        tex.append(r['case'].replace('_',r'\_')+' & '+' & '.join(sci(r[k]) for k in ['lambda_100_200','lambda_100_300','lambda_100_500','step_spread'])+r' \\')
    (OUT/'convergence_rows.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    params=pd.read_csv(OUT/'parameter_sets.csv')
    lines=[]
    for _,r in params.iterrows():
        lines.append(r['case'].replace('_',r'\_')+' & '+' & '.join(f'{r[k]:.6f}' for k in ['beta_1','beta_2','tau_normalized','q_R','q_D','chi'])+r' \\')
    (OUT/'parameter_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    stats=dict(runs=len(d),max_abs_late=float(d.lambda_100_500.abs().max()),max_abs_tail=float(d.lambda_300_500.abs().max()),max_step_spread=float(table.step_spread.max()),max_mass_error=float(d.mass_error.max()),minimum_state=float(d.minimum_state.min()),max_final_active=float(d.final_active.max()),max_final_rhs=float(d.final_rhs.max()))
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2))
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    for event in ['harvey','csdc','ottawa','sydney']:
        r=refined[refined.case.eq(event+'_fold2')].iloc[0]
        trace=pd.read_csv(OUT/'traces'/r.trace)
        t=trace.time.to_numpy();growth=trace.log_growth.to_numpy()
        axs[0].plot(t,np.cumsum(growth)/t,label=event.title())
        j=np.searchsorted(t,100,side='right'); t0=t[j-1]
        axs[1].plot(t[j:],np.cumsum(growth[j:])/(t[j:]-t0),label=event.title())
    axs[0].set(xlabel='Normalized time',ylabel='Running exponent estimate',title='(a) Includes initial transient',xlim=(1,500))
    axs[1].set(xlabel='Normalized time',ylabel='Post-transient exponent estimate',title='(b) Discarding first 100 time units',xlim=(100,500))
    for ax in axs:
        ax.axhline(0,color='black',lw=.7,ls='--');ax.grid(alpha=.25)
    axs[0].legend(fontsize=8)
    for ext in ['pdf','png']:
        fig.savefig(OUT/f'lyapunov_convergence.{ext}',dpi=220)
    plt.close(fig)
    text=f'''# 李雅普诺夫混沌量化分析

已完成 116 组条件性 DDE 变分计算。不是从注意力矩阵或预测曲线直接估计。

- 12 组归档神经参数（四事件、三折）和 4 组独立训练数据 DDE 拟合参数。
- 固定经验模型其余系数；封闭人口、总量为 1；归一化时间单位。
- RK4 分步积分和延迟值线性插值；复步长法计算变分方程的方向导数。
- 每次重归一化整个延迟历史函数；人群扰动和为零。
- 步长上限 0.005、0.0025、0.00125；积分至 500，丢弃前 100；三个初始扰动方向。
- 中间折另检查归一化间隔 0.5、2，活动历史缩放 0.5、1.5，以及时滞 0.01、0.05、0.10、0.15、0.22。

全部计算中 |lambda(100,500)| 最大为 {stats['max_abs_late']:.3e}；(300,500) 区间的指数在浮点精度下为零。16 组基准的步长变化差异最大为 {stats['max_step_spread']:.3e}。末期活动状态总量最大为 {stats['max_final_active']:.3e}。

结果解释：存在连续的平衡点族 X*(s)=(5s/13,8s/13,0,0,0,0,1-s,0.2,0.2)。沿该族的中性方向使最大指数趋零，不能将其称为混沌证据。所有测试轨迹都趋向平衡，未发现持续正指数。此结论只适用于测试的参数与历史，不能证明所有参数下无混沌，也不证明真实事件无混沌。有限时间单方向增长率不是有限时间最大奇异值指数。

数值验证见 numerical_checks.json：解析稳定/不稳定线性 DDE 检验、复步长方向导数与中心差分检验、独立自适应求解器检验。变分算法在每一步显式保持零和扰动，避免守恒方向的舍入误差污染。

复现：python experiments/run_lyapunov_analysis.py；python experiments/publish_lyapunov_analysis.py。
来源和参数哈希见 manifest.json；完整参数见 parameter_sets.csv；逐次归一化记录见 traces/。
方法参考：G. Ansmann, Chaos 28, 043116 (2018), https://doi.org/10.1063/1.5019320 。本实现使用独立固定步长积分器，没有调用 JiTCDDE。
'''
    (OUT/'分析说明.md').write_text(text,encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    main()
