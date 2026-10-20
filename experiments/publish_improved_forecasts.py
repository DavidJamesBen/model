"""Validation-only selection, separate test audit and reproducible forecast plot."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_multievent_results as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/forecast_refit'
CHANNELS=('rumor','debunk')

def score(frame):
    train=frame.split.eq('train')
    validation=frame.split.eq('validation')
    y=frame[['actual_rumor','actual_debunk']].to_numpy(float)
    p=frame[['predicted_rumor','predicted_debunk']].to_numpy(float)
    rms=np.maximum(np.sqrt(np.mean(y[train]**2,axis=0)),1)
    return float(np.mean(((p[validation]-y[validation])/rms)**2)+np.mean((np.log1p(p[validation])-np.log1p(y[validation]))**2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    (OUT/'selected_predictions').mkdir(exist_ok=True)
    old_selection=pd.read_csv(ROOT/'output/tables/validation_selected_initializations.csv')
    original_runs=pd.read_csv(ROOT/'output/tables/baseline_runs.csv')
    cache={}
    selection={}
    choices=[]
    metrics=[]
    provenance=[]
    for event in audit.PREFERRED_EVENTS:
        for model in audit.PREFERRED_MODELS:
            slug=model.lower().replace('-','_')
            old_seed=int(old_selection.loc[(old_selection.event==event)&(old_selection.fold==2)&(old_selection.model==model),'selected_seed'].iloc[0])
            old_path=ROOT/'output/predictions'/f'{event}_fold2_{slug}_seed{old_seed}.csv'
            old=pd.read_csv(old_path)
            candidates=[]
            for origin,folder in [('original',ROOT/'output/predictions'),('balanced_refit',OUT/'runs')]:
                for seed in (11,22,33):
                    path=folder/f'{event}_fold2_{slug}_seed{seed}.csv'
                    frame=pd.read_csv(path)
                    assert frame.split.tolist()==old.split.tolist()
                    np.testing.assert_allclose(frame[['actual_rumor','actual_debunk']],old[['actual_rumor','actual_debunk']],rtol=1e-6)
                    assert np.isfinite(frame[['predicted_rumor','predicted_debunk']]).all().all()
                    assert (frame[['predicted_rumor','predicted_debunk']]>=0).all().all()
                    candidates.append((score(frame),origin,seed,path,frame))
            val,origin,seed,path,selected=min(candidates,key=lambda v:(v[0],v[1],v[2]))
            assert val<=score(old)+1e-10
            old_run=original_runs[(original_runs.event==event)&(original_runs.fold==2)&(original_runs.model==model)&(original_runs.seed==old_seed)].iloc[0]
            if origin=='balanced_refit':
                run_info=json.loads(path.with_suffix('.json').read_text())
                assert np.isclose(val,run_info['validation_score'],rtol=1e-5,atol=1e-6)
                history=pd.read_csv(path.with_suffix('.history.csv'))
                assert int(history.loc[history.validation_score.idxmin(),'epoch'])==run_info['best_epoch']
                new_physics=run_info['physics_residual_rms']
            else:
                new_physics=float(original_runs[(original_runs.event==event)&(original_runs.fold==2)&(original_runs.model==model)&(original_runs.seed==seed)].iloc[0].physics_residual_rms)
            # Restore the exact original observation precision after training in float32.
            selected=selected.copy()
            for ch in CHANNELS:
                selected['actual_'+ch]=old['actual_'+ch]
            cache[(event,2,model,seed)]=selected
            selection[(event,2,model)]=seed
            choices.append(dict(event=event,model=model,seed=seed,origin=origin,validation_score=val,original_validation_score=score(old),physics_residual_rms=new_physics,original_physics_residual_rms=float(old_run.physics_residual_rms)))
            figure_data=OUT/'selected_predictions'/f'{event}_fold2_{slug}.csv'
            selected.to_csv(figure_data,index=False)
            provenance.append(dict(event=event,model=model,path=str(figure_data.relative_to(ROOT)),sha256=hashlib.sha256(figure_data.read_bytes()).hexdigest(),prediction_source=str(path.relative_to(ROOT)),prediction_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observation_source=str(old_path.relative_to(ROOT)),observation_source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest()))
            for version,frame in [('original',old),('improved',selected)]:
                for ch in CHANNELS:
                    train=frame.loc[frame.split.eq('train'),'actual_'+ch].to_numpy(float)
                    rms=max(np.sqrt(np.mean(train**2)),1)
                    for region in ('train','validation','test'):
                        part=frame[frame.split.eq(region)]
                        y=part['actual_'+ch].to_numpy(float)
                        p=part['predicted_'+ch].to_numpy(float)
                        metrics.append(dict(event=event,model=model,channel=ch,split=region,version=version,nrmse=np.sqrt(np.mean((p-y)**2))/rms,log_rmse=np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))))
    pd.DataFrame(choices).to_csv(OUT/'selected_models.csv',index=False)
    detail=pd.DataFrame(metrics)
    detail.to_csv(OUT/'metrics_before_after.csv',index=False)
    summary=detail.groupby(['model','split','version'])[['nrmse','log_rmse']].mean()
    summary.to_csv(OUT/'summary_before_after.csv')
    (OUT/'figure_sources.json').write_text(json.dumps(provenance,indent=2))
    audit.FIGURES=OUT
    index=audit.ExperimentIndex(audit.PREFERRED_EVENTS,audit.PREFERRED_MODELS,(11,22,33),{e:(2,) for e in audit.PREFERRED_EVENTS})
    audit.plot_validation_selected_forecasts(cache,selection,index,
        protocol_note='Chronological splits | White: training; blue: validation; red: test | Model selection: validation only',
        label_splits=True)
    trans=detail[detail.model.eq('TransPINN')].groupby(['split','version'])[['nrmse','log_rmse']].mean()
    lines=['# Forecast refit audit','',
           'This is a new fold-2 optimization experiment. It does not update the manuscript tables or its all-fold benchmark.',
           'All five architectures receive the same 800-update budget, seeds 11/22/33, balanced activity loss, physics/history penalties and validation-only selection rule.',
           'Original predictions are retained as fallback candidates under that same rule. No test values enter optimization or selection.',
           'The data, split boundaries, time features, architecture and physical equations are unchanged. No observations are smoothed.',
           'The new loss combines training-RMS-normalized activity MSE, log1p activity MSE and 0.1 times normalized cumulative MSE; checkpoint selection uses the first two terms on validation only.',
           'Rescaling the observation loss changes its balance against the physics penalties. Physical residuals are reported in selected_models.csv; improved activity prediction must not be interpreted as improved physical consistency.',
           'Execution was optimized by combining independent time queries into one batch. Outputs and parameter gradients were checked against the separate-forward implementation in float64 for all five architectures (rtol 1e-7, atol 1e-8 for gradients). Archived source versions identify the execution variants; tiny float32 differences are possible.',
           '', '## TransPINN: equal-weight mean over four events and two channels','',trans.to_string(),'',
           '## Selected candidates','',pd.DataFrame(choices).to_string(index=False),'',
           '## Reproduce','',
           'Run `python experiments/improve_selected_forecasts.py --epochs 800`, then `python experiments/publish_improved_forecasts.py --publish`.',
           'Model checkpoints, per-checkpoint validation histories, predictions and source hashes are stored beside this report.',
           'For publication, rerun all three folds with the new protocol and update the methods, benchmark tables and captions together. The current manuscript PDF is unchanged.']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    zh=['# 拟合改进结果与适用范围','',
        '本次重新训练图中的第 2 折，覆盖四个事件、五种方法和三个随机种子，共 60 个新候选。新候选使用相同的 800 次参数更新预算；原实验为最多 260 次。',
        '保持原观测值、训练/验证/测试划分、网络结构、时滞方程和历史条件。调整的是观测损失的尺度及训练预算。',
        '按训练集 RMS 标准化每个时间段的活动量误差，加入 log(1+activity) 误差，将按训练总量标准化的累计误差权重设为 0.1。物理残差与历史损失仍保留。',
        '新模型的检查点只根据验证集选择。随后对新旧候选使用同一验证分数选出最终曲线；测试数据不用于训练或选择。',
        '', '## TransPINN 的改进前后对比','',
        '下表对四个事件、两个通道等权平均。NRMSE 衡量原始活动量误差；对数 RMSE 对应图的纵轴。两者越小越好。','',
        '| 区间 | 原 NRMSE | 新 NRMSE | 原对数 RMSE | 新对数 RMSE |',
        '|---|---:|---:|---:|---:|']
    for region,label in [('train','训练'),('validation','验证'),('test','测试')]:
        a=trans.loc[(region,'original')]
        b=trans.loc[(region,'improved')]
        zh.append(f'| {label} | {a.nrmse:.6f} | {b.nrmse:.6f} | {a.log_rmse:.6f} | {b.log_rmse:.6f} |')
    zh.extend(['','## 限制与复核','',
        '提高对数尺度的拟合精度，不保证每个事件、通道、峰值或物理残差都改善。完整的分项误差见 metrics_before_after.csv；物理残差对比见 selected_models.csv。',
        '重新标定观测损失会改变它与物理约束的相对强度，不能把活动量拟合改善解释为物理机制识别改善。',
        '这是第 2 折的新优化实验，不等同于论文原有三折、260 次更新的基准结果。原论文 PDF 和统计表尚未更新；用于论文时需要统一更新实验方法、其余折和统计表。',
        '原图备份位于 original_figure/；原始预测文件没有覆盖。新预测、检查点、验证历史、脚本快照和来源哈希均保存在本目录。'])
    (OUT/'改进说明.md').write_text('\n'.join(zh),encoding='utf-8')
    if args.publish:
        backup=OUT/'original_figure'
        backup.mkdir(exist_ok=True)
        for ext in ('pdf','png'):
            target=ROOT/'output/figures'/f'validation_selected_forecasts.{ext}'
            if not (backup/target.name).exists():
                shutil.copy2(target,backup/target.name)
            shutil.copy2(OUT/target.name,target)
    print(trans.to_string())
    print(pd.DataFrame(choices).groupby('origin').size().to_string())

if __name__=='__main__':
    main()
