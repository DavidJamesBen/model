"""Loss/gradient snapshots at archived checkpoints; no fitting or model selection.

These measurements diagnose scales at selected checkpoints. They are not
training histories and cannot establish the cause of a forecasting difference.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import run_fair_benchmark as fair

OUT=fair.ROOT/'output/benchmark_design'

def gradient(loss,parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p,g in zip(parameters,values)])

def cosine(a,b):
    return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))

def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    selected=pd.read_csv(fair.OUT/'selected_initializations.csv')
    selected=selected[selected.model.eq('TransPINN')]
    rows=[]
    for event in fair.base.EVENTS:
        for fold,frame,split,_,_ in fair.base.load_event_folds(event):
            record=selected[(selected.event==event)&(selected.fold==fold)].iloc[0]
            seed=int(record.seed)
            stem=fair.OUT/'runs'/f'{event}_fold{fold}_TransPINN_seed{seed}'
            info=json.loads(stem.with_suffix('.json').read_text())
            cfg=fair.base.Config(**info['config'])
            model=fair.factory('TransPINN',cfg.sequence_length)
            model.load_state_dict(torch.load(stem.with_suffix('.pt'),weights_only=True))
            train,val,n=map(int,split)
            raw=frame[['rumor_activity','debunk_activity']].to_numpy(np.float32)
            scale=np.maximum(raw[:train].sum(0),1).astype(np.float32)
            target=torch.tensor(raw/scale)
            target_cum=torch.tensor(np.cumsum(raw,axis=0)/scale)
            times=(torch.arange(1,n+1,dtype=torch.float32)/n).view(-1,1)
            pred=model.observation(times)
            pred_cum=pred.cumsum(0)
            inc=F.mse_loss(pred[:train],target[:train])
            cum=F.mse_loss(pred_cum[:train],target_cum[:train])
            val_inc=F.mse_loss(pred[train:val],target[train:val])
            val_cum=F.mse_loss(pred_cum[train:val],target_cum[train:val])
            residual=fair.base.residual_matrix(model,torch.linspace(0,1,96).view(-1,1),cfg.finite_difference_step)
            physical=residual.square().mean()
            hist=F.mse_loss(model(torch.zeros(1,1))[:,:9],fair.base.initial_history(torch.device('cpu'))[:,:9])
            weights=(cfg.data_weight,cfg.physics_weight*min(1.,record.best_epoch/120.),cfg.history_weight)
            pars=list(model.backbone.parameters())
            gd=gradient(weights[0]*(inc+cum),pars)
            gp=gradient(weights[1]*physical,pars)
            gh=gradient(weights[2]*hist,pars)
            gi=gradient(inc,pars); gc=gradient(cum,pars)
            rows.append(dict(event=event,fold=fold,seed=seed,best_epoch=int(record.best_epoch),
                train_increment_mse=float(inc.detach()),train_cumulative_mse=float(cum.detach()),
                train_cumulative_share=float((cum/(cum+inc)).detach()),
                validation_cumulative_share=float((val_cum/(val_cum+val_inc)).detach()),
                weighted_data_gradient_norm=float(gd.norm()),
                weighted_physics_gradient_norm=float(gp.norm()),
                weighted_history_gradient_norm=float(gh.norm()),
                physics_to_data_gradient_ratio=float(gp.norm()/gd.norm().clamp_min(1e-30)),
                data_physics_gradient_cosine=cosine(gd,gp),
                data_history_gradient_cosine=cosine(gd,gh),
                cumulative_to_increment_gradient_ratio=float(gc.norm()/gi.norm().clamp_min(1e-30)),
                state_residual_rms=float(residual[:,:9].square().mean().sqrt().detach()),
                flux_residual_rms=float(residual[:,9:].square().mean().sqrt().detach())))
    results=pd.DataFrame(rows)
    results.to_csv(OUT/'checkpoint_diagnostics.csv',index=False)
    cols=['train_cumulative_share','validation_cumulative_share',
          'physics_to_data_gradient_ratio','data_physics_gradient_cosine',
          'cumulative_to_increment_gradient_ratio']
    summary=results[cols].agg(['min','median','max'])
    summary.to_csv(OUT/'checkpoint_diagnostics_summary.csv')
    print(summary.to_string())
    print('12 archived checkpoints diagnosed; no weights changed or fits performed.')

if __name__=='__main__':
    main()
