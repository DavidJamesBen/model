"""Targeted TransPINN continuation; frozen references, no test-based selection.

Save validation-best, fit-best and fit-constrained validation checkpoints so
the trade-off can be inspected without fabricating a superiority claim.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import improve_selected_forecasts as r
import torch
import numpy as np
import pandas as pd
import copy,json,time,hashlib,argparse
from torch.nn import functional as F

OUT=r.ROOT/'output/transpinn_targeted'

def train(event,frame,split,seed,epochs,device):
    r.base.seed_everything(seed)
    tr,va,n=map(int,split)
    raw=frame[['rumor_activity','debunk_activity']].to_numpy(float)
    scale=np.maximum(raw[:tr].sum(0),1)
    rms=np.maximum(np.sqrt(np.mean(raw[:tr]**2,axis=0)),1)
    target=float('inf')
    reference_scores={}
    for path in (r.OUT/'selected_predictions').glob(event+'_fold2_*.csv'):
        d=pd.read_csv(path)
        p=d[['predicted_rumor','predicted_debunk']].to_numpy(float)
        s=r.score(raw[:tr],p[:tr],rms)
        reference_scores[path.stem]=s
        target=min(target,s)
    times=torch.arange(1,n+1,device=device,dtype=torch.float32)[:,None]/n
    y=torch.tensor(raw[:tr],device=device,dtype=torch.float32)
    scales=torch.tensor(scale,device=device,dtype=torch.float32)
    normalizer=torch.tensor(rms,device=device,dtype=torch.float32)
    cfg=r.base.Config()
    model=r.base.RevisedRumorPINN('TransPINN',16).to(device)
    source=r.OUT/'runs'/f'{event}_fold2_transpinn_seed{seed}.pt'
    model.load_state_dict(torch.load(source,map_location=device,weights_only=True))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0007)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=0.00003)
    collocation=torch.linspace(0,1,96,device=device)[:,None]
    history=r.base.initial_history(device)
    best={}
    log=[]
    started=time.perf_counter()
    for epoch in range(epochs+1):
        if epoch%20==0:
            model.eval()
            with torch.no_grad():
                p=(model.observation(times[:va])*scales).cpu().numpy()
            train_score=r.score(raw[:tr],p[:tr],rms)
            val_score=r.score(raw[tr:va],p[tr:va],rms)
            row=dict(epoch=epoch,training_score=train_score,validation_score=val_score)
            log.append(row)
            objectives={'validation':val_score,'fit':train_score}
            if train_score<=target:
                objectives['constrained']=val_score
            for mode,value in objectives.items():
                if mode not in best or value<best[mode]['objective']:
                    best[mode]=dict(objective=value,**row,state=copy.deepcopy(model.state_dict()))
        if epoch==epochs:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        p,residual,initial=r.batched_training_outputs(model,times[:tr],collocation,cfg.finite_difference_step)
        p=p*scales
        loss=12*(((p-y)/normalizer).square().mean()+(torch.log1p(p)-torch.log1p(y)).square().mean()+0.1*((p.cumsum(0)-y.cumsum(0))/scales).square().mean())+0.2*residual.square().mean()+4*F.mse_loss(initial[:,:9],history[:,:9])
        if not torch.isfinite(loss):
            raise FloatingPointError((event,seed,epoch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        scheduler.step()
    stem=OUT/'runs'/f'{event}_seed{seed}'
    pd.DataFrame(log).to_csv(stem.with_suffix('.history.csv'),index=False)
    records={}
    for mode,item in best.items():
        state=item.pop('state')
        model.load_state_dict(state)
        model.eval()
        # No test prediction is inspected or scored until selection is frozen.
        with torch.no_grad():
            pred=(model.observation(times)*scales).cpu().numpy()
            physics=float(r.base.residual_matrix(model,collocation,.0025).square().mean().sqrt())
        path=stem.with_name(stem.name+'_'+mode)
        torch.save({k:v.cpu() for k,v in state.items()},path.with_suffix('.pt'))
        d=pd.DataFrame({'timestamp':frame.timestamp,'split':['train']*tr+['validation']*(va-tr)+['test']*(n-va)})
        for c,ch in enumerate(r.CHANNELS):
            d['actual_'+ch]=raw[:,c]
            d['predicted_'+ch]=pred[:,c]
        d.to_csv(path.with_suffix('.csv'),index=False)
        records[mode]=dict(**item,physics_residual_rms=physics)
    info=dict(event=event,seed=seed,additional_updates=epochs,training_target=target,reference_training_scores=reference_scores,checkpoints=records,runtime_seconds=time.perf_counter()-started,source_checkpoint=str(source.relative_to(r.ROOT)),source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    stem.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=2400)
    args=parser.parse_args()
    (OUT/'runs').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=dict(events=list(r.base.EVENTS),seeds=[11,22,33],additional_updates=args.epochs,learning_rate=0.0007,selection='Training score must match or improve the best frozen reference; then choose by validation score. If no candidate qualifies, report failure and retain previous published model.',test_use='Final evaluation only. Existing test set has been reported in prior explorations; a new untouched holdout is needed for confirmatory claims.',comparison='Targeted additional optimization of TransPINN against frozen baselines; not an equal-budget benchmark.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for event in r.base.EVENTS:
        _,frame,split,_,_=next(x for x in r.base.load_event_folds(event) if x[0]==2)
        for seed in [11,22,33]:
            if (OUT/'runs'/f'{event}_seed{seed}.json').exists():
                continue
            train(event,frame,split,seed,args.epochs,device)

if __name__=='__main__':
    main()
