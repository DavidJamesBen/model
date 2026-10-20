"""Offline reconstruction of every observed bin in the former fold-2 window.

All methods fit the same full window with the same objective and optimization
budget. These are in-sample reconstructions, never held-out forecasts.
"""
import improve_selected_forecasts as r
import torch
import numpy as np
import pandas as pd
import json,copy,time,argparse,hashlib
from pathlib import Path
from torch.nn import functional as F

OUT=r.ROOT/'output/full_window_reconstruction'

def fit(event,frame,name,seed,epochs,lbfgs_steps,device):
    r.base.seed_everything(seed)
    raw=frame[['rumor_activity','debunk_activity']].to_numpy(float)
    n=len(raw)
    scale=np.maximum(raw.sum(0),1)
    rms=np.maximum(np.sqrt(np.mean(raw**2,axis=0)),1)
    y=torch.tensor(raw,device=device,dtype=torch.float32)
    scales=torch.tensor(scale,device=device,dtype=torch.float32)
    normalizer=torch.tensor(rms,device=device,dtype=torch.float32)
    times=torch.arange(1,n+1,device=device,dtype=torch.float32)[:,None]/n
    collocation=torch.linspace(0,1,96,device=device)[:,None]
    history=r.base.initial_history(device)
    cfg=r.base.Config()
    model=r.base.RevisedRumorPINN(name,16).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.0001)
    best_score=float('inf')
    best_state=None
    best_step=None
    log=[]
    evaluations=0
    started=time.perf_counter()

    def objective():
        nonlocal evaluations
        evaluations+=1
        p,residual,initial=r.batched_training_outputs(model,times,collocation,.0025)
        p=p*scales
        data=((p-y)/normalizer).square().mean()+(torch.log1p(p)-torch.log1p(y)).square().mean()
        cumulative=((p.cumsum(0)-y.cumsum(0))/scales).square().mean()
        physics=r.base.causal_physics_loss(residual,cfg) if name=='TimePINN' else residual.square().mean()
        hist=F.mse_loss(initial[:,:9],history[:,:9])
        return 12*(data+.1*cumulative)+.2*physics+4*hist

    def checkpoint(step):
        nonlocal best_score,best_state,best_step
        model.eval()
        with torch.no_grad():
            p=(model.observation(times)*scales).cpu().numpy()
        score=r.score(raw,p,rms)
        log.append(dict(step=step,reconstruction_score=score))
        if score<best_score:
            best_score=score
            best_state=copy.deepcopy(model.state_dict())
            best_step=step
        model.train()

    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True)
        loss=objective()
        if not torch.isfinite(loss):
            raise FloatingPointError((event,name,seed,epoch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optimizer.step()
        scheduler.step()
        if epoch%20==0:
            checkpoint('adam_'+str(epoch))
    model.load_state_dict(best_state)
    # Same quasi-Newton refinement for every architecture.
    lbfgs=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=20,max_eval=30,history_size=30,line_search_fn='strong_wolfe')
    for block in range(lbfgs_steps//20):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss=objective()
            loss.backward()
            return loss
        lbfgs.step(closure)
        checkpoint('lbfgs_'+str((block+1)*20))
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred=(model.observation(times)*scales).cpu().numpy()
        physics=float(r.base.residual_matrix(model,collocation,.0025).square().mean().sqrt())
    stem=OUT/'runs'/f'{event}_{name.lower().replace("-","_")}_seed{seed}'
    torch.save({k:v.cpu() for k,v in best_state.items()},stem.with_suffix('.pt'))
    d=pd.DataFrame({'timestamp':frame.timestamp,'split':['reconstruction']*n})
    metrics=[]
    for c,ch in enumerate(r.CHANNELS):
        d['actual_'+ch]=raw[:,c]
        d['predicted_'+ch]=pred[:,c]
        metrics.append(dict(channel=ch,nrmse=float(np.sqrt(np.mean((pred[:,c]-raw[:,c])**2))/rms[c]),log_rmse=float(np.sqrt(np.mean((np.log1p(pred[:,c])-np.log1p(raw[:,c]))**2)))))
    d.to_csv(stem.with_suffix('.csv'),index=False)
    pd.DataFrame(log).to_csv(stem.with_suffix('.history.csv'),index=False)
    info=dict(event=event,model=name,seed=seed,observed_bins=n,reconstruction_score=best_score,best_step=best_step,physics_residual_rms=physics,metrics=metrics,adam_updates=epochs,lbfgs_max_iterations=lbfgs_steps,objective_evaluations=evaluations,runtime_seconds=time.perf_counter()-started)
    stem.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=1200)
    parser.add_argument('--lbfgs-steps',type=int,default=300)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11])
    args=parser.parse_args()
    (OUT/'runs').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=dict(task='Full observed-window offline reconstruction, not forecasting',window='Same timestamp range as former fold 2; all bins are fitted',models=list(r.base.MODELS),seeds=args.seeds,adam_updates=args.epochs,lbfgs_max_iterations=args.lbfgs_steps,loss='12*(full-window RMS-normalized activity MSE + log1p activity MSE + 0.1*full-window sum-normalized cumulative MSE) + 0.2*physics + 4*history',checkpoint_selection='Minimum in-sample reconstruction score (normalized MSE + log1p MSE), same rule for all models',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device=str(device))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for event in r.base.EVENTS:
        _,frame,_,_,_=next(x for x in r.base.load_event_folds(event) if x[0]==2)
        for name in r.base.MODELS:
            for seed in args.seeds:
                path=OUT/'runs'/f'{event}_{name.lower().replace("-","_")}_seed{seed}.json'
                if path.exists():
                    continue
                fit(event,frame,name,seed,args.epochs,args.lbfgs_steps,device)

if __name__=='__main__':
    main()
