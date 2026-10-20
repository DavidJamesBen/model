"""Refit the displayed fold with balanced activity losses; never select on test.

Original runs remain immutable. This is a new optimization experiment, not a
replacement for the manuscript's 260-update, three-fold benchmark.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/python-deps'))
sys.path.insert(0, str(ROOT / 'tmp/gpu-deps'))
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import run_multievent_baselines as base

OUT = ROOT / 'output/forecast_refit'
CHANNELS = ('rumor', 'debunk')

def score(y, p, rms):
    return float(np.mean(((p-y)/rms)**2) + np.mean((np.log1p(p)-np.log1p(y))**2))

def fit(event, frame, split, name, seed, epochs, device):
    base.seed_everything(seed)
    tr, va, n = map(int, split)
    # Only the training prefix participates in gradient computation.
    raw = frame[['rumor_activity','debunk_activity']].to_numpy(np.float32)
    scale = np.maximum(raw[:tr].sum(0), 1)
    rms = np.maximum(np.sqrt(np.mean(raw[:tr]**2, axis=0)), 1)
    y = torch.tensor(raw[:tr], device=device)
    scales = torch.tensor(scale, device=device)
    normalizer = torch.tensor(rms, device=device)
    times = (torch.arange(1,n+1,device=device,dtype=torch.float32)/n)[:,None]
    cfg = base.Config(epochs=epochs)
    model = base.RevisedRumorPINN(name,cfg.sequence_length).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=0.0001)
    collocation = torch.linspace(0,1,cfg.collocation_points,device=device)[:,None]
    history = base.initial_history(device)
    best = float('inf')
    best_state = None
    log = []
    started = time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model.observation(times[:tr])*scales
        activity_loss = ((pred-y)/normalizer).square().mean()
        log_loss = (torch.log1p(pred)-torch.log1p(y)).square().mean()
        cumulative_loss = ((pred.cumsum(0)-y.cumsum(0))/scales).square().mean()
        residual = base.residual_matrix(model,collocation,cfg.finite_difference_step)
        physical = base.causal_physics_loss(residual,cfg) if name=='TimePINN' else residual.square().mean()
        hist = F.mse_loss(model(torch.zeros((1,1),device=device))[:,:9],history[:,:9])
        loss = cfg.data_weight*(activity_loss+log_loss+0.1*cumulative_loss) + cfg.physics_weight*min(1,epoch/120)*physical + cfg.history_weight*hist
        if not torch.isfinite(loss):
            raise FloatingPointError((event,name,seed,epoch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip)
        opt.step()
        scheduler.step()
        if epoch%10==0:
            model.eval()
            with torch.no_grad():
                vp = (model.observation(times[tr:va])*scales).cpu().numpy()
            val = score(raw[tr:va],vp,rms)
            log.append(dict(epoch=epoch,validation_score=val,training_loss=float(loss.detach())))
            if val < best:
                best = val
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = (model.observation(times)*scales).cpu().numpy()
        physics_rms = float(base.residual_matrix(model,collocation,cfg.finite_difference_step).square().mean().sqrt())
    stem = OUT/'runs'/f'{event}_fold2_{name.lower().replace("-","_")}_seed{seed}'
    torch.save({k:v.cpu() for k,v in best_state.items()},stem.with_suffix('.pt'))
    result = pd.DataFrame({'timestamp':frame.timestamp,'split':['train']*tr+['validation']*(va-tr)+['test']*(n-va)})
    for c,ch in enumerate(CHANNELS):
        result['actual_'+ch] = raw[:,c]
        result['predicted_'+ch] = predictions[:,c]
    result.to_csv(stem.with_suffix('.csv'),index=False)
    info = dict(event=event,fold=2,model=name,seed=seed,best_epoch=best_epoch,validation_score=best,physics_residual_rms=physics_rms,runtime_seconds=time.perf_counter()-started)
    stem.with_suffix('.json').write_text(json.dumps(info,indent=2))
    pd.DataFrame(log).to_csv(stem.with_suffix('.history.csv'),index=False)
    return info

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=800)
    parser.add_argument('--events',nargs='+',default=list(base.EVENTS))
    parser.add_argument('--models',nargs='+',default=list(base.MODELS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    parser.add_argument('--manifest-name',default='manifest.json')
    args=parser.parse_args()
    (OUT/'runs').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=dict(epochs=args.epochs,seeds=args.seeds,models=args.models,events=args.events,fold=2,device=str(device),torch=torch.__version__,loss='12*(train_rms_normalized_activity_MSE + log1p_activity_MSE + 0.1*train_sum_normalized_cumulative_MSE) + original physics/history penalties',selection='validation normalized activity MSE + log1p MSE; shared rule for all methods; original seeds included as fallback candidates',test_use='final evaluation only',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/args.manifest_name).write_text(json.dumps(manifest,indent=2))
    for event in args.events:
        _,frame,split,_,_=next(f for f in base.load_event_folds(event) if f[0]==2)
        for name in args.models:
            for seed in args.seeds:
                path=OUT/'runs'/f'{event}_fold2_{name.lower().replace("-","_")}_seed{seed}.json'
                if path.exists():
                    continue
                print(json.dumps(fit(event,frame,split,name,seed,args.epochs,device)),flush=True)

if __name__=='__main__':
    main()
