"""Execute the frozen metric-aligned, noise/sparsity and synthetic protocol.

Independent models are vectorized, with separate Adam moments, clipping norms,
validation choices and checkpoints. A sum over per-model losses preserves each
individual model's gradient. No gradients or observations cross model replicas.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import run_multievent_baselines as base
from robustness_models import make_model,stack_models,batched_forward

ROOT=Path(__file__).resolve().parents[1]
# The manuscript's Windows project path is long. Extended paths avoid the
# legacy 260-character limit without changing system-wide registry settings.
if os.name=='nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT=Path('\\\\?\\'+str(ROOT))
OUT=ROOT/'output/robustness'
MODELS=('TransPINN','Matched-PINN','NoAttn-PINN','Data-Transformer','Data-MLP')
HISTORY=[.93,.03,.015,.005,.010,.005,.005,.20,.20,0.,0.]

def stable_seed(*items):
    return int(hashlib.sha256('|'.join(map(str,items)).encode()).hexdigest()[:8],16)

def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(data),indent=2,allow_nan=False),encoding='utf-8')

@dataclass
class Case:
    event:str
    fold:int
    seed:int
    budget:int
    retention:int
    noise:int
    raw:np.ndarray
    labels:np.ndarray
    observed:np.ndarray
    scales:np.ndarray
    denominator:np.ndarray
    split:tuple[int,int,int]
    times:np.ndarray
    timestamp:np.ndarray
    quadrature:np.ndarray
    mask_seed:int
    noise_seed:int
    input_sha256:str
    family:str='empirical'
    reporting_bounds:np.ndarray|None=None
    synthetic_meta:dict|None=None

    def stem(self,model):
        return OUT/self.family/'runs'/f'{self.event}_fold{self.fold}_b{self.budget}_r{self.retention}_n{self.noise}_{model}_seed{self.seed}'

def empirical_case(event,fold,frame,split,seed,budget,retention,noise):
    train,val,n=map(int,split)
    raw=frame[['rumor_activity','debunk_activity']].to_numpy(float)
    mask_seed=stable_seed('mask-v1',event,fold,seed)
    noise_seed=stable_seed('noise-v1',event,fold,seed)
    order=np.random.default_rng(mask_seed).permutation(train)
    observed=np.zeros(n,dtype=bool)
    observed[order[:max(1,int(np.ceil(train*retention/100)))]]=True
    kept=raw[observed]
    denominator=np.maximum(np.sqrt(np.mean(kept**2,axis=0)),1.)
    # Estimate the full training total using observed entries only (SRS mask).
    scales=np.maximum(kept.sum(0)*train/len(kept),1.)
    z=np.random.default_rng(noise_seed).standard_normal((train,2))
    labels=np.full_like(raw,np.nan)
    perturbed=np.maximum(0.,raw[:train]+noise/100*denominator*z)
    labels[observed]=perturbed[observed[:train]]
    digest=hashlib.sha256()
    for a in (raw,labels,observed,scales,denominator,np.asarray(split)):
        digest.update(np.ascontiguousarray(a).tobytes())
    return Case(event,fold,seed,budget,retention,noise,raw,labels,observed,scales,
                denominator,(train,val,n),(np.arange(1,n+1)/n)[:,None],
                frame.timestamp.astype(str).to_numpy(),np.ones(1),mask_seed,noise_seed,digest.hexdigest())

def synthetic_case(path,seed):
    data=np.load(path,allow_pickle=False)
    raw=data['y']; n=len(raw); train,val=100,120
    scales=np.maximum(raw[:train].sum(0),1.)
    denominator=np.maximum(np.sqrt(np.mean(raw[:train]**2,axis=0)),1.)
    observed=np.arange(n)<train
    labels=np.where(observed[:,None],raw,np.nan)
    nodes,weights=np.polynomial.legendre.leggauss(3)
    edges=data['bin_edges']
    times=(edges[:-1,None]+(nodes[None,:]+1)/2*np.diff(edges)[:,None]).reshape(-1,1)
    weights=weights/2
    factor=10000*np.diff(edges)[0]/scales
    bounds=factor[:,None]*np.array([.25,12.])[None,:]
    meta={k:np.asarray(data[k]).tolist() for k in ('theta','states','history','media_decay','generator_seed')}
    meta.update(dataset=str(path.relative_to(ROOT)),gain_raw_to_normalized=factor.tolist())
    return Case(path.stem,1,seed,1040,100,0,raw,labels,observed,scales,denominator,
                (train,val,n),times,np.arange(n),weights,0,0,
                hashlib.sha256(path.read_bytes()).hexdigest(),'synthetic_neural',bounds,meta)

def rhs(state,delayed,theta):
    su,se,e,h,r,d,u,m1,m2=state[:,:,:9].unbind(-1)
    ds,des,_,_,dr,dd,_,dm1,dm2=delayed[:,:,:9].unbind(-1)
    b1,b2,tau,qr,qd,chi,_,_=theta[:,None,:].unbind(-1)
    inc=(b1*r+b2*d)*(su+.55*se)
    past=(b1*dr+b2*dd)*(ds+.55*des)
    gate=5*dm1*dm2/(1+5*dm1*dm2)
    completed=gate*past*torch.exp(-1.4*tau)
    load=chi+m1+m2
    gr=qr*m2*chi/((1+m1)*load)
    gd=qd*m1*chi/((1+m2)*load)
    gu=.3*(m1+m2)/load
    return torch.stack((-.4*m1*su+.25*m2*se-(b1*r+b2*d)*su,
        .4*m1*su-.25*m2*se-.55*(b1*r+b2*d)*se,
        inc-1.4*e-completed,completed-(gr+gd+gu)*h,
        .8*e+gr*h-(.25+.5*m1)*r+.25*m2*d,
        .4*e+gd*h+.5*m1*r-(.25+.25*m2)*d,
        .2*e+.25*r+.25*d+gu*h,.6*d+.2-m1,.6*r+.2-m2,
        .8*e+gr*h+.25*m2*d,.4*e+gd*h+.5*m1*r),-1)

def observations(state,theta,physical,n,quadrature):
    flux=state[:,:,9:11]*theta[:,None,6:8] if physical else state
    return (flux.reshape(len(flux),n,len(quadrature),2)*quadrature[None,None,:,None]).sum(2)

def train_batch(cases,model_name,device,batch_id):
    b=len(cases); first=cases[0]; train,val,n=first.split; epochs=first.budget
    assert all(c.split==first.split and c.budget==epochs and len(c.quadrature)==len(first.quadrature) for c in cases)
    models=[]
    for case in cases:
        base.seed_everything(case.seed)
        models.append(make_model(model_name,case.reporting_bounds))
    template,params,buffers=stack_models(models,device)
    physical=template.physical
    assert sum(v.numel()//b for v in params.values())==18619
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=device)
    times=tensor(np.stack([c.times for c in cases]))
    q=tensor(first.quadrature); qn=len(q)
    scales=tensor(np.stack([c.scales for c in cases]))
    denom=tensor(np.stack([c.denominator for c in cases]))
    raw=tensor(np.stack([c.raw for c in cases]))
    observed=tensor(np.stack([c.observed[:train] for c in cases]))
    labels=tensor(np.stack([np.nan_to_num(c.labels[:train]) for c in cases]))
    colloc=torch.linspace(0,1,96,device=device)[None,:,None].expand(b,-1,-1)
    plus=(colloc+.0025).clamp(0,1);minus=(colloc-.0025).clamp(0,1)
    hist=tensor(HISTORY)[None,None,:]
    optim=torch.optim.Adam(list(params.values()),lr=.002,foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs,eta_min=.0001)
    best_score=torch.full((b,),float('inf'),device=device)
    best_epoch=torch.zeros(b,dtype=torch.int64,device=device)
    best={name:p.detach().clone() for name,p in params.items()}
    trace=[]
    if device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for epoch in range(1,epochs+1):
        optim.zero_grad(set_to_none=True)
        if physical:
            tau=.01+.21*torch.sigmoid(params['core.tau.raw']).reshape(b,1,1)
            delayed=colloc-tau
            all_times=torch.cat([times[:,:train*qn],colloc,plus,minus,delayed.clamp(0,1),times.new_zeros(b,1,1)],1)
            all_states,theta=batched_forward(template,params,buffers,all_times)
            pred_norm=observations(all_states[:,:train*qn],theta,True,train,q)
            center,sp,sm,sd,sh=torch.split(all_states[:,train*qn:],(96,96,96,96,1),dim=1)
            sd=torch.where(delayed>=0,sd,hist.expand(b,96,-1))
            deriv=(sp[:,:,:9]-sm[:,:,:9])/(plus-minus)
            expected=rhs(center,sd,theta)
            residual=torch.cat([deriv-expected[:,:,:9],center[:,:,9:11]-expected[:,:,9:11]],-1)
            physics_loss=residual.square().mean((1,2))
            history_loss=(sh[:,:,:9]-hist[:,:,:9]).square().mean((1,2))
        else:
            state,theta=batched_forward(template,params,buffers,times[:,:train*qn])
            pred_norm=observations(state,theta,False,train,q)
            physics_loss=history_loss=best_score.new_zeros(b)
        error=(pred_norm*scales[:,None,:]-labels)/denom[:,None,:]
        data_loss=(error.square()*observed[:,:,None]).sum((1,2))/(2*observed.sum(1))
        loss=12*data_loss+.2*min(1.,epoch/120)*physics_loss+4*history_loss
        loss.sum().backward()
        # Separate clipping per independent model; global clipping would couple fits.
        norms=torch.stack([p.grad.reshape(b,-1).square().sum(1) for p in params.values() if p.grad is not None]).sum(0).sqrt()
        multiplier=(5/(norms+1e-6)).clamp(max=1)
        for p in params.values():
            if p.grad is None:raise AssertionError('Inactive trainable parameter')
            p.grad.mul_(multiplier.reshape((b,)+(1,)*(p.ndim-1)))
        optim.step();scheduler.step()
        if epoch%10==0:
            with torch.no_grad():
                state,theta=batched_forward(template,params,buffers,times[:,train*qn:val*qn])
                vp=observations(state,theta,physical,val-train,q)*scales[:,None,:]
                score=(((vp-raw[:,train:val])/denom[:,None,:]).square().mean(1).sqrt()).mean(1)
                if not torch.isfinite(score).all() or not torch.isfinite(loss).all():raise FloatingPointError(f'{batch_id} epoch{epoch}')
                improved=score<best_score-1e-8
                for name,p in params.items():
                    best[name]=torch.where(improved.reshape((b,)+(1,)*(p.ndim-1)),p.detach(),best[name])
                best_epoch=torch.where(improved,epoch,best_epoch)
                best_score=torch.where(improved,score,best_score)
                trace.append(torch.stack([score,data_loss,physics_loss,history_loss],1).cpu().numpy())
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    with torch.no_grad():
        state,theta=batched_forward(template,best,buffers,times)
        pred=observations(state,theta,physical,n,q).cpu().numpy()
        coefficients=theta.cpu().numpy()
        if physical and first.family=='synthetic_neural':
            edge_times=tensor(np.stack([np.linspace(0,1,n+1)[:,None]]*b))
            edge_state,_=batched_forward(template,best,buffers,edge_times)
            latent=edge_state[:,:,:9].cpu().numpy()
        else:latent=None
    best_cpu={k:v.detach().cpu() for k,v in best.items()}
    buffer_cpu={k:v.detach().cpu() for k,v in buffers.items()}
    traces=np.stack(trace)
    for i,case in enumerate(cases):
        stem=case.stem(model_name);stem.parent.mkdir(parents=True,exist_ok=True)
        prediction=pred[i]*case.scales
        frame=pd.DataFrame({'timestamp':case.timestamp,'split':np.where(np.arange(n)<train,'train',np.where(np.arange(n)<val,'validation','test')),
            'actual_rumor':case.raw[:,0],'actual_debunk':case.raw[:,1],
            'predicted_rumor':prediction[:,0],'predicted_debunk':prediction[:,1],
            'train_observed':case.observed,'training_label_rumor':case.labels[:,0],
            'training_label_debunk':case.labels[:,1]})
        for j,ch in enumerate(('rumor','debunk')):
            frame[f'actual_cumulative_{ch}']=case.raw[:,j].cumsum()
            frame[f'predicted_cumulative_{ch}']=prediction[:,j].cumsum()
        frame.to_csv(stem.with_suffix('.csv'),index=False)
        state_dict={k:v[i].clone() for k,v in best_cpu.items()}
        state_dict.update({k:v[i].clone() for k,v in buffer_cpu.items()})
        torch.save(state_dict,stem.with_suffix('.pt'))
        pd.DataFrame(traces[:,i,:],columns=['validation_nrmse','train_data_loss','physics_loss','history_loss'],index=np.arange(10,epochs+1,10)).rename_axis('epoch').to_csv(stem.with_suffix('.trace.csv'))
        test_error=(prediction[val:]-case.raw[val:])/case.denominator
        run={'event':case.event,'fold':case.fold,'model':model_name,'seed':case.seed,
             'epochs_completed':epochs,'parameter_count':18619,'best_epoch':int(best_epoch[i]),
             'validation_score':float(best_score[i]),'runtime_seconds':elapsed/b,
             'runtime_kind':'amortized_gpu_batch' if device.type=='cuda' else 'amortized_cpu_batch',
             'batch_id':batch_id,'batch_size':b,'batch_runtime_seconds':elapsed,
             'test_nrmse':float(np.sqrt((test_error**2).mean(0)).mean()),
             'test_nmae':float(np.abs(test_error).mean()),
             'parameters':coefficients[i].tolist() if physical else None}
        payload={'run':run,'split':case.split,'metric_denominator':case.denominator,'scales':case.scales,
                 'retention':case.retention/100,'noise':case.noise/100,'budget':epochs,
                 'condition':f'r{case.retention}_n{case.noise}','mask_seed':case.mask_seed,'noise_seed':case.noise_seed,
                 'input_sha256':case.input_sha256,'quadrature':case.quadrature,'reporting_bounds':case.reporting_bounds}
        if case.synthetic_meta is not None:
            payload['synthetic']=case.synthetic_meta
            payload['horizons']={}
            for horizon in (20,40,80):
                er=(prediction[val:val+horizon]-case.raw[val:val+horizon])/case.denominator
                payload['horizons'][str(horizon)]={'nrmse':float(np.sqrt((er**2).mean(0)).mean()),'nmae':float(np.abs(er).mean())}
            if physical:
                theta_raw=coefficients[i].copy();theta_raw[6:8]/=np.asarray(case.synthetic_meta['gain_raw_to_normalized'])
                truth=np.asarray(case.synthetic_meta['theta'])
                payload['parameter_relative_l2']=float(np.linalg.norm((theta_raw-truth)/truth)/np.sqrt(8))
                payload['parameter_log_rmse']=float(np.sqrt(np.mean(np.log(theta_raw/truth)**2)))
                payload['parameter_mean_relative_error']=float(np.mean(np.abs(theta_raw/truth-1)))
                payload['parameters_raw']=theta_raw.tolist()
                true_states=np.asarray(case.synthetic_meta['states'])
                state_scale=np.maximum(np.sqrt(np.mean(true_states[1:train+1]**2,axis=0)),1e-8)
                payload['latent_relative_l2']=float(np.linalg.norm(latent[i][val+1:]-true_states[val+1:])/np.linalg.norm(true_states[val+1:]))
                payload['recovery_metric_definitions']={
                    'parameter_relative_l2':'RMS of eight individual relative parameter errors',
                    'parameter_log_rmse':'RMS of eight log(estimate/truth) parameter errors',
                    'parameter_mean_relative_error':'Mean absolute relative error across eight parameters',
                    'latent_relative_l2':'Pooled relative L2 across all nine states at the 80 test-bin right endpoints',
                    'state_nrmse':'Mean of nine state RMSEs normalized separately by true training-state RMS',
                    'latent_E_H_U_M1_M2_nrmse':'Same separately normalized metric restricted to E, H, U, M1, M2'}
                for horizon in (20,40,80):
                    state_difference=latent[i][val+1:val+horizon+1]-true_states[val+1:val+horizon+1]
                    state_errors=np.sqrt(np.mean(state_difference**2,axis=0))/state_scale
                    payload['horizons'][str(horizon)]['state_nrmse']=float(np.mean(state_errors))
                    payload['horizons'][str(horizon)]['latent_E_H_U_M1_M2_nrmse']=float(np.mean(state_errors[[2,3,6,7,8]]))
                np.savez_compressed(stem.with_suffix('.latent.npz'),predicted=latent[i],actual=true_states)
        dump(stem.with_suffix('.json'),payload)
    print(f'{batch_id}: {b} fits, {elapsed:.1f}s, {epochs} updates, GPU allocated {torch.cuda.max_memory_allocated()/1024**3:.2f}GB' if device.type=='cuda' else f'{batch_id}: {b} fits {elapsed:.1f}s',flush=True)
    return elapsed

def freeze_manifest(device,args):
    OUT.mkdir(exist_ok=True,parents=True)
    paths=[ROOT/'experiments/run_robustness.py',ROOT/'experiments/robustness_models.py',ROOT/'experiments/run_fair_benchmark.py',ROOT/'experiments/run_multievent_baselines.py']
    info={'protocol':'metric-aligned-v1','torch':torch.__version__,'python':platform.python_version(),
          'device':str(device),'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None,
          'maximum_independent_batch_size':args.batch_size,
          'models':MODELS,'parameters':18619,'stage1_seeds':[11,22,33],
          'stage2_seeds':list(range(11,111,11)),'retentions':[100,50,25],'noises':[0,5,10],
          'empirical_unique_fits':5580,'synthetic_neural_fits':390,
          'optimizer':'Adam 0.002 cosine->0.0001, per-model clip5','data_weight':12,
          'physics_weight':'0.2*min(1,epoch/120)','history_weight':4,'collocation_points':96,
          'normalization':'retained clean training RMS; total estimated as n_train/n_retained * retained clean sum',
          'noise':'clipped additive Gaussian; clean training normalization fixed before corruption',
          'validation':'macro incremental NRMSE, checked every10updates; no early stopping',
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          'protocol_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (ROOT/'output/benchmark_design/experiment_protocol.md',OUT/'execution_addendum.md')},
          'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.DATA_DIR.glob('*.csv'))}}
    manifest=OUT/'neural_protocol.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing!=clean_json(info):raise ValueError('Frozen protocol differs. Use explicit new output root/version after documenting change.')
    else:dump(manifest,info)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['stage1','stage2','synthetic','all'],default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--limit-batches',type=int,default=0)
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    freeze_manifest(device,args)
    completed=0
    stages=('stage1','stage2','synthetic') if args.stage=='all' else (args.stage,)
    for stage in stages:
        groups=[]
        if stage in ('stage1','stage2'):
            for event in base.EVENTS:
                for fold,frame,split,_,_ in base.load_event_folds(event):
                    for budget in ((260,1040) if stage=='stage1' else (1040,)):
                        cases=[]
                        for retention in ((100,) if stage=='stage1' else (100,50,25)):
                            for noise in ((0,) if stage=='stage1' else (0,5,10)):
                                for seed in ((11,22,33) if stage=='stage1' else range(11,111,11)):
                                    cases.append(empirical_case(event,fold,frame,split,seed,budget,retention,noise))
                        groups.append((f'{stage}_{event}_fold{fold}_b{budget}',cases))
        else:
            paths=sorted((OUT/'synthetic/datasets').glob('*.npz'))
            if len(paths)!=26:raise ValueError(f'Expected26syntheticdatasets,found{len(paths)}')
            groups=[('synthetic',[synthetic_case(p,s) for p in paths for s in (11,22,33)])]
        for name,cases in groups:
            for model in ((args.model,) if args.model else MODELS):
                todo=[c for c in cases if not c.stem(model).with_suffix('.json').exists()]
                for start in range(0,len(todo),args.batch_size):
                    train_batch(todo[start:start+args.batch_size],model,device,f'{name}_{model}_batch{start}')
                    completed+=1
                    if args.limit_batches and completed>=args.limit_batches:return
    print('All requested stages complete.',flush=True)

if __name__=='__main__':
    main()
