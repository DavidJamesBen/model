"""Paired retraining of full TransPINN and its tau=0 ablation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/python-deps'))
import json, hashlib, time
import torch
import run_multievent_baselines as base

OUT=ROOT/'output/delay_ablation'
Original=base.RevisedRumorPINN
captured=None
class Capture(Original):
    def __init__(self,*args,**kwargs):
        global captured
        super().__init__(*args,**kwargs)
        captured=self

def check():
    base.seed_everything(11)
    m=Original('TransPINN',16,fixed_tau=0.)
    t=torch.tensor([[.1],[.4],[.8]])
    x=m(t)
    rhs=base.dde_rhs(m,t,x)
    inc=(m.beta_1()*x[:,4]+m.beta_2()*x[:,5])*(x[:,0]+.55*x[:,1])
    gate=5*x[:,7]*x[:,8]/(1+5*x[:,7]*x[:,8])
    expected=(1-gate)*inc-1.4*x[:,2]
    torch.testing.assert_close(rhs[:,2],expected)
    torch.testing.assert_close(rhs[:,:7].sum(1),torch.zeros(3),atol=1e-6,rtol=0)
    assert m.tau is None and m.delay().item()==0
    a=Original('TransPINN',16)
    assert sum(p.numel() for p in a.parameters())-sum(p.numel() for p in m.parameters())==1
    return {'zero_delay_exposed_equation':'passed','human_mass_conservation':'passed','removed_trainable_scalars':1}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    (OUT/'checks.json').write_text(json.dumps(check(),indent=2))
    base.RevisedRumorPINN=Capture
    cfg=base.Config()
    for event in base.EVENTS:
        for fold,frame,split,_,_ in base.load_event_folds(event):
            for seed in (11,22,33):
                for variant,tau in [('full',None),('tau_zero',0.)]:
                    stem=f'{event}_fold{fold}_{variant}_seed{seed}'
                    dest=OUT/(stem+'.json')
                    if dest.exists(): continue
                    print(time.strftime('%H:%M:%S'),'START',stem,flush=True)
                    rows,pred,info=base.train_one(event,fold,frame,split,'TransPINN',seed,cfg,torch.device('cpu'),fixed_tau=tau)
                    pred.to_csv(OUT/(stem+'.csv'),index=False)
                    torch.save(captured.state_dict(),OUT/(stem+'.pt'))
                    dest.write_text(json.dumps({'variant':variant,'config':base.asdict(cfg),'run':info,'rows':rows},indent=2))
                    print('DONE',stem,'seconds',round(info['runtime_seconds'],2),flush=True)
    source=[Path(__file__),Path(base.__file__)]
    (OUT/'manifest.json').write_text(json.dumps({'fits':72,'seeds':[11,22,33],'device':'cpu','threads':4,'torch':torch.__version__,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},'selection':'validation only','intervention':'tau=0, exp(-a_E*tau)=1, retain instantaneous cohort transfer'},indent=2))

if __name__=='__main__': main()
