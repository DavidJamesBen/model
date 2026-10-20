"""Strict 18,619-parameter, 260-update comparison, including a data-only Transformer.

Run from the repository root: python experiments/run_fair_benchmark.py
All four methods are freshly timed on the same CPU. No inactive padding weights.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_multievent_baselines as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/fair_benchmark"
MODELS = ("TransPINN", "Matched-PINN", "NoAttn-PINN", "Data-Transformer")
ORIGINAL = base.RevisedRumorPINN
CAPTURED = None

class MatchedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9,132), nn.Tanh(), nn.Linear(132,120),
                                 nn.Tanh(), nn.Linear(120,11))
        with torch.no_grad():
            self.net[-1].bias[-2:].fill_(-4.5)
    def forward(self,t):
        return self.net(base.time_features(t))

class DataTransformer(nn.Module):
    data_only = True
    def __init__(self, length):
        super().__init__()
        self.backbone = base.PINNsFormerBackbone(length, True)
        # Counts are matched through ordinary dense-layer widths, not unused weights.
        self.backbone.head = nn.Sequential(nn.Linear(24,25),base.WaveAct(),
                                          nn.Linear(25,124),base.WaveAct(),nn.Linear(124,2))
        with torch.no_grad():
            self.backbone.head[-1].bias.fill_(-4.5)
    def observation(self,t):
        return F.softplus(self.backbone(t))
    def parameter_values(self):
        return {}

def factory(model_name,sequence_length,fixed_tau=None):
    global CAPTURED
    if model_name == "Data-Transformer":
        model = DataTransformer(sequence_length)
    else:
        model = ORIGINAL("PINN" if model_name == "Matched-PINN" else model_name,
                         sequence_length, fixed_tau)
        if model_name == "Matched-PINN":
            model.backbone = MatchedMLP()
    assert sum(p.numel() for p in model.parameters()) == 18619
    CAPTURED = model
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"runs").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    base.RevisedRumorPINN=factory
    config=base.Config(epochs=260,min_epochs=260,patience_checks=1000)
    manifest={"models":MODELS,"seeds":[11,22,33],"config":asdict(config),
              "parameter_count":18619,"device":"CPU","threads":4,
              "torch":torch.__version__,"python":platform.python_version(),
              "processor":platform.processor(),"selection":"validation only",
              "data_only":"No DDE, history loss, latent compartments or structural coefficients",
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__),Path(base.__file__)]}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    done=0
    for event in base.EVENTS:
        for fold,frame,split,core,horizon in base.load_event_folds(event):
            for seed in (11,22,33):
                # Rotate execution order to spread warm-up/thermal effects across methods.
                offset=((seed//11-1)+fold-1)%len(MODELS)
                for model_name in MODELS[offset:]+MODELS[:offset]:
                    stem=OUT/"runs"/f"{event}_fold{fold}_{model_name}_seed{seed}"
                    if stem.with_suffix(".json").exists():
                        continue
                    run_config=config
                    if model_name=="Data-Transformer":
                        from dataclasses import replace
                        run_config=replace(config,physics_weight=0.,history_weight=0.)
                    rows,pred,info=base.train_one(event,fold,frame,split,model_name,seed,
                                                run_config,torch.device("cpu"))
                    assert info["epochs_completed"]==260
                    # Every parameter must be connected to the optimized objective.
                    missing=[name for name,p in CAPTURED.named_parameters() if p.grad is None]
                    assert not missing, missing
                    pred.to_csv(stem.with_suffix(".csv"),index=False)
                    torch.save(CAPTURED.state_dict(),stem.with_suffix(".pt"))
                    payload={"run":info,"rows":rows,"config":asdict(run_config),
                             "split":split.tolist(),"gradient_connected":True}
                    stem.with_suffix(".json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
                    print(f"{event} f{fold} {model_name} s{seed}: NRMSE={np.mean([r['nrmse'] for r in rows]):.6f}, {info['runtime_seconds']:.1f}s",flush=True)
                    done+=1
                    if args.limit and done>=args.limit:
                        return
    print("Complete: 144 fresh fits.",flush=True)

if __name__=="__main__":
    main()
