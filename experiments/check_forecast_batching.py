"""Verify that batching preserves predictions, DDE residuals and gradients."""
import improve_selected_forecasts as refit
import torch

def main():
    for name in refit.base.MODELS:
        refit.base.seed_everything(11)
        model=refit.base.RevisedRumorPINN(name,16).double()
        times=torch.linspace(.02,.7,8,dtype=torch.float64)[:,None]
        collocation=torch.linspace(0,1,12,dtype=torch.float64)[:,None]
        separate=(model.observation(times),
                  refit.base.residual_matrix(model,collocation,.0025),
                  model(torch.zeros(1,1,dtype=torch.float64)))
        batched=refit.batched_training_outputs(model,times,collocation,.0025)
        for a,b in zip(separate,batched):
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
        gradients=[]
        for outputs in (separate,batched):
            gradients.append(torch.autograd.grad(
                sum(x.square().mean() for x in outputs),list(model.parameters()),allow_unused=True))
        for a,b in zip(*gradients):
            assert (a is None)==(b is None)
            if a is not None:
                torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-8)
        print(name,'outputs and parameter gradients agree',flush=True)

if __name__=='__main__':
    main()
