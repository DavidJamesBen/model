"""Numerical replay and local/synthetic inverse diagnostics; no test tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import revision_dde as dde

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/rigorous_revision/tables'
OUT.mkdir(parents=True,exist_ok=True)


def replay():
    manifest=json.loads((ROOT/'output/tables/baseline_run_manifest.json').read_text())['runs']
    selected={}
    for run in manifest:
        if run['model']!='TransPINN':continue
        key=(run['event'],run['fold'])
        if key not in selected or (run['validation_score'],run['seed']) < (
                selected[key]['validation_score'],selected[key]['seed']):selected[key]=run
    rows=[];predictions=[]
    for (event,fold),run in selected.items():
        frame=pd.read_csv(ROOT/f'output/predictions/{event}_fold{fold}_transpinn_seed{run["seed"]}.csv')
        times=np.arange(1,len(frame)+1)/len(frame)
        theta=np.array([run['parameters'][k] for k in dde.NAMES])
        states,flux=dde.solve(theta,times)
        xx,ff=dde.solve(theta,times,rtol=1e-10,atol=1e-12)
        train=frame.split.to_numpy()=='train';test=frame.split.to_numpy()=='test'
        _,singular=dde.sensitivity(theta,times[train])
        for j,c in enumerate(('rumor','debunk')):
            y=frame['actual_'+c].to_numpy();scale=max(y[train].sum(),1.)
            pred=flux[:,j]*scale
            denom=max(1.,np.sqrt(np.mean(y[train]**2)))
            rows.append(dict(event=event,fold=fold,channel=c,
                dde_nrmse=np.sqrt(np.mean((pred[test]-y[test])**2))/denom,
                head_nrmse=np.sqrt(np.mean((frame['predicted_'+c].to_numpy()[test]-y[test])**2))/denom,
                replay_head_gap=np.sqrt(np.mean((pred-frame['predicted_'+c].to_numpy())**2))/denom,
                state_refinement_max=np.max(np.abs(states-xx)),
                mass_error=np.max(np.abs(states[:,:7].sum(axis=1)-1)),
                min_state=states.min(),singular_max=singular[0],singular_min=singular[-1],
                condition=singular[0]/singular[-1],rank_1e6=int((singular>singular[0]*1e-6).sum())))
            predictions.append(pd.DataFrame(dict(event=event,fold=fold,channel=c,time=times,
                split=frame.split,actual=y,numerical=pred,neural=frame['predicted_'+c])))
        print('REPLAY',event,fold,'condition',singular[0]/singular[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'dde_replay_diagnostics.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'dde_replay_predictions.csv',index=False)


def synthetic():
    truth=np.array([2.5,1.3,.073,1.,1.,1.,.83,.83])
    rows=[]
    target=OUT/'synthetic_numerical_recovery.csv'
    test=np.linspace(.01,1,100)
    _,true_test=dde.solve(truth,test)
    for n in (12,48):
        times=np.linspace(1/n,1,n)
        _,exact=dde.solve(truth,times)
        for noise in (0.,.01,.05):
            for seed in (11,22,33):
                rng=np.random.default_rng(seed)
                observed=np.maximum(exact*(1+noise*rng.normal(size=exact.shape)),1e-12)
                initial=np.clip(np.log(truth)+rng.normal(0,.3,len(truth)),
                                np.log(dde.LOWER)+1e-5,np.log(dde.UPPER)-1e-5)
                def residual(logtheta):
                    _,y=dde.solve(np.exp(logtheta),times)
                    return (np.log(y)-np.log(observed)).ravel()
                result=least_squares(residual,initial,bounds=(np.log(dde.LOWER),np.log(dde.UPPER)),
                    max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8,diff_step=1e-3)
                fitted=np.exp(result.x);_,pred=dde.solve(fitted,test)
                record=dict(samples=n,noise_fraction=noise,seed=seed,nfev=result.nfev,
                    converged=bool(result.success),train_log_rmse=np.sqrt(np.mean(result.fun**2)),
                    parameter_relative_l2=np.linalg.norm(fitted-truth)/np.linalg.norm(truth),
                    prediction_relative_l2=np.linalg.norm(pred-true_test)/np.linalg.norm(true_test))
                record.update({'estimated_'+k:float(v) for k,v in zip(dde.NAMES,fitted)})
                rows.append(record);pd.DataFrame(rows).to_csv(target,index=False)
                print('SYNTHETIC',n,noise,seed,record['parameter_relative_l2'],flush=True)
    (OUT/'synthetic_protocol.json').write_text(json.dumps(dict(truth=dict(zip(dde.NAMES,truth)),
        method='bounded numerical DDE least squares, not TransPINN recovery',
        noise='independent multiplicative Gaussian, clipped at 1e-12',
        test='noise-free dense truth, same horizon, no forecasting claim',
        starts='one prespecified randomized start for each seed',max_nfev=80),indent=2),encoding='utf-8')


if __name__=='__main__':
    replay()
    synthetic()
