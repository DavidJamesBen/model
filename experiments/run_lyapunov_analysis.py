"""Conditional DDE Lyapunov audit, numerical checks and delay sensitivity."""
import lyapunov_dde as l
import numpy as np
import pandas as pd
import json,hashlib,time
from scipy.special import lambertw
import revision_dde as reference

OUT=l.ROOT/'output/lyapunov'
NAMES=reference.NAMES

def checks():
    p=np.array([2.5,1.3,.073,1.,1.,1.,.83,.83])
    rng=np.random.default_rng(41)
    x=l.HISTORY.copy();xd=x*.99
    v=rng.normal(size=9);w=rng.normal(size=9)
    z=np.r_[x,v];zd=np.r_[xd,w]
    jvp=l.augmented(z,zd,p,0)[9:]
    fd=(reference.rhs(0,x+1e-6*v,xd+1e-6*w,reference.values(p))-reference.rhs(0,x-1e-6*v,xd-1e-6*w,reference.values(p)))/2e-6
    np.testing.assert_allclose(jvp,fd,rtol=1e-7,atol=1e-8)
    expected=-.3+lambertw(.1*p[2]*np.exp(.3*p[2])).real/p[2]
    scalar=[]
    for step in [.005,.0025,.00125]:
        log,*rest=l.integrate(p,l.HISTORY,v,step,total=300,mode=1)
        measured=l.estimate(log,100,300)
        assert abs(measured-expected)<2e-7
        scalar.append(dict(requested_step=step,expected=expected,measured=measured))
    # Compare independent adaptive method of steps, at exactly matching times.
    state=[]
    for step in [.005,.0025,.00125]:
        log,final,minimum,mass,vmass,dt=l.integrate(p,l.HISTORY,v,step,total=1)
        time_end=np.ceil(1/dt)*dt
        exact=reference.solve(p,np.array([time_end]),rtol=1e-11,atol=1e-13)[0][0]
        error=float(np.max(abs(final-exact)))
        assert error<2e-7
        state.append(dict(step=dt,max_error=error))
    positive_expected=.1+lambertw(.1*p[2]*np.exp(-.1*p[2])).real/p[2]
    positive_log,*_=l.integrate(p,l.HISTORY,v,.0025,total=150,mode=2)
    positive_measured=l.estimate(positive_log,50,150)
    assert abs(positive_measured-positive_expected)<2e-7
    (OUT/'numerical_checks.json').write_text(json.dumps(dict(jvp_max_error=float(max(abs(jvp-fd))),analytic_dde=scalar,positive_exponent_check=dict(expected=positive_expected,measured=positive_measured),independent_state_solver=state),indent=2))

def cases():
    rows=pd.read_csv(l.ROOT/'output/identifiability/neural_parameter_estimates.csv')
    items=[]
    for (event,fold),d in rows.groupby(['event','fold'],sort=False):
        p=np.array([float(d.loc[d.parameter==name,'estimate'].iloc[0]) for name in NAMES])
        items.append((f'{event}_fold{fold}',event,'neural',p))
    for event in ['harvey','csdc','ottawa','sydney']:
        fit=json.loads((l.ROOT/f'output/identifiability/fit_{event}.json').read_text())
        items.append((event+'_dde_fit',event,'dde_fit',np.exp(fit['best']['z'])))
    return items

def run_case(label,event,kind,p,step,direction_seed=11,renorm=1.,history_scale=1.):
    history=l.HISTORY.copy()
    history[2:6]*=history_scale
    history[0]-=(history_scale-1)*l.HISTORY[2:6].sum()
    margin=l.compatibility(p,history)
    if margin<=0:
        return dict(case=label,event=event,kind=kind,status='incompatible_history',margin=margin)
    direction=np.random.default_rng(direction_seed).normal(size=9)
    logs,final,minimum,mass,vmass,dt=l.integrate(p,history,direction,step,total=500,renorm=renorm)
    assert minimum>=-1e-9 and mass<1e-9 and vmass<1e-9
    assert np.isfinite(logs).all()
    suffix=f'{label}_h{step}_v{direction_seed}_r{renorm}_hist{history_scale}'
    pd.DataFrame(logs,columns=['time','log_growth','rhs_max']).to_csv(OUT/'traces'/f'{suffix}.csv',index=False)
    record=dict(case=label,event=event,kind=kind,status='ok',tau=p[2],requested_step=step,actual_step=dt,direction_seed=direction_seed,renorm=renorm,history_scale=history_scale,compatibility_margin=margin,minimum_state=minimum,mass_error=mass,tangent_mass_error=vmass,final_rhs=logs[-1,2],final_active=float(final[2:6].sum()),final_susceptible=float(final[:2].sum()),trace=suffix+'.csv')
    for burn,end in [(0,50),(50,150),(100,200),(100,300),(100,500),(300,500)]:
        record[f'lambda_{burn}_{end}']=l.estimate(logs,burn,end)
    return record

def main():
    (OUT/'traces').mkdir(parents=True,exist_ok=True)
    checks()
    items=cases()
    pars=[dict(case=label,event=event,kind=kind,**dict(zip(NAMES,p))) for label,event,kind,p in items]
    pd.DataFrame(pars).to_csv(OUT/'parameter_sets.csv',index=False)
    tasks=[]
    for label,event,kind,p in items:
        for step in [.005,.0025,.00125]:
            tasks.append((label,event,kind,p,step,11,1.,1.))
        for seed in [22,33]:
            tasks.append((label,event,kind,p,.0025,seed,1.,1.))
        if kind=='neural' and 'fold2' in label:
            for renorm in [.5,2.]:
                tasks.append((label,event,kind,p,.0025,11,renorm,1.))
            for scale in [.5,1.5]:
                tasks.append((label,event,kind,p,.0025,11,1.,scale))
            for tau in [.01,.05,.1,.15,.22]:
                pt=p.copy();pt[2]=tau
                tasks.append((label+f'_tau{tau}',event,'delay_sweep',pt,.0025,11,1.,1.))
    rows=[]
    for i,task in enumerate(tasks):
        rows.append(run_case(*task))
        pd.DataFrame(rows).to_csv(OUT/'lyapunov_runs.csv',index=False)
        if (i+1)%10==0:
            print(f'{i+1}/{len(tasks)} completed',flush=True)
    sources=[l.ROOT/'experiments/lyapunov_dde.py',l.ROOT/'experiments/identifiability_solver.py',l.ROOT/'experiments/revision_dde.py',l.ROOT/'experiments/run_lyapunov_analysis.py',l.ROOT/'output/identifiability/neural_parameter_estimates.csv']+[l.ROOT/f'output/identifiability/fit_{e}.json' for e in ['harvey','csdc','ottawa','sydney']]
    manifest=dict(scope='Conditional numerical dynamics of the normalized closed empirical DDE, not of all open-population parameter regimes or measured events',runs=len(tasks),baseline_parameter_sets=len(items),method='RK4 method of steps with linear delayed interpolation; complex-step variational JVP; full history L2 normalization; sum-zero human perturbations enforced',duration=500,burn_in=100,units='inverse normalized time',source_sha256={str(p.relative_to(l.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).groupby('kind')['lambda_100_500'].agg(['min','max']).to_string(),flush=True)

if __name__=='__main__':
    main()
