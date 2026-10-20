"""Structural symmetry certificate, DDE sensitivities, and profile likelihoods.

This is a retrospective audit of the inverse problem, not a replacement for
archived neural forecasts. Profiles use only training observations of the
prespecified middle fold of each event. All eight parameters are profiled.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares, brentq
from scipy.stats import chi2
import revision_dde as reference
from identifiability_solver import solve, HISTORY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/identifiability'
NAMES = reference.NAMES
LOWER, UPPER = reference.LOWER, reference.UPPER
LO, HI = np.log(LOWER), np.log(UPPER)
EVENTS = ('harvey','csdc','ottawa','sydney')
MESH = 4000


def structural_certificate():
    su,se,e,h,r,d,u,m1,m2 = sp.symbols('Su Se E H R D U M1 M2', positive=True)
    b1,b2,eps,k1,k2,xi,sg,ga,mu,al,de1,de2,rho,qr,qd,qu,chi,w1,w2,eta1,eta2,v1,v2,dm1,dm2,dh,lam,rec,c = sp.symbols(
        'b1 b2 eps k1 k2 xi sg ga mu al de1 de2 rho qr qd qu chi w1 w2 eta1 eta2 v1 v2 dm1 dm2 dh lam rec c', positive=True)
    inc=(b1*r+b2*d)*(su+eps*se)/(su+se+e+h+r+d+u)
    gr=qr*m2*chi/((1+w1*m1)*(chi+m1+m2))
    gd=qd*m1*chi/((1+w2*m2)*(chi+m1+m2))
    gu=qu*(m1+m2)/(chi+m1+m2)
    gate=rho*m1*m2/(1+rho*m1*m2)
    cohort=sp.Symbol('Q_tau',positive=True)  # unchanged delayed gate/incidence/survival
    rhs=[(1-lam)*rec-dh*su-k1*m1*su+k2*m2*se-(b1*r+b2*d)*su/(su+se+e+h+r+d+u),
        lam*rec-dh*se+k1*m1*su-k2*m2*se-eps*(b1*r+b2*d)*se/(su+se+e+h+r+d+u),
        inc-(xi+sg+ga+dh)*e-cohort,cohort-(gr+gd+gu+dh)*h,
        xi*e+gr*h-(mu+dh+de1*m1)*r+de2*m2*d,
        sg*e+gd*h+de1*m1*r-(al+dh+de2*m2)*d,
        ga*e+mu*r+al*d+gu*h-dh*u,eta1*d+v1-dm1*m1,eta2*r+v2-dm2*m2]
    transform={m1:c*m1,m2:c*m2,chi:c*chi,rho:rho/c**2,w1:w1/c,w2:w2/c,
        k1:k1/c,k2:k2/c,de1:de1/c,de2:de2/c,qr:qr/c,qd:qd/c,
        eta1:c*eta1,eta2:c*eta2,v1:c*v1,v2:c*v2}
    expressions=rhs+[gr,gd,gu,gate,inc,xi*e+gr*h+de2*m2*d,sg*e+gd*h+de1*m1*r]
    residuals=[sp.simplify(expr.subs(transform,simultaneous=True)-(c if i in (7,8) else 1)*expr)
               for i,expr in enumerate(expressions)]
    assert all(x==0 for x in residuals)
    record={'verified_identities':len(residuals),'all_residuals_zero':True,
        'transformation':{str(k):str(v) for k,v in transform.items()},
        'history':'Media history is multiplied by c; human history, tau, and reporting scales are unchanged.',
        'conclusion':'Unrestricted model is structurally nonidentifiable. Fixing the media scale removes this symmetry but does not prove global identifiability.'}
    (OUT/'structural_certificate.json').write_text(json.dumps(record,indent=2))


def load(event,fold=2):
    selection=pd.read_csv(ROOT/'output/mechanism_ablation/selection.csv')
    row=selection[(selection.event==event)&(selection.fold==fold)&(selection.model=='TransPINN')].iloc[0]
    stem=f'{event}_fold{fold}_full_seed{row.seed}'
    path=ROOT/'output/rigorous_revision/full'/f'{stem}.json'
    record=json.loads(path.read_text())
    frame=pd.read_csv(path.with_suffix('.csv'))
    theta=np.array([record['run']['parameters'][k] for k in NAMES])
    times=np.arange(1,len(frame)+1,dtype=float)/len(frame)
    train=frame.split.to_numpy()=='train'
    actual=frame[['actual_rumor','actual_debunk']].to_numpy()
    total=np.maximum(actual[train].sum(0),1)
    obs=actual[train]/total
    scale=np.maximum(np.sqrt(np.mean(actual[train]**2,axis=0)),1)/total
    return theta,times[train],obs,scale,frame,int(row.seed)


def sensitivity(theta,times,step=1e-3,mesh=MESH):
    cols=[]
    for j in range(8):
        plus=theta.copy();minus=theta.copy()
        plus[j]*=np.exp(step);minus[j]*=np.exp(-step)
        yp=solve(plus,times,mesh)[1];ym=solve(minus,times,mesh)[1]
        cols.append(((np.log(yp)-np.log(ym))/(2*step)).ravel())
    matrix=np.array(cols).T
    return matrix,np.linalg.svd(matrix,compute_uv=False)


def fit(theta,times,obs,scale,fixed=None,start=None,max_nfev=1500):
    z=np.log(theta) if start is None else start.copy()
    mask=np.ones(8,dtype=bool)
    if fixed is not None:
        mask[fixed[0]]=False;z[fixed[0]]=fixed[1]
    def fun(free):
        full=z.copy();full[mask]=free
        return ((solve(np.exp(full),times,MESH)[1]-obs)/scale).ravel()
    result=least_squares(fun,np.clip(z[mask],LO[mask]+1e-12,HI[mask]-1e-12),
        bounds=(LO[mask],HI[mask]),diff_step=1e-4,ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=max_nfev)
    z[mask]=result.x
    return {'z':z,'rss':float(np.dot(result.fun,result.fun)),'success':bool(result.success),
        'nfev':result.nfev,'optimality':float(result.optimality),'status':result.status}


def audit_all_neural():
    rows=[];singular_rows=[];parameter_rows=[]
    for event in EVENTS:
        for fold in (1,2,3):
            theta,times,obs,scale,frame,seed=load(event,fold)
            jac,sv=sensitivity(theta,times)
            jac2,sv2=sensitivity(theta,times,step=5e-4,mesh=8000)
            _,num=reference.solve(theta,times,rtol=1e-10,atol=1e-12)
            fast=solve(theta,times,MESH)[1]
            error=float(np.max(abs(fast-num)))
            assert error<2e-7
            rows.append(dict(event=event,fold=fold,seed=seed,condition=sv[0]/sv[-1],
                rank_1e6=int(sum(sv>sv[0]*1e-6)),rank_1e3=int(sum(sv>sv[0]*1e-3)),
                derivative_refinement_relative=np.linalg.norm(jac-jac2)/np.linalg.norm(jac2),
                smallest_sv_refinement_relative=abs(sv[-1]-sv2[-1])/sv2[-1],integrator_max_error=error))
            for i,v in enumerate(sv):singular_rows.append(dict(event=event,fold=fold,index=i+1,singular_value=v,relative=v/sv[0]))
            for j,name in enumerate(NAMES):parameter_rows.append(dict(event=event,fold=fold,seed=seed,parameter=name,estimate=theta[j]))
        print('SENSITIVITY',event,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'sensitivity.csv',index=False)
    pd.DataFrame(singular_rows).to_csv(OUT/'singular_values.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(OUT/'neural_parameter_estimates.csv',index=False)


def profiles(event):
    theta,times,obs,scale,frame,seed=load(event)
    rng=np.random.default_rng(20260918)
    starts=[np.log(theta),(LO+HI)/2]+[rng.uniform(LO,HI) for _ in range(4)]
    fitted=[]
    for i,start in enumerate(starts):
        result=fit(theta,times,obs,scale,start=start)
        fitted.append(result)
        print('FIT',event,i,result['rss'],result['success'],result['nfev'],flush=True)
    best=min(fitted,key=lambda r:r['rss'])
    # Profiling may uncover a better basin. In that case restart all profiles
    # around the improved fit before computing any likelihood-ratio interval.
    n=obs.size;cutoff=float(chi2.ppf(.95,1))
    for cycle in range(3):
        rows=[];intervals=[];improved=best
        for j,name in enumerate(NAMES):
            grid=np.unique(np.r_[np.linspace(LO[j],HI[j],17),best['z'][j]])
            cache={}
            def evaluate(value):
                nonlocal improved
                if value in cache:return cache[value]
                start=best['z'] if not cache else min(cache.items(),key=lambda kv:abs(kv[0]-value))[1]['z']
                candidates=[fit(theta,times,obs,scale,(j,value),start=start,max_nfev=1000),
                            fit(theta,times,obs,scale,(j,value),start=best['z'],max_nfev=1000)]
                result=min(candidates,key=lambda r:r['rss'])
                if result['rss']<improved['rss']:improved=result
                cache[value]=result
                return result
            for value in sorted(grid,key=lambda z:abs(z-best['z'][j])):evaluate(float(value))
            lr=lambda z: n*np.log(max(evaluate(float(z))['rss']/best['rss'],1e-300))-cutoff
            center=best['z'][j]
            endpoints=[];limited=[]
            for direction in (-1,1):
                candidates=sorted([float(g) for g in grid if (g-center)*direction>1e-12],reverse=direction<0)
                inside=float(center);boundary=float(LO[j] if direction<0 else HI[j]);bound=True
                for value in candidates:
                    if lr(value)>0:
                        boundary=brentq(lr,min(inside,value),max(inside,value),xtol=1e-5)
                        bound=False;break
                    inside=value
                endpoints.append(float(np.exp(boundary)));limited.append(bound)
            for value,result in sorted(cache.items()):
                rows.append(dict(event=event,parameter=name,value=np.exp(value),rss=result['rss'],
                    likelihood_ratio=n*np.log(result['rss']/best['rss']),success=result['success'],
                    nfev=result['nfev'],optimality=result['optimality']))
            intervals.append(dict(event=event,parameter=name,estimate=np.exp(best['z'][j]),
                ci_low=endpoints[0],ci_high=endpoints[1],lower_bound_limited=limited[0],upper_bound_limited=limited[1]))
            print('PROFILE',event,name,endpoints,limited,flush=True)
        if improved['rss']<best['rss']*(1-1e-7):
            best=fit(theta,times,obs,scale,start=improved['z']);fitted.append(best)
            print('RESTART PROFILES',event,cycle,best['rss'],flush=True)
        else:break
    else:raise RuntimeError('Unresolved profile minimum')
    high=solve(np.exp(best['z']),times,8000)[1]
    independent=reference.solve(np.exp(best['z']),times,rtol=1e-10,atol=1e-12)[1]
    fine_error=float(np.max(abs(high-independent)))
    assert fine_error<5e-7
    pd.DataFrame(rows).to_csv(OUT/f'profiles_{event}.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/f'intervals_{event}.csv',index=False)
    for r in fitted:r['z']=r['z'].tolist()
    (OUT/f'fit_{event}.json').write_text(json.dumps(dict(event=event,fold=2,observations=n,
        best=best,multistarts=fitted,profile_cutoff=cutoff,mesh=MESH,
        refined_integrator_max_error=fine_error,noise='working independent Gaussian on training-RMS-normalized bin errors; common variance profiled',
        status='conditional approximate pointwise likelihood-ratio intervals; no empirical coverage or serial-dependence calibration'),indent=2,default=lambda x:x.tolist()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--event',choices=EVENTS)
    parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    structural_certificate()
    if args.audit:audit_all_neural()
    if args.event:profiles(args.event)
    (OUT/'protocol.json').write_text(json.dumps(dict(parameters=NAMES,bounds=list(zip(LOWER,UPPER)),
        history=HISTORY.tolist(),events=EVENTS,profile_fold=2,mesh=MESH,
        seed=20260918,global_starts=6,profile_grid=17,profile_nuisance_starts=2,
        source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ('experiments/run_identifiability.py','experiments/identifiability_solver.py','experiments/revision_dde.py','experiments/run_multievent_baselines.py')}),indent=2))


if __name__=='__main__':main()
