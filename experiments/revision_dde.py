"""Independent method-of-steps integration of the revised normalized DDE.

No neural output is used by the numerical integrator. Parameters below match
the empirical implementation, not measured hourly rates.
"""
import numpy as np
from scipy.integrate import solve_ivp

HISTORY = np.array([.93,.03,.015,.005,.010,.005,.005,.20,.20])
NAMES = ('beta_1','beta_2','tau_normalized','q_R','q_D','chi','reporting_R','reporting_D')
LOWER = np.array([.10,.05,.01,.05,.05,.05,.25,.25])
UPPER = np.array([8.,5.,.22,4.,4.,3.,12.,12.])


def values(theta):
    return dict(zip(NAMES,theta)) if not isinstance(theta,dict) else theta


def rates(x,p,variant='full'):
    m1,m2=x[7:9]
    load=p['chi']+m1+m2
    gr=p['q_R']*m2*p['chi']/((1+m1)*load)
    gd=p['q_D']*m1*p['chi']/((1+m2)*load)
    gu=.3*(m1+m2)/load
    if variant=='no_fatigue':
        gr=p['q_R']*m2/(1+m1);gd=p['q_D']*m1/(1+m2);gu=0.
    elif variant=='no_inhibition':
        gr=p['q_R']*m2*p['chi']/load;gd=p['q_D']*m1*p['chi']/load
    return gr,gd,gu


def rhs(t,x,xd,p,variant='full'):
    su,se,e,h,r,d,u,m1,m2=x
    gr,gd,gu=rates(x,p,variant)
    inc=(p['beta_1']*r+p['beta_2']*d)*(su+.55*se)
    past=(p['beta_1']*xd[4]+p['beta_2']*xd[5])*(xd[0]+.55*xd[1])
    gate=5*xd[7]*xd[8]/(1+5*xd[7]*xd[8])
    q=gate*past*np.exp(-1.4*p['tau_normalized'])
    return np.array([
        -.4*m1*su+.25*m2*se-(p['beta_1']*r+p['beta_2']*d)*su,
        .4*m1*su-.25*m2*se-.55*(p['beta_1']*r+p['beta_2']*d)*se,
        inc-1.4*e-q,q-(gr+gd+gu)*h,
        .8*e+gr*h-(.25+.5*m1)*r+.25*m2*d,
        .4*e+gd*h+.5*m1*r-(.25+.25*m2)*d,
        .2*e+.25*r+.25*d+gu*h,
        .6*d+.2-m1,.6*r+.2-m2])


def solve(theta,times,rtol=1e-8,atol=1e-10,variant='full'):
    p=values(theta);tau=p['tau_normalized'];times=np.asarray(times)
    if tau<=0: raise ValueError('This method-of-steps implementation requires positive delay')
    gate=5*HISTORY[7]*HISTORY[8]/(1+5*HISTORY[7]*HISTORY[8])
    past=(p['beta_1']*HISTORY[4]+p['beta_2']*HISTORY[5])*(HISTORY[0]+.55*HISTORY[1])
    if HISTORY[2]+1e-12 < gate*past*(-np.expm1(-1.4*tau))/1.4:
        raise ValueError('Incompatible cohort history')
    solutions=[];endpoints=[];current=HISTORY.copy();left=0.;horizon=float(times.max())
    def lookup(t):
        if t<=1e-12:return HISTORY
        idx=min(np.searchsorted(endpoints,t,side='left'),len(solutions)-1)
        return solutions[idx].sol(t)
    while left<horizon-1e-12:
        right=min(horizon,left+tau)
        result=solve_ivp(lambda t,x:rhs(t,x,lookup(t-tau),p,variant),
            (left,right),current,rtol=rtol,atol=atol,dense_output=True,max_step=min(.02,tau/2))
        if not result.success: raise RuntimeError(result.message)
        solutions.append(result);endpoints.append(right);current=result.y[:,-1];left=right
    states=np.vstack([lookup(t) for t in times])
    if np.min(states)<-1e-7:raise AssertionError('Negative numerical state')
    if np.max(np.abs(states[:,:7].sum(axis=1)-1))>1e-7:raise AssertionError('Mass conservation')
    flux=[]
    for x in states:
        gr,gd,_=rates(x,p,variant)
        flux.append([.8*x[2]+gr*x[3]+.25*x[8]*x[5],
                     .4*x[2]+gd*x[3]+.5*x[7]*x[4]])
    return states,np.array(flux)*np.array([p['reporting_R'],p['reporting_D']])


def sensitivity(theta,times,relative_step=1e-3):
    theta=np.asarray(theta)
    _,y=solve(theta,times)
    matrix=[]
    for k in range(len(theta)):
        plus=theta.copy();minus=theta.copy()
        plus[k]*=np.exp(relative_step);minus[k]*=np.exp(-relative_step)
        _,yp=solve(plus,times);_,ym=solve(minus,times)
        # Elasticity of the observation relative to each positive parameter.
        matrix.append(((np.log(yp)-np.log(ym))/(2*relative_step)).ravel())
    matrix=np.array(matrix).T
    singular=np.linalg.svd(matrix,compute_uv=False)
    return matrix,singular


if __name__=='__main__':
    p=np.array([2.5,1.3,.073,1.,1.,1.,.83,.83])
    t=np.linspace(0,1,101)
    x,y=solve(p,t)
    xx,yy=solve(p,t,rtol=1e-10,atol=1e-12)
    print('max tolerance-refinement state difference',np.max(np.abs(x-xx)))
    print('mass error',np.max(np.abs(x[:,:7].sum(axis=1)-1)), 'minimum state',x.min())
    _,singular=sensitivity(p,t[1:])
    print('sensitivity singular values',singular)
