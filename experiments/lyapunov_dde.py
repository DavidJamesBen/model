"""RK4 method of steps + history-normalized variational largest exponent.

Complex-step directional differentiation evaluates A(t)v+B(t)v_delay without
finite-amplitude trajectory subtraction. All human perturbations have sum zero.
The complete last-delay history is renormalized, not just the current vector.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/identifiability-deps'))
import numpy as np
from numba import njit
from identifiability_solver import rhs,HISTORY

@njit(cache=True)
def field(x,xd,p,mode):
    if mode==0:
        return rhs(x,xd,p)
    f=x.copy()*0
    # Known stable retarded equation on the sum-zero human tangent space.
    a=-.3 if mode==1 else .1
    f[:6]=a*x[:6]+.1*xd[:6]
    f[6]=-f[:6].sum()
    f[7:]=-.7*x[7:]
    return f

@njit(cache=True)
def augmented(z,zd,p,mode):
    f=field(z[:9],zd[:9],p,mode)
    tangent=field(z[:9].astype(np.complex128)+1e-28j*z[9:],
                  zd[:9].astype(np.complex128)+1e-28j*zd[9:],p,mode).imag/1e-28
    return np.concatenate((f,tangent))

@njit(cache=True)
def interpolate(a,position):
    lo=int(np.floor(position))
    fraction=position-lo
    return a[lo]*(1-fraction)+a[lo+1]*fraction

@njit(cache=True)
def history_norm(a,left,right):
    # L2 RMS norm on [-tau,0], trapezoidal quadrature.
    total=0.
    for j in range(left,right+1):
        w=.5 if j==left or j==right else 1.
        total+=w*np.sum(a[j,9:]**2)
    return np.sqrt(total/(right-left))

@njit(cache=True)
def integrate(p,history,direction,dt_requested,total=500.,renorm=1.,mode=0):
    tau=p[2]
    m=max(4,int(np.ceil(tau/dt_requested)))
    dt=tau/m
    steps=int(np.ceil(total/dt))
    interval=max(1,int(round(renorm/dt)))
    z=np.zeros((m+steps+1,18))
    v=direction.copy()
    v[6]=-np.sum(v[:6])
    v=v/np.sqrt(np.sum(v*v))
    for j in range(m+1):
        z[j,:9]=history
        z[j,9:]=v
    logs=np.empty((steps//interval+1,3))
    k=0
    minimum=history.min()
    mass_error=0.
    tangent_mass_error=0.
    for i in range(m,m+steps):
        current=z[i]
        past=z[i-m]
        middle=(z[i-m]+z[i-m+1])*.5
        end=z[i-m+1]
        k1=augmented(current,past,p,mode)
        k2=augmented(current+.5*dt*k1,middle,p,mode)
        k3=augmented(current+.5*dt*k2,middle,p,mode)
        k4=augmented(current+dt*k3,end,p,mode)
        z[i+1]=current+dt*(k1+2*k2+2*k3+k4)/6
        # Enforce the exact tangent invariant; otherwise roundoff in a neutral
        # mass direction can eventually dominate a genuinely decaying vector.
        z[i+1,15]=-z[i+1,9:15].sum()
        minimum=min(minimum,z[i+1,:9].min())
        mass_error=max(mass_error,abs(z[i+1,:7].sum()-history[:7].sum()))
        tangent_mass_error=max(tangent_mass_error,abs(z[i+1,9:16].sum()))
        if (i+1-m)%interval==0:
            norm=history_norm(z,i+1-m,i+1)
            logs[k,0]=(i+1-m)*dt
            logs[k,1]=np.log(norm)
            logs[k,2]=np.max(np.abs(field(z[i+1,:9],z[i+1-m,:9],p,mode)))
            k+=1
            z[i+1-m:i+2,9:]/=norm
    return logs[:k],z[-1,:9],minimum,mass_error,tangent_mass_error,dt

def compatibility(p,history=HISTORY):
    gate=5*history[7]*history[8]/(1+5*history[7]*history[8])
    incidence=(p[0]*history[4]+p[1]*history[5])*(history[0]+.55*history[1])
    return float(history[2]-gate*incidence*(-np.expm1(-1.4*p[2]))/1.4)

def estimate(logs,burn,end):
    start=np.searchsorted(logs[:,0],burn,side='right')
    stop=np.searchsorted(logs[:,0],end,side='right')
    t0=0 if start==0 else logs[start-1,0]
    return float(logs[start:stop,1].sum()/(logs[stop-1,0]-t0))
