"""Deterministic RK4 method of steps for conditional identifiability diagnostics.

All constants, state order, history, and right-endpoint observation convention
match revision_dde.py. The mesh is fixed during optimization. Numba accelerates
arithmetic only; no neural outputs enter the solution.
"""
import numpy as np
from numba import njit

HISTORY = np.array([.93, .03, .015, .005, .010, .005, .005, .20, .20])


@njit(cache=True)
def rates(x, p):
    m1, m2 = x[7], x[8]
    load = p[5]+m1+m2
    return (p[3]*m2*p[5]/((1+m1)*load),
            p[4]*m1*p[5]/((1+m2)*load), .3*(m1+m2)/load)


@njit(cache=True)
def rhs(x, xd, p):
    su,se,e,h,r,d,u,m1,m2 = x
    gr,gd,gu = rates(x,p)
    inc = (p[0]*r+p[1]*d)*(su+.55*se)
    past = (p[0]*xd[4]+p[1]*xd[5])*(xd[0]+.55*xd[1])
    gate = 5*xd[7]*xd[8]/(1+5*xd[7]*xd[8])
    q = gate*past*np.exp(-1.4*p[2])
    return np.array([-.4*m1*su+.25*m2*se-(p[0]*r+p[1]*d)*su,
        .4*m1*su-.25*m2*se-.55*(p[0]*r+p[1]*d)*se,
        inc-1.4*e-q, q-(gr+gd+gu)*h,
        .8*e+gr*h-(.25+.5*m1)*r+.25*m2*d,
        .4*e+gd*h+.5*m1*r-(.25+.25*m2)*d,
        .2*e+.25*r+.25*d+gu*h, .6*d+.2-m1, .6*r+.2-m2])


@njit(cache=True)
def lookup(trajectory, t, step):
    if t <= 0:
        return HISTORY.copy()
    position = t/step
    i = int(position)
    fraction = position-i
    return trajectory[i]*(1-fraction)+trajectory[i+1]*fraction


@njit(cache=True)
def solve(p, times, mesh=4000):
    step = 1./mesh
    n = int(np.ceil(times[-1]/step))+1
    trajectory = np.empty((n+1,9))
    trajectory[0] = HISTORY
    for i in range(n):
        x = trajectory[i]
        t = i*step
        k1 = rhs(x,lookup(trajectory,t-p[2],step),p)
        xd = lookup(trajectory,t+step/2-p[2],step)
        k2 = rhs(x+step*k1/2,xd,p)
        k3 = rhs(x+step*k2/2,xd,p)
        k4 = rhs(x+step*k3,lookup(trajectory,t+step-p[2],step),p)
        trajectory[i+1] = x+step*(k1+2*k2+2*k3+k4)/6
    states = np.empty((len(times),9))
    flux = np.empty((len(times),2))
    for i in range(len(times)):
        x = lookup(trajectory,times[i],step)
        states[i] = x
        gr,gd,_ = rates(x,p)
        flux[i,0] = p[6]*(.8*x[2]+gr*x[3]+.25*x[8]*x[5])
        flux[i,1] = p[7]*(.4*x[2]+gd*x[3]+.5*x[7]*x[4])
    return states,flux
