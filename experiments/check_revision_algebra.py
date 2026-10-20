"""Independent numerical checks of characteristic and control formulas."""
import json
from pathlib import Path
import numpy as np
import sympy as sp


def main():
    rng=np.random.default_rng(20260912)
    errors=[];threshold_errors=[]
    for _ in range(100):
        ae,h,kr,kd,c1,c2,xi,sigma,gr,gd,b1,b2=rng.uniform(.1,1,12)
        kr+=c1;kd+=c2
        psi,p,tau=rng.uniform(.1,.9,3)
        z=complex(*rng.uniform(-.5,.5,2))
        B=np.array([[z+kr,-c2],[-c1,z+kd]])
        adj=np.array([[z+kd,c2],[c1,z+kr]])
        nE=psi*np.array([b1,b2])@adj@np.array([xi,sigma])
        nH=psi*np.array([b1,b2])@adj@np.array([gr,gd])
        P=(z+ae)*(z+h)*np.linalg.det(B)-(z+h)*nE
        Q=(z+h)*nE-(z+ae)*nH
        delayed=p*np.exp(-(z+ae)*tau)
        F=P+delayed*Q
        J=np.array([[-ae,0,psi*b1,psi*b2],[0,-h,0,0],
                    [xi,gr,-kr,c2],[sigma,gd,c1,-kd]],dtype=complex)
        J[0,2:]-=delayed*psi*np.array([b1,b2])
        J[1,2:]+=delayed*psi*np.array([b1,b2])
        determinant=np.linalg.det(z*np.eye(4)-J)
        errors.append(abs(determinant-F)/max(1,abs(F)))
        theta=p*np.exp(-ae*tau);le=(1-theta)/ae
        delta=kr*kd-c1*c2
        zr=xi*le+theta*gr/h;zd=sigma*le+theta*gd/h
        R0=psi*(b1*(kd*zr+c2*zd)+b2*(c1*zr+kr*zd))/delta
        ne=psi*(b1*(kd*xi+c2*sigma)+b2*(c1*xi+kr*sigma))
        nh=psi*(b1*(kd*gr+c2*gd)+b2*(c1*gr+kr*gd))
        f0=ae*h*delta-h*ne+theta*(h*ne-ae*nh)
        threshold_errors.append(abs(f0-ae*h*delta*(1-R0)))
    m1,m2,qR,qD,qU,chi,w1,w2,rho=sp.symbols('m1 m2 qR qD qU chi w1 w2 rho',positive=True)
    Z=chi+m1+m2
    funcs=[qR*chi*m2/((1+w1*m1)*Z),qD*chi*m1/((1+w2*m2)*Z),qU*(m1+m2)/Z,
           rho*m1*m2/(1+rho*m1*m2)]
    expected=[[-qR*chi*m2*(w1*Z+1+w1*m1)/((1+w1*m1)**2*Z**2),
               qR*chi*(chi+m1)/((1+w1*m1)*Z**2)],
              [qD*chi*(chi+m2)/((1+w2*m2)*Z**2),
               -qD*chi*m1*(w2*Z+1+w2*m2)/((1+w2*m2)**2*Z**2)],
              [qU*chi/Z**2,qU*chi/Z**2],
              [rho*m2/(1+rho*m1*m2)**2,rho*m1/(1+rho*m1*m2)**2]]
    for f,derivatives in zip(funcs,expected):
        for v,d in zip((m1,m2),derivatives):assert sp.simplify(sp.diff(f,v)-d)==0
    assert max(errors)<1e-12 and max(threshold_errors)<1e-12
    output={'complex_determinant_cases':100,'maximum_relative_error':max(errors),
            'threshold_identity_maximum_error':max(threshold_errors),'symbolic_derivative_checks':8}
    path=Path(__file__).resolve().parents[1]/'output/rigorous_revision/tables/algebra_checks.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':main()
