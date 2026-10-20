"""Check a sufficient decay condition at the archived terminal equilibria.

This reuses the 116 archived integrations. It does not claim to calculate
the full Lyapunov spectrum or to provide interval-certified numerics.
"""
from pathlib import Path
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/identifiability-deps'))
import numpy as np
import pandas as pd
from scipy.special import lambertw
from identifiability_solver import rhs

OUT = ROOT / 'output/lyapunov_structure'
OUT.mkdir(parents=True, exist_ok=True)
runs = pd.read_csv(ROOT / 'output/lyapunov/lyapunov_runs.csv')
parameters = pd.read_csv(ROOT / 'output/lyapunov/parameter_sets.csv').set_index('case')
names = ['beta_1', 'beta_2', 'tau_normalized', 'q_R', 'q_D', 'chi', 'reporting_R', 'reporting_D']
rows = []
max_jacobian_error = 0.0
for record in runs.to_dict('records'):
    base = record['case'].split('_tau')[0]
    p = parameters.loc[base, names].to_numpy(dtype=float, copy=True)
    p[2] = record['tau']
    s = record['final_susceptible']
    psi = (5 / 13 + .55 * 8 / 13) * s
    beta1, beta2, tau, qr, qd, chi = p[:6]
    gr = qr * (.2 / 1.2) * chi / (chi + .4)
    gd = qd * (.2 / 1.2) * chi / (chi + .4)
    gu = .12 / (chi + .4)
    k = np.exp(-1.4 * tau) / 6
    A = np.array([[-1.4, 0, psi * beta1, psi * beta2],
                  [0, -(gr + gd + gu), 0, 0],
                  [.8, gr, -.35, .05],
                  [.4, gd, .1, -.3]])
    B = np.zeros((4, 4))
    B[0, 2:] = -k * psi * np.array([beta1, beta2])
    B[1, 2:] = k * psi * np.array([beta1, beta2])
    # The logarithmic 1-norm of A and the induced 1-norm of B.
    mu = np.max(np.diag(A) + np.sum(np.abs(A), axis=0) - np.abs(np.diag(A)))
    a = min(.2, gu, .25 - psi * beta1, .25 - psi * beta2)
    b = 2 * k * psi * max(beta1, beta2)
    np.testing.assert_allclose(-mu, a, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(B, 1), b, atol=1e-14)

    # Independent complex-step check against the implementation actually used.
    eq = np.array([5*s/13, 8*s/13, 0, 0, 0, 0, 1-s, .2, .2])
    J1 = np.empty((9, 9))
    J2 = np.empty((9, 9))
    for j in range(9):
        delta = np.zeros(9, dtype=complex)
        delta[j] = 1e-28j
        J1[:, j] = rhs(eq.astype(complex) + delta, eq.astype(complex), p).imag / 1e-28
        J2[:, j] = rhs(eq.astype(complex), eq.astype(complex) + delta, p).imag / 1e-28
    error = max(np.max(np.abs(J1[2:6, 2:6] - A)), np.max(np.abs(J2[2:6, 2:6] - B)))
    max_jacobian_error = max(max_jacobian_error, float(error))
    np.testing.assert_allclose(J1[2:6, 2:6], A, atol=2e-14)
    np.testing.assert_allclose(J2[2:6, 2:6], B, atol=2e-14)
    np.testing.assert_allclose(J1[2:6, [0, 1, 6, 7, 8]], 0, atol=2e-14)
    np.testing.assert_allclose(J2[2:6, [0, 1, 6, 7, 8]], 0, atol=2e-14)
    w = np.array([5/13, 8/13, 0, 0, 0, 0, -1, 0, 0])
    np.testing.assert_allclose((J1 + J2) @ w, 0, atol=2e-14)
    assert a > b >= 0
    eta = a - lambertw(b*tau*np.exp(a*tau)).real/tau
    assert eta > 0
    np.testing.assert_allclose(eta, a - b*np.exp(eta*tau), atol=2e-14)
    rows.append(dict(case=record['case'], trace=record['trace'], s=s,
                     tau=tau, a=a, b=b, margin=a-b, eta=eta,
                     transverse_upper_bound=-min(.13, 1., eta),
                     lambda_100_500=record['lambda_100_500'],
                     lambda_300_500=record['lambda_300_500']))

result = pd.DataFrame(rows)
result.to_csv(OUT/'terminal_equilibrium_decay.csv', index=False)
summary = dict(
    archived_runs=len(result),
    all_sufficient_conditions_pass=bool((result.margin > 0).all()),
    minimum_margin=float(result.margin.min()),
    maximum_margin=float(result.margin.max()),
    minimum_eta=float(result.eta.min()), maximum_eta=float(result.eta.max()),
    largest_transverse_upper_bound=float(result.transverse_upper_bound.max()),
    jacobian_check_max_error=max_jacobian_error,
    max_absolute_generic_estimate=float(result.lambda_100_500.abs().max()),
    later_window_max_absolute_estimate=float(result.lambda_300_500.abs().max()),
    units='inverse normalized time',
    scope='Sufficient exponential decay condition at numerical terminal equilibria of archived runs, not a full spectrum or interval certification.',
    source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        ROOT/'output/lyapunov/lyapunov_runs.csv', ROOT/'output/lyapunov/parameter_sets.csv', ROOT/'experiments/identifiability_solver.py']},
)
(OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
