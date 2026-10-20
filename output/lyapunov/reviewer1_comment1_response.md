# Suggested replacement for the Lyapunov part of Reviewer 1, Comment 1

We agree that a claim of chaos requires quantification based on Lyapunov theory; oscillatory trajectories or a Hopf bifurcation alone do not establish chaos. We have therefore added a numerical largest-Lyapunov-exponent analysis of the underlying closed empirical delay differential equation, integrating its variational equation and repeatedly normalizing the complete perturbation history while preserving the fixed human population. The analysis covers 12 archived neural parameter sets and four independent training-data-only DDE fits, with 116 runs assessing mesh, integration duration, perturbation direction, renormalization interval, history and delay sensitivity. Analytic stable and positive-exponent DDE benchmarks and an independent adaptive state solver check the implementation.

For integration to normalized time 500 after discarding the first 100 time units, the absolute exponent estimates are below 2.95 × 10^-8 and converge to zero on later windows. The trajectories approach a continuous family of no-spreading equilibria; its neutral tangent direction explains the zero limit. Thus, these computations detect no sustained positive largest Lyapunov exponent and provide no evidence of sustained chaos under the tested conditions. This is a conditional numerical conclusion, not a global no-chaos theorem or a claim about the observed events themselves. We have added a dedicated subsection with Theorem III.3 and its proof, establishing the complete-history growth identity and the neutral equilibrium direction (with an explicit spectral condition for a zero maximal exponent), and supplied the numerical method, convergence plots, parameter tables and reproducible outputs in the accompanying supplementary Lyapunov analysis.

## 中文说明

现在已经实际完成李雅普诺夫量化，回复中应删除此前“未计算／不适合分析／留待未来工作”的对应表述。时滞系统并非不能做李雅普诺夫分析；关键是要在整个历史函数空间传播和归一化扰动。本次分析针对固定参数的底层 DDE，而不是神经网络本身。计算未发现持续正指数，因此不能宣称发现混沌。

这段仅用于替换 Comment 1 中关于混沌量化的部分，其他审稿问题的回复保留。
