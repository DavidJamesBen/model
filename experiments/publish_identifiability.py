"""Build traceable tables/figures after all four profile jobs complete."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_identifiability import OUT,ROOT,NAMES,EVENTS,LOWER,UPPER,load,sensitivity,fit
from identifiability_solver import solve

LABELS=[r'$\beta_1$',r'$\beta_2$',r'$\tau_s$',r'$q_R$',r'$q_D$',r'$\chi$',r'$\widetilde s_R$',r'$\widetilde s_D$']
TITLES={'harvey':'Harvey','csdc':'CSDC','ottawa':'Ottawa','sydney':'Sydney'}


def fmt(x):
    return f'{x:.3f}'


def main():
    intervals=pd.concat([pd.read_csv(OUT/f'intervals_{e}.csv') for e in EVENTS],ignore_index=True)
    profiles=pd.concat([pd.read_csv(OUT/f'profiles_{e}.csv') for e in EVENTS],ignore_index=True)
    repairs=[]
    for i,row in profiles[~profiles.success].iterrows():
        theta,times,obs,scale,_,_=load(row.event)
        record=json.loads((OUT/f'fit_{row.event}.json').read_text())
        result=fit(theta,times,obs,scale,(NAMES.index(row.parameter),np.log(row.value)),
                   start=np.array(record['best']['z']),max_nfev=6000)
        assert result['success']
        assert abs(result['rss']-row.rss)/row.rss<1e-6, 'Refinement requires recomputing profile crossings'
        result['z']=result['z'].tolist()
        repairs.append(dict(event=row.event,parameter=row.parameter,value=row.value,previous_rss=row.rss,result=result))
        for key in ('rss','success','nfev','optimality'):profiles.loc[i,key]=result[key]
        profiles.loc[i,'likelihood_ratio']=record['observations']*np.log(result['rss']/record['best']['rss'])
    (OUT/'profile_refinements.json').write_text(json.dumps(repairs,indent=2))
    assert len(intervals)==32
    assert np.isfinite(intervals[['estimate','ci_low','ci_high']]).all().all()
    assert (intervals.ci_low<=intervals.estimate+1e-7).all()
    assert (intervals.ci_high>=intervals.estimate-1e-7).all()
    assert profiles.likelihood_ratio.min()>-1e-4
    assert profiles.success.all(), 'Some nuisance profile fits did not converge'
    intervals.to_csv(OUT/'parameter_profile_intervals.csv',index=False)
    profiles.to_csv(OUT/'parameter_profiles.csv',index=False)
    rows=[]
    for name,label in zip(NAMES,LABELS):
        cells=[]
        for event in EVENTS:
            r=intervals[(intervals.parameter==name)&(intervals.event==event)].iloc[0]
            low=fmt(r.ci_low)+('^{*}' if r.lower_bound_limited else '')
            high=fmt(r.ci_high)+('^{*}' if r.upper_bound_limited else '')
            cells.append(rf'${fmt(r.estimate)}\;[{low},\,{high}]$')
        rows.append(label+' & '+' & '.join(cells)+r' \\')
    (OUT/'profile_rows.tex').write_text('\n'.join(rows),encoding='utf-8')
    points=pd.read_csv(OUT/'neural_parameter_estimates.csv')
    rows=[]
    for name,label in zip(NAMES,LABELS):
        cells=[]
        for event in EVENTS:
            p=points[(points.parameter==name)&(points.event==event)].estimate
            cells.append(rf'${p.median():.4f}\;[{p.min():.4f},\,{p.max():.4f}]$')
        rows.append(label+' & '+' & '.join(cells)+r' \\')
    (OUT/'neural_rows.tex').write_text('\n'.join(rows),encoding='utf-8')
    diagnostics=[];correlations=[];replays=[]
    for event in EVENTS:
        theta,times,obs,scale,frame,seed=load(event)
        m,sv=sensitivity(theta,times)
        cosine=m.T@m/np.outer(np.linalg.norm(m,axis=0),np.linalg.norm(m,axis=0))
        for j in range(8):
            for k in range(8):correlations.append(dict(event=event,parameter_1=NAMES[j],parameter_2=NAMES[k],sensitivity_cosine=cosine[j,k]))
        obj=json.loads((OUT/f'fit_{event}.json').read_text())
        assert obj['best']['success']
        z=np.array(obj['best']['z']);est=np.exp(z)
        # Weighted local Jacobian at the independent DDE likelihood optimum.
        jcols=[]
        for k in range(8):
            zp=z.copy();zm=z.copy();zp[k]+=1e-3;zm[k]-=1e-3
            jcols.append(((solve(np.exp(zp),times)[1]-solve(np.exp(zm),times)[1])/(2e-3*scale)).ravel())
        jac=np.array(jcols).T;ss=np.linalg.svd(jac,compute_uv=False)
        covariance=np.linalg.pinv(jac.T@jac,rcond=1e-14)
        corr=covariance/np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
        pd.DataFrame(corr,index=NAMES,columns=NAMES).to_csv(OUT/f'local_parameter_correlation_{event}.csv')
        diagnostic=intervals[intervals.event==event]
        full=0
        for r in diagnostic.itertuples():
            k=NAMES.index(r.parameter)
            full+=int(r.lower_bound_limited and r.upper_bound_limited)
        diagnostics.append(dict(event=event,training_bins=len(times),scalars=obj['observations'],
            rms_standardized=np.sqrt(obj['best']['rss']/obj['observations']),
            profile_intervals_at_both_bounds=full,profile_intervals_at_either_bound=int((diagnostic.lower_bound_limited|diagnostic.upper_bound_limited).sum()),
            all_multistarts_converged=all(r['success'] for r in obj['multistarts']),
            likelihood_sensitivity_condition=ss[0]/ss[-1],minimum_sv=ss[-1],
            integrator_error=obj['refined_integrator_max_error']))
        # Neural coefficients integrated independently: quantify why DDE
        # likelihood intervals cannot be assigned to neural coefficients.
        all_times=np.arange(1,len(frame)+1,dtype=float)/len(frame)
        raw=frame[['actual_rumor','actual_debunk']].to_numpy()
        tr=frame.split.to_numpy()=='train';te=frame.split.to_numpy()=='test'
        den=np.maximum(np.sqrt(np.mean(raw[tr]**2,axis=0)),1)
        counts=solve(theta,all_times)[1]*np.maximum(raw[tr].sum(0),1)
        pred=frame[['predicted_rumor','predicted_debunk']].to_numpy()
        replays.append(dict(event=event,fold=2,neural_nrmse=np.mean(np.sqrt(np.mean((pred[te]-raw[te])**2,axis=0))/den),
            dde_replay_nrmse=np.mean(np.sqrt(np.mean((counts[te]-raw[te])**2,axis=0))/den)))
    pd.DataFrame(diagnostics).to_csv(OUT/'profile_diagnostics.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'sensitivity_correlations.csv',index=False)
    pd.DataFrame(replays).to_csv(OUT/'neural_dde_consistency.csv',index=False)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'font.size': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.bfit': 'Times New Roman:italic:bold',
        'mathtext.sf': 'Times New Roman',
        'mathtext.tt': 'Times New Roman',
        'mathtext.cal': 'Times New Roman:italic',
        'mathtext.fallback': None,
    })
    fig,axes=plt.subplots(2,4,figsize=(7.3,4.3),layout='constrained')
    colors=['#276298','#bc5034','#368064','#875a9e']
    for j,ax in enumerate(axes.flat):
        for color,event in zip(colors,EVENTS):
            p=profiles[(profiles.event==event)&(profiles.parameter==NAMES[j])].sort_values('value')
            ax.plot(p.value,p.likelihood_ratio,color=color,label=TITLES[event],lw=1)
        ax.axhline(3.8414588,color='black',ls='--',lw=.8)
        ax.set_xscale('log');ax.set_xlim(LOWER[j],UPPER[j]);ax.set_ylim(-.1,8)
        ax.set_title(LABELS[j]);ax.grid(alpha=.15)
        if j%4==0:ax.set_ylabel('Profile likelihood ratio')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(OUT/'profile_likelihoods.pdf');fig.savefig(OUT/'profile_likelihoods.png',dpi=220);plt.close(fig)
    sv=pd.read_csv(OUT/'singular_values.csv')
    fig,ax=plt.subplots(figsize=(3.5,2.4),layout='constrained')
    for color,event in zip(colors,EVENTS):
        q=sv[(sv.event==event)&(sv.fold==2)]
        ax.semilogy(q['index'],q.relative,'o-',label=TITLES[event],color=color,ms=3)
    ax.axhline(1e-3,color='gray',ls='--',lw=.8);ax.axhline(1e-6,color='gray',ls=':',lw=.8)
    ax.set_xlabel('Singular-value index');ax.set_ylabel(r'$\sigma_i/\sigma_1$');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
    fig.savefig(OUT/'sensitivity_spectrum.pdf');fig.savefig(OUT/'sensitivity_spectrum.png',dpi=220);plt.close(fig)
    (OUT/'publication_audit.json').write_text(json.dumps(dict(profile_parameters=32,
        profile_points=len(profiles),all_profile_fits_converged=bool(profiles.success.all()),
        intervals_at_either_bound=int((intervals.lower_bound_limited|intervals.upper_bound_limited).sum()),
        intervals_at_both_bounds=int((intervals.lower_bound_limited&intervals.upper_bound_limited).sum())),indent=2))
    at_either=int((intervals.lower_bound_limited|intervals.upper_bound_limited).sum())
    at_both=int((intervals.lower_bound_limited&intervals.upper_bound_limited).sum())
    (OUT/'results_discussion.tex').write_text(
        rf'Table~\ref{{tab:profile_parameters}} shows that {at_either} of the 32 intervals reach at least one imposed bound; '
        rf'{at_both} span both bounds. In every event, $\tau_s$, $q_R$, $q_D$, and $\chi$ retain their entire search ranges. '
        r'The data therefore do not provide two-sided constraints on these delay, adoption, and fatigue parameters. '
        r'The reporting gains are better constrained in this conditional problem, although their ranges remain wide. '
        r'All 24 initial fits satisfy an optimizer stopping criterion; this is not a global-optimum certificate. '
        r'One nuisance fit required an increased evaluation budget, which changed its RSS by less than $10^{-6}$ relative and did not alter the reported interval crossings. '
        r'At the independent DDE optima, weighted sensitivity condition numbers range from $1.66\times10^5$ to $2.82\times10^8$, reinforcing the practical-identifiability limitation.'
        '\n\n'
        r'The distinction from neural forecasting is material. Reintegrating the middle-fold Harvey neural coefficients gives test NRMSE 1.529, whereas the neural activity head gives 0.261 under the same normalization. '
        r'This replay does not refit parameters and is not a new forecast benchmark. It shows that a soft residual penalty does not make the neural trajectory an exact DDE solution. '
        r'Accordingly, we report neural fitted values and their fold variation separately from DDE profile intervals; assigning the latter to the former would be unjustified.'
        '\n',encoding='utf-8')
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print(pd.DataFrame(replays).to_string(index=False))


if __name__=='__main__':main()
