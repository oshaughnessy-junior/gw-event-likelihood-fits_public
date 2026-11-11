"""Weighted EM mixtures of boundary-aware quadratic likelihood components.
Finite support in eta is included in every component's normalization.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.cluster.vq import kmeans2
from bounded_fit import fit,log_shape

def component_logpdf(x,f):
    if f.get('schema')=='bounded-eta-chi-quadratic-v1':return log_shape(x,f,physical=False)
    R=np.asarray(f['conditional_cov']);return log_shape(x,f,physical=False)-f['eta_log_integral']-np.log(2*np.pi)-.5*np.linalg.slogdet(R)[1]

def fit_mixture(x,w,k=2,maxiter=15,component_fitter=None,max_reference_rows=12000,
                min_component_weight=.01,min_component_ess=0):
    fit_component=fit if component_fitter is None else component_fitter
    x=np.asarray(x);w=np.asarray(w,float);w=w/w.sum();source_ess=float(1/(w@w));rng=np.random.default_rng(600+k)
    # Fixed equal-weight importance resample for EM cost; validate on full reference.
    if max_reference_rows and len(x)>max_reference_rows:
        idx=rng.choice(len(x),max_reference_rows,replace=True,p=w);x=x[idx];w=np.ones(len(x))/len(x)
    sd=np.sqrt(w@((x-w@x)**2));cent,labels=kmeans2(x/sd,k,iter=15,minit='++',seed=14)
    r=np.eye(k)[labels];prev=-np.inf;components=[]
    for step in range(maxiter):
        counts=(w[:,None]*r).sum(0);components=[]
        component_ess=counts**2/np.sum((w[:,None]*r)**2,axis=0)
        if counts.min()<min_component_weight or component_ess.min()<min_component_ess:raise ValueError('Collapsed or insufficient-information mixture component')
        for j in range(k):components.append(fit_component(x,w*r[:,j]+1e-14))
        logp=np.stack([component_logpdf(x,f) for f in components],1)+np.log(counts)
        norm=logsumexp(logp,axis=1);score=float(w@norm);r=np.exp(logp-norm[:,None])
        if step>2 and abs(score-prev)<1e-5:break
        prev=score
    return {'schema':'bounded-conditional-quadratic-mixture-v1','coords':['mc_source','eta','chi_eff'],
            'components':components,'component_weights':counts.tolist(),'em_iterations':step+1,
            'fit_method':'weighted EM of normalized eta-bounded quadratic components',
            'effective_sample_size':source_ess,'em_fitter_settings':{'max_reference_rows':max_reference_rows,'min_component_weight':min_component_weight,'min_component_ess':min_component_ess},
            'component_weight_ess':component_ess.tolist(),'em_objective':score,'em_reached_iteration_limit':bool(step+1>=maxiter),
            'conditioned_gradient_max':max(f.get('conditioned_gradient_max',f['gradient_max']) for f in components),
            'gaussian_parameters_identifiable':False,'gradient_max':max(f['gradient_max'] for f in components)}

def mixture_logpdf(x,f):
    return logsumexp(np.stack([component_logpdf(x,c)+np.log(p) for c,p in zip(f['components'],f['component_weights'])]),axis=0)
