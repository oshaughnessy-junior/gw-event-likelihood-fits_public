"""Boundary-aware 3D likelihood-shape fits from prior-divided reference weights.

The Gaussian is fitted by maximizing the normalized TRUNCATED likelihood.
Weighted sample moments are sufficient statistics, NOT Gaussian parameters.
Truncation during fitting is 0 < eta <= .25. Other physical bounds are enforced
in evaluation/sampling and their effect must pass reference validation.
A natural-parameter boundary limit is stored without inventing a covariance.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad_vec

ETA_MAX=.25

def fit(x, w, eta_bounds=(0.,.25)):
    """MLE in centered, scaled eta natural coordinates (same model family)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if x.ndim != 2 or x.shape[1] != 3 or len(w) != len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid 3D reference")
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid reference weights")
    w = w / w.sum()
    t = x[:, 1] / .25
    if np.any((t <= 0) | (t > 1)):
        raise ValueError('Reference outside physical eta support')
    m = float(w @ t)
    sd = float(np.sqrt(w @ ((t-m)**2)))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Degenerate eta reference")
    eta_lo,eta_hi=map(float,eta_bounds)
    if not 0<=eta_lo<eta_hi<=.25 or np.any((x[:,1]<eta_lo)|(x[:,1]>eta_hi)):
        raise ValueError('Reference outside declared fitting eta interval')
    lo, hi = (eta_lo/.25-m)/sd, (eta_hi/.25-m)/sd

    def integral(p):
        a, b = p
        peak = np.clip(-a/(2*b), lo, hi) if b < 0 else (hi if a*hi+b*hi*hi > a*lo+b*lo*lo else lo)
        top = a*peak+b*peak*peak
        width = min(hi-lo, 1/np.sqrt(max(abs(2*b), 1e-15)), 1/max(abs(a+2*b*peak), 1e-15))
        pts = sorted(set([lo, hi, float(peak)] + [float(np.clip(peak+j*width, lo, hi)) for j in [-12, -4, -1, 1, 4, 12]]))
        z, _ = quad_vec(lambda u: np.exp(a*u+b*u*u-top)*np.array([1., u, u*u, u**3, u**4]), lo, hi,
                        points=pts, epsabs=1e-11, epsrel=1e-11)
        v = z/z[0]
        h = np.array([[v[2]-v[1]**2, v[3]-v[1]*v[2]], [v[3]-v[1]*v[2], v[4]-v[2]**2]])
        return top+np.log(z[0]), v[1:3]-[0., 1.], h

    def fun(p):
        a, g, _ = integral(p)
        return a-p[1], g

    res = minimize(fun, [0., -.5], jac=True, hess=lambda p: integral(p)[2], method='trust-exact',
                   options={'gtol':1e-10, 'maxiter':500})
    lz, grad, _ = integral(res.x)
    aa, bb = res.x
    b = bb/sd**2
    a = aa/sd-2*b*m
    y = x[:, [0, 2]]
    ym = w @ y
    beta = np.sum(w[:, None]*(t-m)[:, None]*(y-ym), axis=0)/sd**2
    intercept = ym-beta*m
    e = y-intercept-t[:, None]*beta
    R = (w[:, None]*e).T @ e
    np.linalg.cholesky(R)
    # Original E[t], E[t^2] residuals, directly transformed from scaled residuals.
    residual = [sd*grad[0], 2*m*sd*grad[0]+sd*sd*grad[1]]
    f = {'schema':'bounded-conditional-quadratic-v1', 'coords':['mc_source','eta','chi_eff'],
         'eta_a':float(a), 'eta_b':float(b), 'conditional_intercept':intercept.tolist(),
         'conditional_slope':beta.tolist(), 'conditional_cov':R.tolist(),
         'eta_log_integral':float(lz+np.log(.25*sd)-b*m*m+aa*m/sd),
         'gradient_max':float(np.max(np.abs(residual))), 'optimizer_success':bool(res.success),
         'conditioned_gradient_max':float(np.max(np.abs(grad))),
         'effective_sample_size':float(1/(w@w)),
         'fit_method':'weighted truncated maximum likelihood in centered scaled eta natural coordinates',
         'fit_truncation':['eta normalized on declared interval'], 'fit_eta_bounds':[eta_lo,eta_hi],
         'evaluation_bounds':{'mc_source':[0,None],'eta':[eta_lo,eta_hi],'chi_eff':[-1,1]},
         'gaussian_parameters_identifiable':False}
    if b < -1e-5:
        vt = -1/(2*b)
        mt = -a/(2*b)
        mu = np.empty(3); mu[1] = mt*.25; mu[[0,2]] = intercept+beta*mt
        C = np.zeros((3,3)); C[1,1] = vt*.25**2
        C[np.ix_([0,2],[0,2])] = R+np.outer(beta,beta)*vt
        C[[0,2],1] = C[1,[0,2]] = beta*vt*.25
        f.update(gaussian_mu=mu.tolist(), gaussian_cov=C.tolist(), gaussian_parameters_identifiable=bool(abs(b)>.01))
    return f


def _physical_mask(x,f):
    x=np.asarray(x,float)
    bounds=f.get('evaluation_bounds')
    if bounds is None and f.get('components'):
        bounds=f['components'][0].get('evaluation_bounds')
    bounds=bounds or {'mc_source':[0,None],'eta':[0,.25],'chi_eff':[-1,1]}
    good=np.isfinite(x).all(-1)&(x[...,0]>0)&(x[...,1]>0)&(x[...,1]<=.25)
    lo,hi=bounds.get('chi_eff',[-1,1]);good&=(x[...,2]>=lo)&(x[...,2]<=hi)
    if f.get('analysis_support'):
        from prior_folded_likelihood import support
        good&=support(x,f['analysis_support'])
    return good

def log_shape(x,f,physical=True):
    if f.get("schema")=="prior-folded-joint-quadratic-mixture-v1":
        from prior_folded_likelihood import log_shape as folded_log_shape
        return folded_log_shape(x,f,physical)
    if f.get("schema")=="bounded-eta-chi-quadratic-v1":
        from bounded2d_fit import log_shape as bounded_log_shape
        a=np.asarray(x,float);out=bounded_log_shape(a.reshape(-1,3),f,physical).reshape(a.shape[:-1])
        return np.where(_physical_mask(a,f),out,-np.inf)if physical else out
    if 'components' in f:
        from mixture_fit import mixture_logpdf
        out=mixture_logpdf(np.asarray(x,float),f)
        if physical:
            a=np.asarray(x,float);out=np.where(_physical_mask(a,f),out,-np.inf)
        return out
    x=np.asarray(x,float);t=x[...,1]/ETA_MAX;y=x[...,[0,2]]
    e=y-np.array(f['conditional_intercept'])-t[...,None]*np.array(f['conditional_slope'])
    ri=np.linalg.inv(np.array(f['conditional_cov']))
    out=f['eta_a']*t+f['eta_b']*t*t-.5*np.einsum('...i,ij,...j->...',e,ri,e)
    eta_lo,eta_hi=f.get('fit_eta_bounds',[0.,.25])
    good=np.isfinite(x).all(-1)&(t>0)&(x[...,1]>=eta_lo)&(x[...,1]<=eta_hi)
    if physical:good&=_physical_mask(x,f)
    return np.where(good,out,-np.inf)

def _draw(f,n,rng):
    if f.get("schema")=="bounded-eta-chi-quadratic-v1":
        from bounded2d_fit import sample as bounded_sample
        return bounded_sample(f,n,seed=int(rng.integers(2**31)),mc_positive=False)[0]
    if 'components' in f:
        labels=rng.choice(len(f['components']),n,p=f['component_weights'])
        out=np.empty((n,3))
        for j,c in enumerate(f['components']):
            mask=labels==j
            if mask.any():out[mask]=_draw(c,int(mask.sum()),rng)
        return out
    elo,ehi=f.get('fit_eta_bounds',[0.,.25]);t=np.linspace(elo/.25,ehi/.25,100001);lp=f['eta_a']*t+f['eta_b']*t*t;v=np.exp(lp-lp.max())
    c=np.concatenate(([0.],np.cumsum((v[1:]+v[:-1])*.5*np.diff(t))));c/=c[-1]
    z=np.interp(rng.random(n),c,t)
    e=rng.normal(size=(n,2))@np.linalg.cholesky(f['conditional_cov']).T
    y=np.array(f['conditional_intercept'])+z[:,None]*np.array(f['conditional_slope'])+e
    return np.column_stack([y[:,0],z*ETA_MAX,y[:,1]])

def sample(f,n=40000,seed=1):
    if f.get("schema")=="prior-folded-joint-quadratic-mixture-v1":
        from prior_folded_likelihood import sample as folded_sample
        return folded_sample(f,n,seed)
    # Reject after mixing: physical truncation changes relative component masses.
    rng=np.random.default_rng(seed);out=[];total=0;drawn=0
    for _ in range(50):
        a=_draw(f,max(2*n,1000),rng);drawn+=len(a)
        ok=_physical_mask(a,f)
        out.append(a[ok]);total+=int(ok.sum())
        if total>=n:return np.concatenate(out)[:n],1-total/drawn
    raise ValueError('Physical sampling acceptance too low')

def weighted_ks(x,w,y):
    i=np.argsort(x);x=x[i];w=w[i]/w.sum();y=np.sort(y);F=np.cumsum(w);G=np.searchsorted(y,x,side='right')/len(y)
    j=np.searchsorted(x,y,side='right');H=np.concatenate(([0.],F))[j];U=np.arange(1,len(y)+1)/len(y)
    return float(max(abs(F-G).max(),abs(F-w-G).max(),abs(H-U).max(),abs(H-U+1/len(y)).max()))

def validate(x,w,f,n=40000):
    y,reject=sample(f,n);w=w/w.sum();ks=[weighted_ks(x[:,k],w,y[:,k]) for k in range(3)]
    # Correlation-sensitive projected CDF checks in standardized coordinates.
    mean=w@x;sd=np.sqrt(w@((x-mean)**2));a=(x-mean)/sd;b=(y-mean)/sd
    vec=np.random.default_rng(902).normal(size=(12,3));vec/=np.linalg.norm(vec,axis=1)[:,None]
    proj=[weighted_ks(a@v,w,b@v) for v in vec]
    reasons=[]
    if max(ks)>.05:reasons.append('marginal_KS_gt_0.05')
    if max(proj)>.07:reasons.append('projected_KS_gt_0.07')
    if 1/(w@w)<1000:reasons.append('ESS_lt_1000')
    if f['gradient_max']>1e-6 or f.get('conditioned_gradient_max',0)>1e-6:reasons.append('optimizer_moment_equations_unresolved')
    if reject>.01 and f.get('schema')!='prior-folded-joint-quadratic-mixture-v1':reasons.append('other_bounds_affect_gt_1pct')
    return {'source_weight_ess':float(1/(w@w)),'marginal_ks':ks,'projected_ks':proj,'other_bound_rejection_fraction':reject,
            'status':'pass_screen' if not reasons else 'requires_fallback','reasons':reasons,
            'criteria':'exploratory shape screens; not calibrated p-values or population-error guarantees'}
