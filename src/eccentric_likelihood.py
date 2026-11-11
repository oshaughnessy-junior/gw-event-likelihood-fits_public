"""Joint eccentric likelihood as an explicit normalized-density/prior ratio.

Physical API: (Mc_source, eta, chi_eff, eccentricity, redshift).
Each fit retains actual joint redshift, and averages mean anomaly and the
remaining spins under native PE conditional nuisance priors. Event constants
are arbitrary. No generic prior-independent nuisance-marginal claim is made.
"""
import numpy as np
from scipy.special import expit,logit,logsumexp
from numpy.polynomial.legendre import leggauss
COORDINATES=['mc_source','eta','chi_eff','eccentricity','redshift']
SCHEMA='eccentric-joint-density-ratio-v1'

def q_from_eta(eta):
 eta=np.asarray(eta);return 2*eta/(1-2*eta+np.sqrt(np.maximum(1-4*eta,0)))
def _fit(f):return f.get('fit',f)
def support(x,f):
 f=_fit(f);x=np.atleast_2d(x);s=f['support'];mc,eta,chi,e,z=x.T;q=q_from_eta(eta);md=mc*(1+z);mt=md/(q/(1+q)**2)**.6;m1=mt/(1+q);m2=q*m1
 return np.isfinite(x).all(1)&(mc>0)&(eta>0)&(eta<=.25)&(q>s['q_min'])&(q<s['q_max'])&(abs(chi)<s['amax'])&(e>=s['ecc_min'])&(e<=s['ecc_max'])&(z>s['z_min'])&(z<s['z_max'])&(md>=s['mc_det_min'])&(md<=s['mc_det_max'])&(m1>=s.get('m1_det_min',1))&(m1<=s.get('m1_det_max',1000))&(m2>=s.get('m2_det_min',1))&(m2<=s.get('m2_det_max',1000))

def forward_coordinates(x,f):
 f=_fit(f);x=np.atleast_2d(x);mc,eta,chi,e,z=x.T;s=f['support'];a,b=f['mass_shear']['a'],f['mass_shear'].get('b',0);p=f['priors']['eccentricity_power'];v=(e/s['ecc_max'])**p
 return np.column_stack([np.log(mc)+a*e*e+b*e**4,logit((q_from_eta(eta)-s['q_min'])/(s['q_max']-s['q_min'])),np.arctanh(chi/s['amax']),v,logit((z-s['z_min'])/(s['z_max']-s['z_min']))])

def inverse_coordinates(t,f):
 f=_fit(f);s=f['support'];t=np.atleast_2d(t);q=s['q_min']+(s['q_max']-s['q_min'])*expit(t[:,1]);chi=s['amax']*np.tanh(t[:,2]);e=s['ecc_max']*abs(t[:,3])**(1/f['priors']['eccentricity_power']);z=s['z_min']+(s['z_max']-s['z_min'])*expit(t[:,4]);a,b=f['mass_shear']['a'],f['mass_shear'].get('b',0);mc=np.exp(np.clip(t[:,0]-a*e*e-b*e**4,-700,700));return np.column_stack([mc,q/(1+q)**2,chi,e,z])

def _latent_logpdf(t,f):
 g=f['density'];means=np.asarray(g['means']);covs=np.asarray(g['covariances']);weights=np.asarray(g['weights']);out=[]
 for w,m,cov in zip(weights,means,covs):
  delta=t-m;L=np.linalg.cholesky(cov);u=np.linalg.solve(L,delta.T).T;out.append(np.log(w)-.5*(5*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+(u*u).sum(1)))
 return logsumexp(out,axis=0)

def _folded_latent_logpdf(t,f):
 other=t.copy();other[:,3]*=-1;return np.logaddexp(_latent_logpdf(t,f),_latent_logpdf(other,f))

def conditional_chi_prior(chi,q,amax=.99,epsilon=1e-7,order=64):
 """Native alignedspin-zprior convolution, including its source regularizer."""
 chi=np.asarray(chi);q=np.asarray(q);u=(1+q)*chi;lo=np.maximum(-amax,u-q*amax);hi=np.minimum(amax,u+q*amax);cuts=np.sort(np.stack([lo,hi,np.clip(np.zeros_like(lo),lo,hi),np.clip(u,lo,hi)],axis=1),axis=1);nodes,w=leggauss(order);out=np.zeros(len(chi));normalizer=1+epsilon*np.log(epsilon)-epsilon
 for j in range(3):
  mid=(cuts[:,j]+cuts[:,j+1])/2;width=np.maximum((cuts[:,j+1]-cuts[:,j])/2,0);s=mid[:,None]+width[:,None]*nodes;f=np.maximum(-np.log(abs(s)/amax+epsilon),0)*np.maximum(-np.log(abs((u[:,None]-s)/q[:,None])/amax+epsilon),0);out+=width*(f@w)
 return np.maximum(out*(1+q)/(4*amax**2*q*normalizer**2),1e-300)

def _log_prior_z(z,f):
 p=f['priors']['redshift_density'];return np.interp(np.log(z),np.log(p['z']),p['log_density'])

def log_prior(x,f):
 """Declared native prior density in physical Mc_source,eta,chi,e,z measure.
 The absolute component-mass normalizer is arbitrary and cancels eventwise.
 """
 f=_fit(f);x=np.atleast_2d(x);out=np.full(len(x),-np.inf);ok=support(x,f)
 if not ok.any():return out
 mc,eta,chi,e,z=x[ok].T;q=q_from_eta(eta);s=f['support'];p=f['priors']['eccentricity_power'];jac=mc*eta**(-1.2)/np.sqrt(1-4*eta);pe=np.ones(len(e))/s['ecc_max']if p==1 else 2*e/s['ecc_max']**2
 with np.errstate(divide='ignore',invalid='ignore'):out[ok]=np.log(jac)+2*np.log1p(z)+np.log(conditional_chi_prior(chi,q,s['amax'],f['priors']['spin_regularizer']))+np.log(pe)+_log_prior_z(z,f)
 return out

def log_likelihood(x,f):
 """Prior-divided likelihood, with stable analytic eccentric-prior cancellation.
 q/eta Jacobian cancels between posterior density and native component-mass
 prior. Finite e=0 follows from folding the native e-CDF coordinate.
 """
 f=_fit(f);x=np.atleast_2d(x);out=np.full(len(x),-np.inf);ok=support(x,f)
 if not ok.any():return out
 xx=x[ok];mc,eta,chi,e,z=xx.T;q=q_from_eta(eta);s=f['support'];t=forward_coordinates(xx,f);dqdt=(q-s['q_min'])*(s['q_max']-q)/(s['q_max']-s['q_min']);dchidt=s['amax']*(1-(chi/s['amax'])**2);dzdt=(z-s['z_min'])*(s['z_max']-z)/(s['z_max']-s['z_min']);jmcq=mc*(1+q)**.4*q**(-1.2)
 out[ok]=_folded_latent_logpdf(t,f)-np.log(mc)-np.log(dqdt)-np.log(dchidt)-np.log(dzdt)-np.log(jmcq)-2*np.log1p(z)-np.log(conditional_chi_prior(chi,q,s['amax'],f['priors']['spin_regularizer']))-_log_prior_z(z,f)-np.log(f.get('posterior_support_probability',1))
 return out

def logpdf_native_posterior(x,f):
 f=_fit(f);return log_likelihood(x,f)+log_prior(x,f)

def sample(f,n=10000,seed=0,return_diagnostics=False):
 """Samples normalized native posterior; sample weights need no Jacobian."""
 f=_fit(f);g=f['density'];rng=np.random.default_rng(seed);weights=np.array(g['weights']);means=np.array(g['means']);covs=np.array(g['covariances']);rows=[];seen=accepted=0
 while sum(len(x)for x in rows)<n:
  size=max(2000,2*(n-sum(len(x)for x in rows)));k=rng.choice(len(weights),size,p=weights);t=np.empty((size,5))
  for j in range(len(weights)):
   ix=np.flatnonzero(k==j)
   if len(ix):t[ix]=rng.multivariate_normal(means[j],covs[j],len(ix))
  x=inverse_coordinates(t,f);mask=support(x,f);rows.append(x[mask]);seen+=size;accepted+=int(mask.sum())
  if seen>100*max(n,1000):raise ValueError('support acceptance below1%; fit not usable')
 out=np.concatenate(rows)[:n];diag={'proposed':seen,'accepted':accepted,'support_probability':accepted/seen}
 return(out,diag)if return_diagnostics else out

def sample_marginal4(f,n=10000,seed=0):
 """Native posterior 4D marginal, preserving original joint-z correlations."""
 return sample(f,n,seed)[:,:4]

def log_likelihood_marginal4(x4,f,order=96):
    """Explicit native-p_z likelihood view, not native posterior marginal.
    Stable conditional-Gaussian CDF quadrature respects coupled mass bounds.
    The native conditional prior at fixed source coordinates also contains
    (1+z)^2 and coupled mass cuts; it is a different measure.
    """
    from eccentric_marginal import log_likelihood_marginal4 as integrate
    return integrate(x4,f,order)
