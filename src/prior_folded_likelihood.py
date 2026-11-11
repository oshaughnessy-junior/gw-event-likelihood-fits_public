"""Evaluate experimental prior-folded Gaussian likelihood mixtures.

The posterior fractions alpha and likelihood amplitudes alpha/Z are distinct.
This module evaluates the latter. Flat sampling uses normalized whole-space
Gaussian integrals followed by exact rejection of the declared finite support.
It never returns forward-prior posterior samples through the flat-sample API.
"""
import numpy as np
from scipy.special import logsumexp,log_ndtr
from scipy.stats import truncnorm


def features(x,f):
 y=(np.asarray(x,float)-f['center'])/f['scale']
 return np.stack([y[...,0],y[...,1],y[...,2],y[...,0]**2,y[...,1]**2,y[...,2]**2,y[...,0]*y[...,1],y[...,0]*y[...,2],y[...,1]*y[...,2]],axis=-1)


def support(x,b):
 a=np.asarray(x,float);shape=a.shape[:-1];a=a.reshape(-1,3);mc,e,c=a.T
 good=np.isfinite(a).all(1)&(mc>=b['mc_min'])&(mc<=b['mc_max'])&(e>0)&(e<=.25)&(abs(c)<=b['amax'])
 q=np.zeros(len(a));q[good]=2*e[good]/(1-2*e[good]+np.sqrt(1-4*e[good]));good&=(q>=b['q_min'])&(q<=b['q_max'])
 i=np.flatnonzero(good)
 if len(i):
  m1=mc[i]*(1+q[i])**.2/q[i]**.6;m2=m1*q[i]
  low=np.maximum.reduce([np.full(len(i),b['z_min']),b['mc_det_min']/mc[i]-1,b['m1_det_min']/m1-1,b['m2_det_min']/m2-1])
  high=np.minimum.reduce([np.full(len(i),b['z_max']),b['mc_det_max']/mc[i]-1,b['m1_det_max']/m1-1,b['m2_det_max']/m2-1])
  if 'mtotal_det_min' in b:
   total=m1+m2;low=np.maximum(low,b['mtotal_det_min']/total-1);high=np.minimum(high,b['mtotal_det_max']/total-1)
  good[i]=low<high
 return good.reshape(shape)


def check_amplitudes(f):
 expected=np.log(f['posterior_component_weights'])-np.array([c['forward_log_normalizer']for c in f['components']])
 if not np.allclose(f['log_likelihood_amplitudes'],expected,rtol=0,atol=1e-10):
  raise ValueError('Likelihood amplitudes must be posterior fractions divided by forward normalizers')


def log_shape(x,f,physical=True):
 check_amplitudes(f);t=features(x,f)
 lp=logsumexp(np.stack([t@c['theta']+a for c,a in zip(f['components'],f['log_likelihood_amplitudes'])]),axis=0)
 # Analysis support is part of this target even when generic physical=False.
 return np.where(support(x,f['bounds']),lp,-np.inf)


def gaussian_parameters(c,f):
 t=np.asarray(c['theta']);P=np.diag(-2*t[3:6]);P[0,1]=P[1,0]=-t[6];P[0,2]=P[2,0]=-t[7];P[1,2]=P[2,1]=-t[8]
 if np.linalg.eigvalsh(P).min()<=0:raise ValueError('Flat sampling requires positive definite Gaussian likelihood precision')
 C=np.linalg.inv(P);v=C@t[:3];scale=np.asarray(f['scale']);mu=np.asarray(f['center'])+scale*v
 cov=C*np.outer(scale,scale)
 log_volume=.5*t[:3]@v+1.5*np.log(2*np.pi)-.5*np.linalg.slogdet(P)[1]+np.log(scale).sum()
 return mu,cov,float(log_volume)


def log_normal_interval(a,b):
 """Stable probability of a Gaussian interval, including extreme tails."""
 if a>=b:return -np.inf
 if a>0:
  lo=log_ndtr(-a);hi=log_ndtr(-b)
 else:
  lo=log_ndtr(b);hi=log_ndtr(a)
 return float(lo+np.log(-np.expm1(hi-lo)))


def sample(f,n=40000,seed=1):
 """Exact eta-conditioned Gaussian mixture followed by global support rejection.

Known Gaussian eta interval masses enter component proposal weights. All
remaining support rejection occurs AFTER mixing; no physical fraction is
mistaken for a likelihood mixture fraction. Rejection is exact conditioning,
not a missing normalization correction in the fitted model.
 """
 check_amplitudes(f);pars=[gaussian_parameters(c,f)for c in f['components']]
 b=f['bounds'];elo=b['q_min']/(1+b['q_min'])**2;ehi=b['q_max']/(1+b['q_max'])**2
 lm=[];conditionals=[]
 for j,(mu,cov,volume)in enumerate(pars):
  se=np.sqrt(cov[1,1]);aa=(elo-mu[1])/se;bb=(ehi-mu[1])/se
  lm.append(f['log_likelihood_amplitudes'][j]+volume+log_normal_interval(aa,bb))
  yi=[0,2];slope=cov[yi,1]/cov[1,1];R=cov[np.ix_(yi,yi)]-np.outer(cov[yi,1],cov[1,yi])/cov[1,1]
  conditionals.append((mu,se,aa,bb,slope,R))
 probs=np.exp(np.array(lm)-logsumexp(lm));rng=np.random.default_rng(seed);ret=[];accepted=0;drawn=0
 for _ in range(100):
  count=max(2*n,1000);labels=rng.choice(len(pars),count,p=probs);x=np.empty((count,3))
  for j,(mu,se,aa,bb,slope,R)in enumerate(conditionals):
   mask=labels==j;nn=int(mask.sum())
   if not nn:continue
   eta=truncnorm.rvs(aa,bb,loc=mu[1],scale=se,size=nn,random_state=rng)
   y=mu[[0,2]]+(eta-mu[1])[:,None]*slope+rng.multivariate_normal(np.zeros(2),R,nn)
   x[mask]=np.column_stack([y[:,0],eta,y[:,1]])
  keep=support(x,f['bounds']);ret.append(x[keep]);accepted+=int(keep.sum());drawn+=count
  if accepted>=n:return np.concatenate(ret)[:n],1-accepted/drawn
 raise ValueError('Flat likelihood sampling acceptance too low; no forward-prior samples substituted')


def _chi_prior(chi,q,amax,order=64):
 from numpy.polynomial.legendre import leggauss
 chi=np.asarray(chi);q=np.asarray(q);u=(1+q)*chi
 lo=np.maximum(-amax,u-q*amax);hi=np.minimum(amax,u+q*amax)
 cuts=np.sort(np.stack([lo,hi,np.clip(np.zeros_like(lo),lo,hi),np.clip(u,lo,hi)],axis=1),axis=1)
 nodes,weights=leggauss(order);out=np.zeros(len(chi))
 for j in range(3):
  mid=(cuts[:,j]+cuts[:,j+1])/2;width=(cuts[:,j+1]-cuts[:,j])/2
  z=mid[:,None]+width[:,None]*nodes
  p=np.log(amax/np.maximum(abs(z),1e-300))*np.log(amax/np.maximum(abs((u[:,None]-z)/q[:,None]),1e-300))
  out+=width*(p@weights)
 return np.maximum(out*(1+q)/(4*amax**2*q),0)


def intrinsic_logprior(x,f):
 """Unnormalized proper retained prior on the bounding Mc/q/chi box.

Its scalar normalization cancels posterior sampling. The tighter original PE
union support is carried by log_shape, not counted twice in this density.
 """
 b=f['bounds'];x=np.asarray(x);mc,e,c=x.T;out=np.full(len(x),-np.inf)
 ok=(mc>=b['mc_min'])&(mc<=b['mc_max'])&(e>0)&(e<.25)&(abs(c)<b['amax'])
 i=np.flatnonzero(ok);q=2*e[i]/(1-2*e[i]+np.sqrt(1-4*e[i]));okq=(q>=b['q_min'])&(q<=b['q_max']);i=i[okq];q=q[okq]
 if len(i):
  pc=np.concatenate([_chi_prior(c[i[j:j+10000]],q[j:j+10000],b['amax'])for j in range(0,len(i),10000)])
  out[i]=np.log(mc[i])-1.2*np.log(e[i])-.5*np.log(1-4*e[i])+np.log(pc)
 return out


def sample_forward(f,n=40000,seed=1,integration_samples=262144,modifier='none'):
 """Importance-sample L*pi_retained with true integration ESS diagnostics.

The fixed proposal includes full-box prior and full-box uniform strata; it is
not a bootstrap of the source reference. Returned samples are model posterior
samples under the declared forward prior, never flat-likelihood samples.
 """
 from scipy.integrate import quad,cumulative_trapezoid
 from scipy.stats import multivariate_normal
 if modifier not in ['none','flat_retained_coordinates','spin_gaussian']:raise ValueError('Unknown forward-prior modifier')
 b=f['bounds'];rng=np.random.default_rng(seed);nn=integration_samples;mu=np.array(f['forward_proposal_mean']);cov=np.array(f['forward_proposal_cov'])
 labels=rng.choice(4,nn,p=[.6,.2,.1,.1]);x=np.empty((nn,3))
 for j,scale in enumerate([1,4]):
  mask=labels==j;x[mask]=rng.multivariate_normal(mu,cov*scale,int(mask.sum()))
 mask=labels==2;count=int(mask.sum());qg=np.geomspace(b['q_min'],b['q_max'],30001);density=qg**(-1.2)*(1+qg)**.4
 cq=np.r_[0,cumulative_trapezoid(density,qg)];cq/=cq[-1];q=np.interp(rng.random(count),cq,qg)
 mc=np.sqrt(rng.uniform(b['mc_min']**2,b['mc_max']**2,count));a1=rng.random(count)*b['amax'];a2=rng.random(count)*b['amax']
 chi=(a1*rng.uniform(-1,1,count)+q*a2*rng.uniform(-1,1,count))/(1+q);x[mask]=np.column_stack([mc,q/(1+q)**2,chi])
 mask=labels==3;count=int(mask.sum());q=rng.uniform(b['q_min'],b['q_max'],count)
 x[mask]=np.column_stack([rng.uniform(b['mc_min'],b['mc_max'],count),q/(1+q)**2,rng.uniform(-b['amax'],b['amax'],count)])
 lp=intrinsic_logprior(x,f);constant=.5*(b['mc_max']**2-b['mc_min']**2)*quad(lambda q:q**(-1.2)*(1+q)**.4,b['q_min'],b['q_max'])[0]
 uniform=np.full(nn,-np.inf);ii=np.flatnonzero(np.isfinite(lp));qq=2*x[ii,1]/(1-2*x[ii,1]+np.sqrt(1-4*x[ii,1]))
 uniform[ii]=-np.log((b['mc_max']-b['mc_min'])*(b['q_max']-b['q_min'])*2*b['amax'])-np.log((1-qq)/(1+qq)**3)
 lh=logsumexp(np.stack([np.log(.6)+multivariate_normal.logpdf(x,mu,cov),np.log(.2)+multivariate_normal.logpdf(x,mu,cov*4),np.log(.1)+lp-np.log(constant),np.log(.1)+uniform]),axis=0)
 lw=log_shape(x,f)+lp-lh
 if modifier!='none':lw=np.where((x[:,0]>=.1)&(x[:,0]<=500),lw,-np.inf)
 if modifier=='spin_gaussian':lw-=.5*(x[:,2]/.2)**2
 normalizer=logsumexp(lw);weights=np.exp(lw-normalizer)
 if not np.isfinite(weights).all()or weights.sum()<=0:raise ValueError('Forward integrator found no support')
 effective=float(1/(weights@weights));idx=rng.choice(nn,n,p=weights)
 return x[idx],{'integration_ess':effective,'integration_samples':nn,'output_resample_count':n,
   'output_unique_integration_points':int(len(np.unique(idx))),
   'normalizer_relative_mc_se':float(np.sqrt(max(0,nn/effective-1)/nn)),
   'log_normalizer_up_to_prior_constant':float(normalizer-np.log(nn)),
   'forward_prior':'J(Mc,eta)*pchi(chi|q), finite original PE union support',
   'modifier':modifier,'scope':'Model forward posterior samples, not a flat likelihood or independent PE reference'}
