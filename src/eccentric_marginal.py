"""Stable optional 4D likelihood view; native-z measure explicit.

Integrate full5D L against declared p_z(z), retaining actual joint posterior
correlations. Component conditioning and CDF quadrature resolve sharp z
structure and exact coupled mass cuts. This view is not the native marginal
posterior nor a likelihood marginalized under its conditional z prior.
"""
import numpy as np
from scipy.special import log_ndtr,ndtri_exp,logsumexp,expit
from numpy.polynomial.legendre import leggauss

def _logdiff(a,b):
 # log(exp(a)-exp(b)), with a>=b; equal tiny tails give zero mass.
 if b>=a:return -np.inf
 return a+np.log(-np.expm1(b-a))

def _conditional_expectation(mu,sig,tlo,thi,zmin,zmax,nodes,weights):
 a=(tlo-mu)/sig;b=(thi-mu)/sig
 if a>0:
  la=float(log_ndtr(-a));lb=float(log_ndtr(-b));lp=_logdiff(la,lb)
  if not np.isfinite(lp):return -np.inf
  # SF(t)=SF(a)-u*[SF(a)-SF(b)], evaluated entirely in log space.
  logsf=la+np.log1p(-nodes*(-np.expm1(lb-la)));standard=-ndtri_exp(logsf)
 else:
  la=float(log_ndtr(a));lb=float(log_ndtr(b));lp=_logdiff(lb,la)
  if not np.isfinite(lp):return -np.inf
  logcdf=np.logaddexp(la,np.log(nodes)+lp);standard=ndtri_exp(logcdf)
 z=zmin+(zmax-zmin)*expit(mu+sig*standard);expectation=np.dot(weights,(1+z)**(-2));return lp+np.log(expectation)

def log_likelihood_marginal4(x4,f,order=96):
 from eccentric_likelihood import _fit,q_from_eta,forward_coordinates,conditional_chi_prior
 f=_fit(f);x4=np.atleast_2d(x4);out=np.full(len(x4),-np.inf);s=f['support'];g=f['density'];nodes,weights=leggauss(order);nodes=(nodes+1)/2;weights=weights/2;others=[0,1,2,3]
 for index,row in enumerate(x4):
  mc,eta,chi,e=row;q=float(q_from_eta(eta))
  if not(np.isfinite(row).all()and mc>0 and 0<eta<=.25 and s['q_min']<q<s['q_max']and abs(chi)<s['amax']and s['ecc_min']<=e<=s['ecc_max']):continue
  mone=mc*(1+q)**.2*q**(-.6);mtwo=q*mone;lo=max(s['z_min'],s['mc_det_min']/mc-1,s['m1_det_min']/mone-1,s['m2_det_min']/mtwo-1);hi=min(s['z_max'],s['mc_det_max']/mc-1,s['m1_det_max']/mone-1,s['m2_det_max']/mtwo-1)
  if hi<=lo:continue
  def tz(z):
   if z<=s['z_min']:return -np.inf
   if z>=s['z_max']:return np.inf
   return np.log((z-s['z_min'])/(s['z_max']-z))
  tlo,thi=tz(lo),tz(hi);t=forward_coordinates(np.r_[row,(lo+hi)/2],f)[0];dqdt=(q-s['q_min'])*(s['q_max']-q)/(s['q_max']-s['q_min']);dchidt=s['amax']*(1-(chi/s['amax'])**2);jmcq=mc*(1+q)**.4*q**(-1.2);pc=conditional_chi_prior(np.array([chi]),np.array([q]),s['amax'],f['priors']['spin_regularizer'])[0];pref=-np.log(mc)-np.log(dqdt)-np.log(dchidt)-np.log(jmcq)-np.log(pc)-np.log(f.get('posterior_support_probability',1));terms=[]
  for w,mean,cov in zip(g['weights'],np.asarray(g['means']),np.asarray(g['covariances'])):
   c4=cov[np.ix_(others,others)];L=np.linalg.cholesky(c4);cross=np.linalg.solve(c4,cov[others,4]);sigma=np.sqrt(max(cov[4,4]-cov[4,others]@cross,1e-15))
   for sign in [1,-1]:
    t4=t[others].copy();t4[3]*=sign;delta=t4-mean[others];standard=np.linalg.solve(L,delta);lmarg=-.5*(4*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+standard@standard);mu=mean[4]+cov[4,others]@np.linalg.solve(c4,delta);le=_conditional_expectation(mu,sigma,tlo,thi,s['z_min'],s['z_max'],nodes,weights)
    terms.append(np.log(w)+lmarg+le)
  out[index]=pref+logsumexp(terms)
 return out
