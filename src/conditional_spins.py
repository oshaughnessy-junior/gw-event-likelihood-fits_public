"""Sample isotropic, uniform-magnitude spins conditioned on (chi_eff, q).

Quadrature resolves the logarithmic projected-spin densities on their exact
support. Conditioning before sampling avoids inverse-spin-prior tail weights.
Order is configurable; compare orders for the intended application.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def sample_conditioned(chi, q, seed=1, amax=.99, order=64):
    chi,q=np.broadcast_arrays(np.asarray(chi,float),np.asarray(q,float))
    chi=chi.ravel();q=q.ravel()
    if not np.isfinite(chi).all() or not np.isfinite(q).all() or np.any((q<=0)|(q>1)) or np.any(abs(chi)>=amax):
        raise ValueError('Require 0<q<=1 and |chi_eff|<amax')
    rng=np.random.default_rng(seed);u=(1+q)*chi
    lo=np.maximum(-amax,u-q*amax);hi=np.minimum(amax,u+q*amax)
    cuts=np.sort(np.stack([lo,hi,np.clip(np.zeros_like(lo),lo,hi),np.clip(u,lo,hi)],axis=1),axis=1)
    nodes,weights=leggauss(order);zs=[];ws=[]
    for j in range(3):
        mid=(cuts[:,j]+cuts[:,j+1])/2;width=(cuts[:,j+1]-cuts[:,j])/2
        z=mid[:,None]+width[:,None]*nodes
        z2=(u[:,None]-z)/q[:,None]
        value=np.log(amax/np.maximum(abs(z),1e-300))*np.log(amax/np.maximum(abs(z2),1e-300))
        zs.append(z);ws.append(np.maximum(value*width[:,None]*weights,0))
    z=np.concatenate(zs,axis=1);w=np.concatenate(ws,axis=1)
    if np.any(w.sum(axis=1)<=0):raise ValueError('Degenerate conditional-spin quadrature')
    cdf=np.cumsum(w,axis=1);cdf/=cdf[:,-1,None]
    index=(cdf<rng.random((len(chi),1))).sum(axis=1)
    z1=z[np.arange(len(chi)),np.minimum(index,z.shape[1]-1)];z2=(u-z1)/q
    out=[]
    for zz in [z1,z2]:
        # p(a|s_z) proportional to 1/a on |s_z|<a<amax.
        aa=np.exp(np.log(np.maximum(abs(zz),1e-300))+rng.random(len(chi))*np.log(amax/np.maximum(abs(zz),1e-300)))
        phi=rng.uniform(0,2*np.pi,len(chi));r=np.sqrt(np.maximum(aa*aa-zz*zz,0))
        out.extend([r*np.cos(phi),r*np.sin(phi),zz])
    return np.column_stack(out)
