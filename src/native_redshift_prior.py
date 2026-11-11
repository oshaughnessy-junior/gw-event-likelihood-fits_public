"""Native distance/spin prior utilities for source-Mc/eta/chi/redshift models.

All densities are unnormalized by an event-wise constant. No population redshift
law is included: the native distance prior is removed, not replaced here.
"""
import re
import numpy as np


def text(v):return ' '.join(map(str,v))if isinstance(v,list)else str(v)


def number(value,key):
    m=re.search(r'\b'+re.escape(key)+r'\s*=\s*([-+0-9.eE]+)',text(value))
    if not m:raise ValueError('Missing prior parameter '+key+': '+text(value))
    return float(m.group(1))


def common_cosmology():
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=67.9,Om0=.3065,Tcmb0=0)


def distance_cosmology(spec):
    from astropy import cosmology as ac
    from astropy.cosmology import FlatLambdaCDM
    if spec.get('kind')=='common_source_time':return common_cosmology()
    s=spec['analytic'];m=re.search(r"cosmology=['\"]([A-Za-z0-9_]+)['\"]",s)
    if m:
        if m.group(1)=='Planck15_LAL':return common_cosmology()
        if m.group(1)in ['Planck15','Planck18','WMAP9','WMAP7','WMAP5']:return getattr(ac,m.group(1))
    if 'LambdaCDM('in s:
        if [number(s,k)for k in ['H0','Om0','Ode0','Tcmb0']]==[67.9,.3065,.6935,0.]:return common_cosmology()
    raise ValueError('Unverified native distance cosmology '+s)


def inverse_distance(dl,cosmology):
    from scipy.interpolate import PchipInterpolator
    dl=np.asarray(dl,float);hi=1.
    while float(cosmology.luminosity_distance(hi).value)<float(dl.max()):hi*=2
    z=np.linspace(0,hi,50001)
    return PchipInterpolator(cosmology.luminosity_distance(z).value,z)(dl)


def dlprime(z,cosmology):
    return cosmology.luminosity_distance(z).value/(1+z)+(1+z)*299792.458/cosmology.H(z).value


def log_distance_density(dl,spec):
    if spec['kind']=='distance_squared':return 2*np.log(dl)
    c=distance_cosmology(spec);z=inverse_distance(dl,c)
    value=np.log(c.differential_comoving_volume(z).value)-np.log(dlprime(z,c))
    if spec['kind']in ['UniformSourceFrame','common_source_time']:value-=np.log1p(z)
    elif spec['kind']!='UniformComovingVolume':raise ValueError(spec)
    return value


def source_coordinates(x):
    mc,eta,chi,z=np.asarray(x,float).T;delta=np.sqrt(np.maximum(0,1-4*eta));total=mc*eta**(-.6)
    return total*(1+delta)/2,total*(1-delta)/2,chi,z


def support(x,contract,tolerance=1e-7):
    x=np.asarray(x,float);mc,eta,chi,z=x.T;b=contract['analysis_support']
    m1,m2,_,_=source_coordinates(x);q=m2/m1
    mask=np.isfinite(x).all(1)&(mc>0)&(eta>0)&(eta<=.25)&(z>=0)&(abs(chi)<=b['amax']+tolerance)
    values={'mc_det':mc*(1+z),'q':q,'m1_det':m1*(1+z),'m2_det':m2*(1+z),'mtotal_det':(m1+m2)*(1+z),'z':z}
    for name,v in values.items():
        for suffix,sign in [('min',1),('max',-1)]:
            cut=b.get(name+'_'+suffix)
            if cut is not None:mask&=sign*(v-cut)>=-tolerance*np.maximum(1,abs(cut))
    return mask


def log_native_prior(x,contract,chi_prior_function,order=None):
    x=np.asarray(x,float);mc,eta,chi,z=x.T;m1,m2,_,_=source_coordinates(x);q=m2/m1
    order=contract.get('spin_quadrature_order',128)if order is None else order
    c=common_cosmology();dl=c.luminosity_distance(z).value
    a1,a2=contract['spin_caps'];pchi=np.concatenate([chi_prior_function(chi[k:k+4096],q[k:k+4096],a1,a2,order=order)for k in range(0,len(x),4096)])
    result=2*np.log1p(z)+np.log(mc)-1.2*np.log(eta)-.5*np.log1p(-4*eta)+np.log(pchi)+log_distance_density(dl,contract['distance_prior'])+np.log(dlprime(z,c))
    return np.where(support(x,contract),result,-np.inf)
