"""Positive aligned 4D likelihood mixtures and explicit delta-chart proposals.

The fitted density is a normalized common-baseline F=L*rho0 density in the
declared eta or source component-mass reference measure. Physical mass/logz
chart Jacobians and rho0 are divided out when evaluating scalar L. Proposal banks integrate in source component
masses, chi and z, and contain no cosmological redshift measure.
"""
import numpy as np
from scipy.special import logsumexp

SCHEMA = 'aligned-redshift-common-baseline-mass-mixture-v3'
ETA_SCHEMA = 'aligned-redshift-common-baseline-bounded-mixture-v2'
LEGACY_SCHEMA = 'aligned-source-redshift-common-baseline-bounded-mixture-v1'


def rho0(z):
    return _common_cosmology().differential_comoving_volume(np.asarray(z,float)).value*4*np.pi/1e9/(1+np.asarray(z,float))


from functools import lru_cache
@lru_cache(maxsize=1)
def _common_cosmology():
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=67.9,Om0=.3065,Tcmb0=0)


def validate_model(model):
    expected=dict(kind='source_time',H0=67.9,Om0=.3065,Tcmb0=0)
    if model.get('schema')not in [SCHEMA,ETA_SCHEMA,LEGACY_SCHEMA] or model.get('baseline_redshift_measure')!=expected or model.get('rho_embedded_in_scalar_likelihood')is not False:
        raise ValueError('Model does not declare the supported common-baseline likelihood quotient')
    chart=model.get('conditional_mass_chart','source'if model.get('schema')==LEGACY_SCHEMA else None)
    if model.get('schema')==LEGACY_SCHEMA and chart!='source':raise ValueError('Legacy schema only declares the source-Mc chart')
    if chart not in ['source','detector']or any(c.get('conditional_mass_chart','source')!=chart for c in model['components']):raise ValueError('Unspecified or inconsistent conditional mass chart')
    bounded=model.get('bounded_mass_coordinate','eta'if model.get('schema')in[ETA_SCHEMA,LEGACY_SCHEMA]else None)
    if bounded not in ['eta','delta']or any(c.get('bounded_mass_coordinate','eta')!=bounded for c in model['components']):raise ValueError('Unspecified or inconsistent bounded mass coordinate')
    if (model.get('schema')==SCHEMA)!=(bounded=='delta'):raise ValueError('Bounded mass coordinate disagrees with model schema')
    p=np.asarray(model['component_weights'],float)
    if p.shape!=(len(model['components']),)or not np.isfinite(p).all()or np.any(p<=0)or not np.isclose(p.sum(),1.,atol=1e-10):
        raise ValueError('Invalid joint mixture weights')


def points(x):
    a=np.asarray(x,float)
    if a.shape[-1]!=4:raise ValueError('Expected Mc_source, eta, chi_eff, z')
    return a


def masses_from_x(x):
    a=points(x);eta=a[...,1];total=a[...,0]*eta**(-.6);delta=np.sqrt(np.maximum(1-4*eta,0))
    return np.stack([total*(1+delta)/2,total*(1-delta)/2,a[...,2],a[...,3]],axis=-1)


def support(x,model):
    a=points(x);mc,eta,chi,z=np.moveaxis(a,-1,0)
    b=model.get('analysis_support',{});amax=float(b.get('amax',.99))
    good=np.isfinite(a).all(-1)&(mc>0)&(eta>0)&(eta<=.25)&(abs(chi)<=amax)&(z>0)
    safe_eta=np.where(good,eta,.2);safe_mc=np.where(good,mc,20.)
    delta=np.sqrt(np.maximum(1-4*safe_eta,0));total=safe_mc*safe_eta**(-.6)
    m1=total*(1+delta)/2;m2=total*(1-delta)/2;q=m2/m1
    good&=abs(chi)<=(float(b.get('a1max',amax))+q*float(b.get('a2max',amax)))/(1+q)
    values={'mc':mc,'mc_det':mc*(1+z),'m1_det':m1*(1+z),'m2_det':m2*(1+z),'mtotal_det':total*(1+z),'q':q,'z':z}
    for name,value in values.items():
        lo=b.get(name+'_min');hi=b.get(name+'_max')
        if lo is not None:good&=value>=float(lo)
        if hi is not None:good&=value<=float(hi)
    return good


def bounded_axis(eta,c):
    return np.sqrt(np.maximum(1-4*np.asarray(eta),0))/4 if c.get('bounded_mass_coordinate','eta')=='delta'else eta


def bounded_logpdf(eta,chi,c):
    uv=(np.stack([bounded_axis(eta,c),chi],axis=-1)-np.asarray(c['bounded_center']))/np.asarray(c['bounded_scale']);u,v=np.moveaxis(uv,-1,0)
    a,b,aa,ab,bb=c['natural_parameters']
    value=a*u+b*v+aa*u*u+ab*u*v+bb*v*v-c['bounded_log_integral_standardized']-np.log(np.prod(c['bounded_scale']))
    return value-np.log(4.)if c.get('bounded_mass_coordinate','eta')=='delta'else value


def conditional(eta,chi,c):
    uv=(np.stack([bounded_axis(eta,c),chi],axis=-1)-np.asarray(c['bounded_center']))/np.asarray(c['bounded_scale'])
    return np.asarray(c['joint_conditional_intercept'])+uv@np.asarray(c['joint_conditional_slopes']),np.asarray(c['joint_conditional_cov'])


def conditional_logpdf(mc,logz,eta,chi,c):
    mean,cov=conditional(eta,chi,c)
    mass=mc*(1+np.exp(logz))if c.get('conditional_mass_chart','source')=='detector'else mc
    r=np.stack([mass,logz],axis=-1)-mean
    return -.5*np.einsum('...i,ij,...j->...',r,np.linalg.inv(cov),r)-np.log(2*np.pi)-.5*np.linalg.slogdet(cov)[1]


def component_logpdf(x,c):
    a=points(x);eta,chi=a[...,1],a[...,2]
    return bounded_logpdf(eta,chi,c)+conditional_logpdf(a[...,0],np.log(a[...,3]),eta,chi,c)


def log_chart_density(x,model,physical=True):
    validate_model(model)
    a=points(x);shape=a.shape[:-1];flat=a.reshape(-1,4)
    good=support(flat,model)if physical else np.isfinite(flat).all(1)&(flat[:,3]>0)
    out=np.full(len(flat),-np.inf)
    if good.any():
        p=np.asarray(model['component_weights'],float)
        out[good]=logsumexp(np.stack([component_logpdf(flat[good],c)+np.log(w)for c,w in zip(model['components'],p)]),axis=0)
    return out.reshape(shape)


def log_likelihood_x(x,model):
    a=points(x)
    z=np.where(a[...,3]>0,a[...,3],1.)
    value=log_chart_density(a,model)-np.log(z)-np.log(rho0(z))
    if model.get('conditional_mass_chart','source')=='detector':value+=np.log1p(z)
    if model.get('bounded_mass_coordinate','eta')=='delta':
        safe_mc=np.where(a[...,0]>0,a[...,0],1.);safe_eta=np.where(a[...,1]>0,a[...,1],.2)
        value-=np.log(safe_mc)-1.2*np.log(safe_eta)-np.log(2.)
    return value


def log_mass_jacobian(x):
    a=points(x)
    return np.log(a[...,0])-1.2*np.log(a[...,1])-.5*np.log1p(-4*a[...,1])


def proposal_tables(model,grid_size=384,defensive_fraction=.0005):
    validate_model(model)
    result=[]
    for c in model['components']:
        amax=float(model['analysis_support'].get('amax',.99))
        if c.get('bounded_mass_coordinate','eta')=='delta':dlo,dhi=c['fit_bounded_mass_bounds']
        else:
            lo,hi=c['fit_eta_bounds'];dlo,dhi=np.sqrt(max(0,1-4*hi)),np.sqrt(max(0,1-4*lo))
        de=np.linspace(dlo,dhi,grid_size+1)
        ce=np.linspace(-amax,amax,grid_size+1)
        D,C=np.meshgrid((de[:-1]+de[1:])/2,(ce[:-1]+ce[1:])/2,indexing='ij')
        lp=bounded_logpdf((1-D*D)/4,C,c).ravel();prob=(1-defensive_fraction)*np.exp(lp-logsumexp(lp))+defensive_fraction/len(lp)
        area=(de[1]-de[0])*(ce[1]-ce[0])
        result.append((de,ce,prob,np.log(prob)-np.log(area)))
    return result


def proposal_logpdf(x,model,tables,weights):
    a=points(x);eta,chi=a[:,1],a[:,2];delta=np.sqrt(np.maximum(1-4*eta,0));logz=np.log(a[:,3]);values=[]
    for c,(de,ce,prob,logcell),weight in zip(model['components'],tables,weights):
        good=(delta>=de[0])&(delta<=de[-1])&(chi>=ce[0])&(chi<=ce[-1])
        di=np.clip(np.searchsorted(de,delta,side='right')-1,0,len(de)-2);ci=np.clip(np.searchsorted(ce,chi,side='right')-1,0,len(ce)-2)
        lp=np.log(weight)+logcell[di*(len(ce)-1)+ci]+conditional_logpdf(a[:,0],logz,eta,chi,c)
        values.append(np.where(good,lp,-np.inf))
    return logsumexp(np.stack(values),axis=0)


def integration_bank(model,n=32768,seed=73,grid_size=384):
    """Known proposal in Mc,delta,chi,logz; returns source-mass integration weights.

    Globally rejecting native support changes the proposal by one event constant,
    omitted from shape-only likelihoods. No rho(z) enters these weights.
    """
    validate_model(model)
    tables=proposal_tables(model,grid_size);p=np.asarray(model['component_weights'],float);p=.9999*p/p.sum()+.0001/len(p)
    rng=np.random.default_rng(seed);accepted=[];count=drawn=0
    while count<n:
        k=max(2048,2*(n-count));labels=rng.choice(len(p),k,p=p);x=np.empty((k,4))
        for j,c in enumerate(model['components']):
            ids=np.flatnonzero(labels==j)
            if not len(ids):continue
            de,ce,prob,_=tables[j];cell=rng.choice(len(prob),len(ids),p=prob)
            di,ci=cell//(len(ce)-1),cell%(len(ce)-1)
            delta=de[di]+rng.random(len(ids))*(de[1]-de[0]);chi=ce[ci]+rng.random(len(ids))*(ce[1]-ce[0]);eta=(1-delta*delta)/4
            mean,cov=conditional(eta,chi,c);y=mean+rng.normal(size=(len(ids),2))@np.linalg.cholesky(cov).T
            z=np.exp(y[:,1]);mc=y[:,0]/(1+z)if c.get('conditional_mass_chart','source')=='detector'else y[:,0]
            x[ids]=np.column_stack([mc,eta,chi,z])
        good=support(x,model);accepted.append(x[good]);count+=int(good.sum());drawn+=k
        if drawn>max(100000,100*n):raise RuntimeError('Joint proposal native-support acceptance is unresolved')
    x=np.vstack(accepted)[:n];logq=proposal_logpdf(x,model,tables,p)
    # z and, for detector Mc, (1+z) cancel between scalar L and the mass/redshift chart Jacobian. Mc here is SOURCE Mc.
    logw=log_chart_density(x,model)-logq-np.log(rho0(x[:,3]))-np.log(n)
    if model.get('bounded_mass_coordinate','eta')=='eta':logw+=np.log(x[:,0])-1.2*np.log(x[:,1])-np.log(2.)
    if not np.isfinite(logw).all():raise RuntimeError('Nonfinite joint integration weights')
    logw-=logsumexp(logw)  # fixed event constant; population shape is unchanged
    chart=model.get('conditional_mass_chart','source')
    bounded=model.get('bounded_mass_coordinate','eta')
    scalar='q_chart*(1+z)/(z*rho0)'if chart=='detector'else'q_chart/(z*rho0)'
    if bounded=='delta':scalar=scalar.replace('z*rho0','J_delta*z*rho0')
    meta=dict(n=n,seed=seed,grid_size=grid_size,physical_acceptance=count/drawn,proposal='Piecewise-uniform delta/chi grid plus exact conditional Gaussian mass/logz; global support rejection',conditional_mass_chart=chart,bounded_mass_coordinate=bounded,density_chart=('Mc_detector'if chart=='detector'else'Mc_source')+','+bounded+',chi_eff,logz',scalar_likelihood=scalar,integration_measure='dm1_source dm2_source dchi_eff dz',rho_embedded=False,baseline_density=('F_t=L*rho0*J_delta*z'if bounded=='delta'else'F_t=L*rho0*z')+('/(1+z)'if chart=='detector'else''),baseline_redshift_measure=dict(kind='source_time',H0=67.9,Om0=.3065,Tcmb0=0),event_normalization='Arbitrary constant fixed once per event; no absolute evidence')
    return masses_from_x(x),logw,x,meta
