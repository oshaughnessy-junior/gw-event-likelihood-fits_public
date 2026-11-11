"""Native-posterior 4D leaf with a conditional-spin-CDF coordinate.

v=(Mc_detector,delta,s,logz), s=2a F_native(chi_eff|q)-a.
The native retained prior in v is J_detector_delta*z*p_DL*dDLdz/(2a).
Its quotient therefore has a finite spin-boundary limit; no 1/p_chi factor
is applied pointwise to the scalar likelihood.
"""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from aligned_redshift import component_logpdf,conditional,conditional_logpdf,support,masses_from_x
from native_redshift_prior import common_cosmology,dlprime,log_distance_density

SCHEMA='aligned-redshift-native-spin-cdf-posterior-v1'


@lru_cache(maxsize=8)
def rule(order):return np.polynomial.legendre.leggauss(order)


def projected_cdf(s,a):
    s=np.asarray(s,float);u=np.clip(s,-a,a);v=np.maximum(abs(u),1e-300)
    return np.clip(.5+u/(2*a)*(np.log(a/v)+1),0.,1.)


def projected_pdf(s,a):
    s=np.asarray(s,float)
    return np.where(abs(s)<a,np.log(a/np.maximum(abs(s),1e-300))/(2*a),0.)


def native_cdf(chi,q,a1,a2,order=64):
    """Conditional CDF, with the leading log singularity integrated analytically."""
    chi,q=np.broadcast_arrays(np.asarray(chi,float),np.asarray(q,float));shape=chi.shape;chi=chi.ravel();q=q.ravel();out=np.empty(len(q));nodes,weights=rule(order)
    for start in range(0,len(q),4096):
        h=chi[start:start+4096];qq=q[start:start+4096];qq=np.maximum(qq,1e-14);u=(1+qq)*abs(h);lo=np.maximum(-a1,u-qq*a2);hi=np.minimum(a1,u+qq*a2)
        edge=abs(h)>=(a1+qq*a2)/(1+qq);hi=np.maximum(hi,lo);anchor=projected_cdf(u/qq,a2)
        value=projected_cdf(lo,a1)+anchor*(projected_cdf(hi,a1)-projected_cdf(lo,a1))
        cuts=np.sort(np.stack([lo,hi,np.clip(np.zeros_like(lo),lo,hi),np.clip(u,lo,hi)],axis=1),axis=1)
        for j in range(3):
            mid=(cuts[:,j]+cuts[:,j+1])/2;width=(cuts[:,j+1]-cuts[:,j])/2;s=mid[:,None]+width[:,None]*nodes
            value+=width*np.sum(projected_pdf(s,a1)*(projected_cdf((u[:,None]-s)/qq[:,None],a2)-anchor[:,None])*weights,axis=1)
        value=np.where(edge,1.,np.clip(value,.5,1.));value=np.where(h<0,1-value,value);value=np.where(h==0,.5,value)
        out[start:start+len(h)]=value
    return out.reshape(shape)


def native_pdf(chi,q,a1,a2,order=128):
    chi,q=np.broadcast_arrays(np.asarray(chi,float),np.asarray(q,float));shape=chi.shape;chi=chi.ravel();q=q.ravel();out=np.zeros(len(q));nodes,weights=rule(order)
    for start in range(0,len(q),4096):
        h=chi[start:start+4096];qq=np.maximum(q[start:start+4096],1e-14);u=(1+qq)*h;lo=np.maximum(-a1,u-qq*a2);hi=np.minimum(a1,u+qq*a2);hi=np.maximum(lo,hi)
        cuts=np.sort(np.stack([lo,hi,np.clip(np.zeros_like(lo),lo,hi),np.clip(u,lo,hi)],axis=1),axis=1);value=np.zeros(len(h))
        for j in range(3):
            mid=(cuts[:,j]+cuts[:,j+1])/2;width=(cuts[:,j+1]-cuts[:,j])/2;s=mid[:,None]+width[:,None]*nodes
            value+=width*np.sum(projected_pdf(s,a1)*projected_pdf((u[:,None]-s)/qq[:,None],a2)*weights,axis=1)
        out[start:start+len(h)]=(1+qq)*value/qq
    return out.reshape(shape)


def spin_coordinate(chi,q,caps,order=64):
    a=max(caps);return 2*a*native_cdf(chi,q,*caps,order=order)-a


def inverse_spin_coordinate(s,q,caps,order=64,iterations=25):
    a=max(caps);s,q=np.broadcast_arrays(np.asarray(s,float),np.asarray(q,float));p=(s+a)/(2*a);target=np.maximum(p,1-p);lo=np.zeros_like(q);hi=(caps[0]+q*caps[1])/(1+q)
    for _ in range(iterations):
        mid=(lo+hi)/2;left=native_cdf(mid,q,*caps,order=order)<target;lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    val=np.where(p<.5,-(lo+hi)/2,(lo+hi)/2);edge=(caps[0]+q*caps[1])/(1+q)
    return np.where(p<=0,-edge,np.where(p>=1,edge,np.where(s==0,0.,val)))


def validate_model(model):
    """Reject ambiguous coordinate/prior conventions before evaluating a model."""
    if model.get('schema')!=SCHEMA or model.get('conditional_mass_chart')!='detector' or model.get('bounded_mass_coordinate')!='delta':
        raise ValueError('Unsupported native-posterior leaf')
    caps=np.asarray(model['spin_caps'],float)
    if caps.shape!=(2,) or not np.isfinite(caps).all() or np.any(caps<=0) or np.any(caps>1):
        raise ValueError('Spin caps must be two finite values in (0,1]')
    if not np.array_equal(caps,np.asarray(model['prior_contract']['spin_caps'])):
        raise ValueError('Native spin prior and CDF caps disagree')
    if model.get('rho_embedded_in_scalar_likelihood',False) is not False:
        raise ValueError('Scalar likelihood must not embed a population redshift law')
    weights=np.asarray(model['component_weights'],float)
    if weights.shape!=(len(model['components']),) or not len(weights) or not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.sum(),1.):
        raise ValueError('Invalid native-posterior mixture weights')
    for c in model['components']:
        if c.get('conditional_mass_chart')!='detector' or c.get('bounded_mass_coordinate')!='delta':
            raise ValueError('Inconsistent native-posterior component chart')


def transformed_x(x,model,s=None):
    a=np.array(x,float,copy=True);y=masses_from_x(a);q=y[:,1]/y[:,0]
    a[:,2]=spin_coordinate(a[:,2],q,model['spin_caps'],model['cdf_order'])if s is None else s
    return a


def log_posterior_density(x,model,s=None):
    validate_model(model)
    a=transformed_x(x,model,s);return logsumexp(np.stack([component_logpdf(a,c)+np.log(p)for c,p in zip(model['components'],model['component_weights'])]),axis=0)


def distance_logfactor(z,contract):
    c=common_cosmology();dl=c.luminosity_distance(z).value
    return log_distance_density(dl,contract['distance_prior'])+np.log(dlprime(z,c))


def log_likelihood_x(x,model):
    x=np.asarray(x,float);scalar=x.ndim==1;a=x.reshape(-1,4);good=support(a,model);out=np.full(len(a),-np.inf)
    if good.any():
        b=a[good];y=masses_from_x(b);q=y[:,1]/y[:,0];cap=(model['spin_caps'][0]+q*model['spin_caps'][1])/(1+q);valid=abs(b[:,2])<=cap
        lp=log_posterior_density(b,model);mc_det=b[:,0]*(1+b[:,3]);logJ=np.log(mc_det)-1.2*np.log(b[:,1])-np.log(2.)
        value=lp+np.log(2*max(model['spin_caps']))-logJ-np.log(b[:,3])-distance_logfactor(b[:,3],model['prior_contract']);out[good]=np.where(valid,value,-np.inf)
    return float(out[0])if scalar else out.reshape(x.shape[:-1])


def integration_bank(model,n=32768,seed=1064,grid_size=384,uniform_fraction=.1):
    """Known mixture of CDF-coordinate and uniform-physical-chi proposals."""
    validate_model(model)
    if not 0<uniform_fraction<1 or n<1 or grid_size<2:raise ValueError('Invalid integration proposal settings')
    rng=np.random.default_rng(seed);caps=model['spin_caps'];scale=max(caps);tables=[];p=np.asarray(model['component_weights']);p=.9999*p/p.sum()+.0001/len(p)
    for c in model['components']:
        de=np.linspace(*c['fit_bounded_mass_bounds'],grid_size+1);se=np.linspace(-scale,scale,grid_size+1);D,S=np.meshgrid((de[:-1]+de[1:])/2,(se[:-1]+se[1:])/2,indexing='ij')
        from aligned_redshift import bounded_logpdf
        lp=bounded_logpdf((1-D*D)/4,S,c).ravel();prob=.9995*np.exp(lp-logsumexp(lp))+.0005/len(lp);tables.append((de,se,prob,prob.reshape(grid_size,grid_size).sum(1)))
    blocks=[];sblocks=[];count=drawn=0
    while count<n:
        k=max(1024,int(np.ceil(1.2*(n-count))));labels=rng.choice(len(p),k,p=p);flat=rng.random(k)<uniform_fraction;x=np.empty((k,4));sv=np.empty(k)
        for j,c in enumerate(model['components']):
            ids=np.flatnonzero(labels==j)
            if not len(ids):continue
            de,se,prob,pdelta=tables[j];isflat=flat[ids];delta=np.empty(len(ids));s=np.empty(len(ids));chi=np.empty(len(ids))
            take=~isflat;cells=rng.choice(len(prob),int(take.sum()),p=prob);di,si=cells//grid_size,cells%grid_size;delta[take]=de[di]+rng.random(len(cells))*(de[1]-de[0]);s[take]=se[si]+rng.random(len(cells))*(se[1]-se[0])
            di=rng.choice(grid_size,int(isflat.sum()),p=pdelta);delta[isflat]=de[di]+rng.random(len(di))*(de[1]-de[0]);eta=(1-delta*delta)/4;q=(1-delta)/(1+delta)
            chi[isflat]=(2*rng.random(isflat.sum())-1)*(caps[0]+q[isflat]*caps[1])/(1+q[isflat]);s[isflat]=spin_coordinate(chi[isflat],q[isflat],caps,model['cdf_order'])
            chi[take]=inverse_spin_coordinate(s[take],q[take],caps,model['cdf_order'],model['inverse_iterations']);mean,cov=conditional(eta,s,c);Y=mean+rng.normal(size=(len(ids),2))@np.linalg.cholesky(cov).T;z=np.exp(Y[:,1]);x[ids]=np.column_stack([Y[:,0]/(1+z),eta,chi,z]);sv[ids]=s
        good=support(x,model)&(abs(sv)<=scale);blocks.append(x[good]);sblocks.append(sv[good]);count+=int(good.sum());drawn+=k
        if drawn>max(100000,100*n):raise RuntimeError('Native-posterior proposal support unresolved')
    x=np.vstack(blocks)[:n];s=np.concatenate(sblocks)[:n];y=masses_from_x(x);q=y[:,1]/y[:,0];delta=(1-q)/(1+q);pdf=native_pdf(x[:,2],q,*caps,order=model['prior_contract'].get('spin_quadrature_order',128));logpdf=np.full_like(pdf,-np.inf);np.log(pdf,out=logpdf,where=pdf>0);cdf_terms=[];flat_terms=[]
    for c,(de,se,prob,pdelta),weight in zip(model['components'],tables,p):
        di=np.clip(np.searchsorted(de,delta,side='right')-1,0,grid_size-1);si=np.clip(np.searchsorted(se,s,side='right')-1,0,grid_size-1);lp=conditional_logpdf(x[:,0],np.log(x[:,3]),x[:,1],s,c)+np.log(weight)
        cdf_terms.append(lp+np.log(prob[di*grid_size+si])-np.log((de[1]-de[0])*(se[1]-se[0]))+np.log(2*scale)+logpdf)
        flat_terms.append(lp+np.log(pdelta[di])-np.log(de[1]-de[0])-np.log(2*(caps[0]+q*caps[1])/(1+q)))
    logq=logsumexp(np.stack([np.log1p(-uniform_fraction)+logsumexp(np.stack(cdf_terms),axis=0),np.log(uniform_fraction)+logsumexp(np.stack(flat_terms),axis=0)]),axis=0)
    lp=log_posterior_density(x,model,s);ld=distance_logfactor(x[:,3],model['prior_contract'])
    # Native posterior density / native prior, times the physical-mass chart Jacobian.
    lw=lp-logq+np.log(2*scale)-2*np.log1p(x[:,3])-ld-np.log(n)
    if not np.isfinite(lw).all():raise RuntimeError('Nonfinite native-posterior likelihood bank')
    lw-=logsumexp(lw)
    return y,lw,x,dict(n=n,seed=seed,rho_embedded=False,uniform_physical_chi_fraction=uniform_fraction,physical_acceptance=count/drawn,cdf_order=model['cdf_order'],inverse_iterations=model['inverse_iterations'],density_chart='Mc_detector,delta,s_native_CDF,logz',integration_measure='dm1_source dm2_source dchi_eff dz',event_normalization='Arbitrary fixed event constant; no absolute evidence')
