"""Normalized likelihood-shape MLE in bounded (eta,chi), conditional Gaussian Mc.

The five canonical parameters of the rectangle-truncated quadratic are fitted,
not equated with physical means/covariances. The Mc conditional mean contains
both bounded coordinates, preserving all joint correlations. Mc>0 is enforced
at evaluation/sampling, but not in the fit normalization; its rejection fraction
must therefore pass the release's bound-effect gate.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, dawsn, erfcx
from scipy.stats import truncnorm

SCHEMA='bounded-eta-chi-quadratic-v1'

def _segments(lo,hi):
    offset=np.array([0.,.001,.01,.1,1.,2.,6.,12.])
    return np.unique(np.clip(np.r_[lo+offset,hi-offset,[-12.,-6.,-2.,0.,2.,6.,12.]],lo,hi))

def _axis(lo,hi,order):
    edges=_segments(lo,hi);n=max(8,int(np.ceil(order/8)));nodes,weights=leggauss(n)
    mids=(edges[:-1]+edges[1:])/2;widths=np.diff(edges)/2
    return (mids[:,None]+widths[:,None]*nodes).ravel(),(widths[:,None]*weights).ravel()

def _rule(bounds,order):
    u,wu=_axis(*bounds[0],order);v,wv=_axis(*bounds[1],order)
    U,V=np.meshgrid(u,v,indexing='ij');U=U.ravel();V=V.ravel()
    phi=np.column_stack([U,V,U*U,U*V,V*V]);logweight=np.log(np.outer(wu,wv).ravel())
    return phi,logweight

def _integral(theta,rule):
    phi,lw=rule;ll=phi@theta+lw;lz=logsumexp(ll);p=np.exp(ll-lz);m=p@phi
    d=phi-m;H=(d*p[:,None]).T@d
    return float(lz),m,H

def fit(x,w,chi_bound=.99,order=128,check_order=192,eta_bounds=(0.,.25)):
    x=np.asarray(x,float);w=np.asarray(w,float)
    if not np.isfinite(w).all()or np.any(w<0)or w.sum()<=0:raise ValueError('Invalid nonnegative source weights')
    w=w/w.sum()
    if x.ndim!=2 or x.shape[1]!=3 or len(w)!=len(x) or not np.isfinite(x).all() or np.any(w<0):raise ValueError('Invalid weighted3D reference')
    eta_bounds=np.asarray(eta_bounds,float)
    if eta_bounds.shape!=(2,)or not 0<=eta_bounds[0]<eta_bounds[1]<=.25:raise ValueError('Invalid declared eta interval')
    z=x[:,[1,2]];physical=np.array([eta_bounds,[-chi_bound,chi_bound]])
    if np.any((z<=physical[:,0])|(z>=physical[:,1])):raise ValueError('Reference outside declared open eta/chi support')
    center=w@z;scale=np.sqrt(w@((z-center)**2))
    if np.any(scale<=0):raise ValueError('Degenerate bounded coordinate')
    uv=(z-center)/scale;u,v=uv.T;target=w@np.column_stack([u,v,u*u,u*v,v*v]);bounds=(physical-center[:,None])/scale[:,None]
    initial=np.array([0.,0.,-.5,0.,-.5]);attempts=[];solution=None
    for qorder in [order,check_order,256]:
        if solution is not None and qorder<=solution['order']:continue
        rule=_rule(bounds,qorder)
        def fun(theta):
            lz,m,_=_integral(theta,rule);return lz-theta@target,m-target
        res=minimize(fun,initial,jac=True,hess=lambda p:_integral(p,rule)[2],method='trust-exact',options={'gtol':2e-10,'maxiter':300})
        lz,m,_=_integral(res.x,rule);higher=_rule(bounds,max(qorder+64,check_order));hz,hm,_=_integral(res.x,higher)
        residual=float(np.max(abs(m-target)));cross=float(max(abs(hz-lz),np.max(abs(hm-m))))
        attempts.append({'order':qorder,'moment_equation_residual':residual,'higher_order':max(qorder+64,check_order),'higher_order_error':cross,'optimizer_success':bool(res.success)})
        solution={'order':qorder,'theta':res.x,'logZ':hz,'gradient':float(np.max(abs(hm-target))),'cross':cross};initial=res.x
        if residual<1e-7 and cross<2e-7:break
    if solution['gradient']>1e-6 or solution['cross']>2e-6:raise RuntimeError('Bounded2D quadrature/normal equations unresolved '+str(attempts))
    design=np.column_stack([np.ones(len(x)),uv]);B=np.linalg.solve((design*w[:,None]).T@design,(design*w[:,None]).T@x[:,0]);r=x[:,0]-design@B;variance=float(w@(r*r))
    if variance<=0:raise ValueError('Degenerate conditional chirp mass')
    f={'schema':SCHEMA,'coords':['mc_source','eta','chi_eff'],'bounded_center':center.tolist(),'bounded_scale':scale.tolist(),'bounded_standardized_bounds':bounds.tolist(),'natural_parameters':solution['theta'].tolist(),'bounded_log_integral_standardized':solution['logZ'],'mc_conditional_intercept':float(B[0]),'mc_conditional_slopes':B[1:].tolist(),'mc_conditional_variance':variance,'effective_sample_size':float(1/(w@w)),'gradient_max':solution['gradient'],'conditioned_gradient_max':solution['gradient'],'quadrature_crosscheck_max_error':solution['cross'],'quadrature_attempts':attempts,'fit_method':'normalized joint eta/chi bounded-quadratic maximum likelihood plus conditional Gaussian Mc; full joint cross terms','fit_truncation':['0<eta<.25',f'abs(chi_eff)<{chi_bound}'],'evaluation_bounds':{'mc_source':[0,None],'eta':[0,.25],'chi_eff':[-chi_bound,chi_bound]},'other_bound_normalization':'Mc>0 omitted from fit normalization; gate its sampling rejection fraction','gaussian_parameters_identifiable':False}
    f['evaluation_bounds']['eta']=eta_bounds.tolist()
    f['fit_eta_bounds']=eta_bounds.tolist()
    f['fit_truncation'][0]=f'{eta_bounds[0]}<eta<{eta_bounds[1]}'
    a1,a2,b1,c,b2=solution['theta'];P=-np.array([[2*b1,c],[c,2*b2]])
    if np.linalg.eigvalsh(P).min()>1e-9:
        Cuv=np.linalg.inv(P);muv=Cuv@np.array([a1,a2]);mu=np.array([B[0]+B[1:]@muv,*(center+scale*muv)]);T=np.zeros((3,2));T[0]=B[1:];T[1,0]=scale[0];T[2,1]=scale[1];C=T@Cuv@T.T;C[0,0]+=variance
        f.update(gaussian_mu=mu.tolist(),gaussian_cov=C.tolist(),gaussian_parameters_identifiable=True)
    return f

def log_shape(x,f,physical=True):
    x=np.asarray(x,float);uv=(x[:,[1,2]]-f['bounded_center'])/f['bounded_scale'];u,v=uv.T;a1,a2,b1,c,b2=f['natural_parameters'];mean=f['mc_conditional_intercept']+uv@np.array(f['mc_conditional_slopes']);var=f['mc_conditional_variance']
    ll=a1*u+a2*v+b1*u*u+c*u*v+b2*v*v-f['bounded_log_integral_standardized']-np.log(np.prod(f['bounded_scale']))-.5*((x[:,0]-mean)**2/var+np.log(2*np.pi*var));bounds=np.array(f['bounded_standardized_bounds']);good=((x[:,0]>0)if physical else np.ones(len(x),dtype=bool))&np.all((uv>=bounds[:,0])&(uv<=bounds[:,1]),axis=1)
    return np.where(good,ll,-np.inf)

def _log_linear_integral(a,lo,hi):
    a=np.asarray(a);width=hi-lo;out=np.empty_like(a,dtype=float);small=abs(a*width)<1e-7
    out[small]=np.log(width)+a[small]*(lo+hi)/2
    t=abs(a[~small]);out[~small]=np.maximum(a[~small]*lo,a[~small]*hi)+np.log(-np.expm1(-t*width))-np.log(t)
    return out

def _log_integral_v(a,b,lo,hi):
    a=np.asarray(a,float)
    if abs(b)<1e-10:return _log_linear_integral(a,lo,hi)
    if b<0:
        sd=np.sqrt(-1/(2*b));mean=-a/(2*b);L=(lo-mean)/sd;H=(hi-mean)/sd;out=np.empty_like(a)
        # Scaled erfc avoids cancellation of two enormous completing-square
        # constants when an almost-linear component has a remote Gaussian vertex.
        right=L>0;left=H<0;cross=~(right|left)
        for mask,upper,lower,zupper,zlower in[(right,lo,hi,L,-H),(left,hi,lo,-H,L)]:
            if not mask.any():continue
            U=a[mask]*upper+b*upper*upper+np.log(erfcx(zupper[mask]/np.sqrt(2)))
            V=a[mask]*lower+b*lower*lower+np.log(erfcx(-zlower[mask]/np.sqrt(2)))
            out[mask]=np.log(sd)+.5*np.log(np.pi/2)+U+np.log(-np.expm1(np.minimum(V-U,-1e-16)))
        if cross.any():
            lm=np.log1p(-np.exp(log_ndtr(L[cross]))-np.exp(log_ndtr(-H[cross])))
            out[cross]=-.25*a[cross]*a[cross]/b+.5*np.log(2*np.pi)+np.log(sd)+lm
        return out
    root=np.sqrt(b);dl=dawsn(root*lo+a/(2*root));dh=dawsn(root*hi+a/(2*root));logs=np.stack([a*hi+b*hi*hi+np.log(abs(dh)),a*lo+b*lo*lo+np.log(abs(dl))]);signs=np.stack([np.sign(dh),-np.sign(dl)]);lz,sg=logsumexp(logs,b=signs,axis=0,return_sign=True)
    if np.any(sg<=0):raise ValueError('Positive-curvature conditional integral unresolved')
    return lz-.5*np.log(b)

def _draw_linear(a,lo,hi,rng):
    a=np.asarray(a,float);p=rng.random(len(a));width=hi-lo;out=np.empty_like(a);small=abs(a*width)<1e-7;out[small]=lo+p[small]*width
    pos=(a>0)&~small;neg=~pos&~small
    out[pos]=hi+np.log(p[pos]+(1-p[pos])*np.exp(-a[pos]*width))/a[pos]
    out[neg]=lo+np.log1p(-p[neg]*(-np.expm1(a[neg]*width)))/a[neg]
    return out

def _draw_v(a,b,lo,hi,rng):
    a=np.asarray(a,float)
    if abs(b)<1e-10:return _draw_linear(a,lo,hi,rng)
    if b<0 and abs(b)*(hi-lo)**2<1e-4:
        # Exact tangent-exponential rejection avoids subtracting a remote
        # Gaussian vertex from a huge normal deviate in the nearly-linear limit.
        mode=np.clip(-a/(2*b),lo,hi);rate=a+2*b*mode;out=np.empty(len(a));pending=np.arange(len(a))
        while len(pending):
            val=_draw_linear(rate[pending],lo,hi,rng);good=np.log(rng.random(len(pending)))<b*(val-mode[pending])**2;out[pending[good]]=val[good];pending=pending[~good]
        return out
    if b<0:
        sd=np.sqrt(-1/(2*b));mean=-a/(2*b)
        return truncnorm.rvs((lo-mean)/sd,(hi-mean)/sd,loc=mean,scale=sd,random_state=rng)
    # Exact rejection from a mixture of the two secant exponential envelopes.
    mid=(lo+hi)/2;L=np.array([lo,mid]);H=np.array([mid,hi]);out=np.empty(len(a));pending=np.arange(len(a));tries=0
    while len(pending):
        aa=a[pending];rates=aa[:,None]+b*(L+H);logZ=np.column_stack([_log_linear_integral(rates[:,j],L[j],H[j])-b*L[j]*H[j]for j in range(2)]);p0=np.exp(logZ[:,0]-logsumexp(logZ,axis=1));choice=(rng.random(len(aa))>=p0).astype(int);val=np.empty(len(aa))
        for j in range(2):
            take=choice==j;val[take]=_draw_linear(rates[take,j],L[j],H[j],rng)
        accept=np.log(rng.random(len(aa)))<b*(val-L[choice])*(val-H[choice]);out[pending[accept]]=val[accept];pending=pending[~accept];tries+=1
        if tries>1000:raise RuntimeError('Conditional sampler exhausted')
    return out

def sample(f,n,seed=810,grid_per_segment=1601,mc_positive=True):
    rng=np.random.default_rng(seed);bounds=np.asarray(f['bounded_standardized_bounds']);a1,a2,b1,c,b2=f['natural_parameters'];edges=_segments(*bounds[0]);grid=np.unique(np.concatenate([np.linspace(a,b,grid_per_segment)for a,b in zip(edges[:-1],edges[1:])]))
    ll=a1*grid+b1*grid*grid+_log_integral_v(a2+c*grid,b2,*bounds[1]);p=np.exp(ll-ll.max());cdf=np.r_[0,cumulative_trapezoid(p,grid)];cdf/=cdf[-1]
    accepted=[];drawn=0;rejected=0;remaining=n
    while remaining:
        count=max(remaining,100);u=np.interp(rng.random(count),cdf,grid);v=_draw_v(a2+c*u,b2,*bounds[1],rng);uv=np.column_stack([u,v]);mc=f['mc_conditional_intercept']+uv@np.array(f['mc_conditional_slopes'])+rng.normal(size=count)*np.sqrt(f['mc_conditional_variance']);z=np.array(f['bounded_center'])+uv*np.array(f['bounded_scale']);x=np.column_stack([mc,z]);physical=mc>0;good=physical if mc_positive else np.ones(count,dtype=bool);accepted.append(x[good]);drawn+=count;rejected+=int((~physical).sum());remaining=max(0,n-sum(len(a)for a in accepted))
        if drawn>max(10000,1000*n):raise RuntimeError('Mc positive-bound acceptance too small')
    return np.vstack(accepted)[:n],rejected/drawn
