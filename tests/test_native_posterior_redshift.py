"""Known correlated posterior, prior quotient, proposal and endpoint controls."""
from pathlib import Path
import sys,unittest,time
import numpy as np
from scipy.stats import truncnorm,multivariate_normal,kstest
from scipy.special import ndtr,logsumexp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import native_posterior_redshift as p
from native_redshift_prior import log_native_prior,common_cosmology,dlprime
from aligned_redshift import log_mass_jacobian,masses_from_x

class NativePosteriorRedshiftTests(unittest.TestCase):
    def test_correlated_native_restoration_and_negative_controls(self):
        r=np.random.default_rng(192);caps=[.99,.99];a=.99;dlo,dhi=0.,(1-.05)/(1+.05);dm,ds=.35,.14;sm,ss=0.,.45
        c=dict(bounded_center=[dm/4,sm],bounded_scale=[ds/4,ss],natural_parameters=[0.,0.,-.5,0.,-.5],bounded_log_integral_standardized=float(np.log(2*np.pi)+np.log(ndtr((dhi-dm)/ds)-ndtr((dlo-dm)/ds))+np.log(ndtr((a-sm)/ss)-ndtr((-a-sm)/ss))),fit_bounded_mass_bounds=[dlo,dhi],bounded_mass_coordinate='delta',conditional_mass_chart='detector',joint_conditional_intercept=[30.,-1.5],joint_conditional_slopes=[[.5,.05],[.1,.04]],joint_conditional_cov=[[.25,.005],[.005,.0225]])
        b=dict(amax=a,a1max=a,a2max=a,q_min=.05,q_max=1.,m1_det_min=1.,m1_det_max=250.,m2_det_min=1.,m2_det_max=250.,z_min=.0001,z_max=3.)
        contract=dict(analysis_support=b,spin_caps=caps,spin_quadrature_order=128,distance_prior=dict(kind='distance_squared'))
        m=dict(schema=p.SCHEMA,conditional_mass_chart='detector',bounded_mass_coordinate='delta',components=[c],component_weights=[1.],analysis_support=b,prior_contract=contract,spin_caps=caps,cdf_order=64,inverse_iterations=25)
        t=time.monotonic();y,lw,x,meta=p.integration_bank(m,n=6000,seed=73,grid_size=128);delta=(y[:,0]-y[:,1])/(y[:,0]+y[:,1]);q=y[:,1]/y[:,0];s=p.spin_coordinate(x[:,2],q,caps);z=x[:,3];mc_det=x[:,0]*(1+z);u=(delta-dm)/ds;v=s/ss;mean=np.column_stack([30+.5*u+.1*v,-1.5+.05*u+.04*v]);Y=np.column_stack([mc_det,np.log(z)])
        known_P=truncnorm.logpdf(delta,(dlo-dm)/ds,(dhi-dm)/ds,loc=dm,scale=ds)+truncnorm.logpdf(s,-a/ss,a/ss,loc=0,scale=ss)+multivariate_normal.logpdf(Y-mean,mean=[0,0],cov=c['joint_conditional_cov'])
        cos=common_cosmology();dl=cos.luminosity_distance(z).value;distance=2*np.log(dl)+np.log(dlprime(z,cos));expected=known_P+np.log(2*a)-(np.log(mc_det)-1.2*np.log(x[:,1])-np.log(2))-np.log(z)-distance
        actual=p.log_likelihood_x(x,m);point_error=float(np.max(abs(expected-actual)));assert point_error<2e-10,point_error
        native=log_native_prior(x,contract,p.native_pdf,order=128);nw=lw+native-log_mass_jacobian(x);nw=np.exp(nw-logsumexp(nw));expected_d=truncnorm.mean((dlo-dm)/ds,(dhi-dm)/ds,loc=dm,scale=ds);moment_error=[float(nw@delta-expected_d),float(nw@s),float(nw@mc_det-(30+.5*(expected_d-dm)/ds))];assert max(abs(np.array(moment_error)))<.03,moment_error
        # No pchi divisor remains in scalar L, including the finite spin endpoints.
        assert np.isfinite(p.log_likelihood_x(np.array([[30.,.2,a,.3],[30.,.2,-a,.3],[30.,.25,0.,.3]]),m)).all()
        qq=r.uniform(.03,1,12000);chi=(r.uniform(0,a,len(qq))*r.uniform(-1,1,len(qq))+qq*r.uniform(0,a,len(qq))*r.uniform(-1,1,len(qq)))/(1+qq);U=p.native_cdf(chi,qq,*caps);correct_ks=float(kstest(U,'uniform').statistic);wrong_ks=float(kstest((chi+a)/(2*a),'uniform').statistic);assert correct_ks<.02 and wrong_ks>.15,(correct_ks,wrong_ks)
        # Absolute event normalization is arbitrary; omitted shape factors are not.
        wrong_missing_distance=actual+distance;diff=(wrong_missing_distance-expected);diff-=np.median(diff);omission_error=float(np.max(abs(diff)));assert omission_error>.1,omission_error
        qs=np.repeat([.03,.1,.3,.7,1.],7);cs=np.tile([-.9,-.3,-.03,0.,.03,.3,.9],5);h=1e-5;der=(p.native_cdf(cs+h,qs,*caps)-p.native_cdf(cs-h,qs,*caps))/(2*h);pdf=p.native_pdf(cs,qs,*caps,order=256);jac_error=float(np.max(abs(der/pdf-1)));assert jac_error<5e-4,jac_error

    def test_unequal_caps_inverse_and_native_distance_jacobian(self):
        q=np.repeat([.03,.2,1.],7)
        caps=[.7,.99]
        chi=np.tile(np.linspace(-.6,.6,7),3)
        feature=p.spin_coordinate(chi,q,caps)
        recovered=p.inverse_spin_coordinate(feature,q,caps,iterations=30)
        self.assertLess(np.max(abs(recovered-chi)),2e-8)
        z=np.array([.03,.2,.8])
        cos=common_cosmology();dl=cos.luminosity_distance(z).value
        contract={'distance_prior':{'kind':'common_source_time'}}
        actual=p.distance_logfactor(z,contract)
        expected=np.log(cos.differential_comoving_volume(z).value)-np.log1p(z)
        self.assertLess(np.max(abs(actual-expected)),2e-7)
        with self.assertRaises(ValueError):
            p.integration_bank({},n=10)

if __name__=="__main__":unittest.main()
