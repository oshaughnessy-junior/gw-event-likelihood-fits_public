"""Independent correlated truncated-Gaussian and positive-curvature recovery."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad,dblquad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bounded2d_fit as b

def truth_gaussian(n=50000,seed=42):
 r=np.random.default_rng(seed);mu=np.array([35.,.27,1.12]);sd=np.array([4.,.055,.43]);cor=np.array([[1,.5,-.3],[.5,1,.25],[-.3,.25,1]]);C=cor*np.outer(sd,sd);out=[]
 while sum(len(x)for x in out)<n:
  x=r.multivariate_normal(mu,C,100000);out.append(x[(x[:,0]>0)&(x[:,1]>0)&(x[:,1]<.25)&(abs(x[:,2])<.99)])
 return np.vstack(out)[:n],mu,C

class BoundedJointTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.x,cls.mu,cls.C=truth_gaussian();cls.f=b.fit(cls.x,np.ones(len(cls.x)));cls.y,cls.reject=b.sample(cls.f,70000,seed=73)
 def test_parameters_are_underlying_shape(self):
  mu=np.array(self.f['gaussian_mu']);C=np.array(self.f['gaussian_cov']);np.testing.assert_allclose((mu-self.mu)/np.sqrt(np.diag(self.C)),0,atol=.08);np.testing.assert_allclose(C/self.C,1,atol=.20)
  self.assertGreater(mu[1],.25);self.assertGreater(mu[2],.99)
 def test_joint_recovery_and_bounds(self):
  self.assertTrue(((self.y[:,1]>0)&(self.y[:,1]<.25)&(abs(self.y[:,2])<.99)).all());self.assertLess(self.reject,.001)
  sd=self.x.std(axis=0);np.testing.assert_allclose((self.y.mean(axis=0)-self.x.mean(axis=0))/sd,0,atol=.035);np.testing.assert_allclose(np.corrcoef(self.y.T),np.corrcoef(self.x.T),atol=.035)
 def test_normalized_full_shape(self):
  l=b.log_shape(self.x[:2000],self.f);P=np.linalg.inv(self.C);d=self.x[:2000]-self.mu;truth=-.5*np.einsum('ni,ij,nj->n',d,P,d);difference=l-truth;self.assertLess(np.std(difference),.06);self.assertLess(self.f['quadrature_crosscheck_max_error'],2e-6)
 def test_absolute_mixture_component_normalization(self):
  a1,a2,b1,c,b2=self.f['natural_parameters'];(lo,hi),(vl,vh)=self.f['bounded_standardized_bounds'];lz=self.f['bounded_log_integral_standardized']
  norm,error=dblquad(lambda v,u:np.exp(a1*u+a2*v+b1*u*u+c*u*v+b2*v*v-lz),lo,hi,lambda _:vl,lambda _:vh,epsabs=1e-9,epsrel=1e-9);self.assertAlmostEqual(norm,1,places=7)
 def test_conditional_integral_against_independent_quad(self):
  for a,bb in [(0.,-.5),(6.,-.1),(-12.,-.3),(2.,3.),(-20.,2.),(.4,0.),(100.,-1e-9),(-100.,-1e-9)]:
   expected=quad(lambda v:np.exp(a*v+bb*v*v),-1.4,2.1,epsabs=1e-10)[0];got=np.exp(b._log_integral_v(np.array([a]),bb,-1.4,2.1))[0];self.assertAlmostEqual(got/expected,1,places=8)
 def test_nearly_linear_remote_gaussian_vertex(self):
  rng=np.random.default_rng(50)
  for a in[-100.,100.]:
   values=b._draw_v(np.full(50000,a),-1e-9,-1.4,2.1,rng);expected=-1.4+.01 if a<0 else 2.1-.01;self.assertLess(abs(values.mean()-expected),.0002)
 def test_positive_bounded_curvature(self):
  rng=np.random.default_rng(98);u=rng.uniform(-.99,.99,180000);u=u[rng.random(len(u))<np.exp(2*(u*u-.99**2))][:35000];eta=rng.normal(.20,.012,len(u));mc=25+2*u+60*(eta-.20)+rng.normal(0,.7,len(u));x=np.column_stack([mc,eta,u]);f=b.fit(x,np.ones(len(x)));self.assertGreater(f['natural_parameters'][4],0);y,_=b.sample(f,50000,seed=3)
  np.testing.assert_allclose(np.quantile(y[:,2],[.1,.5,.9]),np.quantile(u,[.1,.5,.9]),atol=.025)
 def test_narrow_eta(self):
  rng=np.random.default_rng(7);eta=rng.normal(.2499,2e-5,25000);chi=rng.normal(.7,.14,len(eta));valid=abs(chi)<.99;eta=eta[valid];chi=chi[valid];mc=20+1000*(eta-.2499)+.4*chi+rng.normal(0,.05,len(eta));x=np.column_stack([mc,eta,chi]);f=b.fit(x,np.ones(len(x)));y,_=b.sample(f,30000,seed=4);self.assertLess(abs(y[:,1].mean()-eta.mean())/eta.std(),.04)
 def test_native_lower_eta_wall_is_normalized(self):
  rng=np.random.default_rng(711);qmin=.125;lo=qmin/(1+qmin)**2
  # Underlying vertex lies outside the original analysis's q support.
  eta=rng.normal(.08,.025,240000);eta=eta[(eta>lo)&(eta<.25)][:30000]
  chi=.3+3*(eta-lo)+rng.normal(0,.12,len(eta));mc=30+80*(eta-lo)+2*chi+rng.normal(0,.6,len(eta));x=np.column_stack([mc,eta,chi])
  f=b.fit(x,np.ones(len(x)),eta_bounds=(lo,.25));y,reject=b.sample(f,50000,seed=45)
  self.assertTrue((y[:,1]>=lo).all());self.assertLess(reject,.001)
  np.testing.assert_allclose((y.mean(0)-x.mean(0))/x.std(0),0,atol=.04)
  self.assertTrue(np.isneginf(b.log_shape(np.array([[30.,lo-.001,.3]]),f))[0])
  a1,a2,b1,c,b2=f['natural_parameters'];(ul,uh),(vl,vh)=f['bounded_standardized_bounds'];lz=f['bounded_log_integral_standardized']
  norm,_=dblquad(lambda v,u:np.exp(a1*u+a2*v+b1*u*u+c*u*v+b2*v*v-lz),ul,uh,lambda _:vl,lambda _:vh)
  self.assertAlmostEqual(norm,1,places=7)
if __name__=='__main__':unittest.main()
