import sys,unittest
from pathlib import Path
import numpy as np
from scipy.stats import truncnorm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bounded_fit import fit,sample,log_shape,weighted_ks
class NativeEtaTest(unittest.TestCase):
 def test_vertex_below_analysis_lower_wall(self):
  rng=np.random.default_rng(931);lo=.125/1.125**2;mu=.08;sd=.025
  eta=truncnorm.rvs((lo-mu)/sd,(.25-mu)/sd,loc=mu,scale=sd,size=80000,random_state=rng)
  eps=rng.multivariate_normal([0,0],[[1,.02],[.02,.01]],len(eta))
  x=np.column_stack([30+80*eta+eps[:,0],eta,.1+eta+eps[:,1]])
  f=fit(x,np.ones(len(x)),eta_bounds=(lo,.25))
  self.assertLess(f['gaussian_mu'][1],lo)
  self.assertAlmostEqual(f['gaussian_mu'][1],mu,delta=.003)
  self.assertAlmostEqual(np.sqrt(f['gaussian_cov'][1][1]),sd,delta=.001)
  self.assertTrue(np.isneginf(log_shape([40,lo/2,0],f,physical=False)))
  y,_=sample(f,50000,seed=852);self.assertTrue(np.all(y[:,1]>=lo))
  self.assertLess(weighted_ks(x[:,1],np.ones(len(x))/len(x),y[:,1]),.012)
if __name__=='__main__':unittest.main()
