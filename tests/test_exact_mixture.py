"""A small likelihood mode must survive independent validation-prior reweighting."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mixture_fit import fit_mixture
from bounded_fit import sample
class ExactWeightedMixture(unittest.TestCase):
 def test_small_mode_is_retained_without_resampling(self):
  rng=np.random.default_rng(124)
  a=rng.normal([50,.18,.5],[.5,.005,.02],(1600,3))
  b=rng.normal([80,.18,0],[.5,.005,.02],(400,3))
  x=np.concatenate([a,b]);w=np.r_[np.full(len(a),.995/len(a)),np.full(len(b),.005/len(b))]
  f=fit_mixture(x,w,k=2,maxiter=50,max_reference_rows=0,min_component_weight=1e-4,min_component_ess=30)
  self.assertAlmostEqual(min(f['component_weights']),.005,places=5)
  self.assertAlmostEqual(f['effective_sample_size'],1/(w@w))
  self.assertGreater(min(f['component_weight_ess']),300)
  y,_=sample(f,100000,seed=74)
  py=np.exp(-.5*(y[:,2]/.1)**2);px=np.exp(-.5*(x[:,2]/.1)**2)*w
  # The 0.5% likelihood mode dominates under this spin prior; compare its mass.
  self.assertLess(abs(py[y[:,0]>65].sum()/py.sum()-px[x[:,0]>65].sum()/px.sum()),.015)
if __name__=='__main__':unittest.main()
