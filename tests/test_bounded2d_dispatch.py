import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bounded_fit import log_shape,sample

class BoundedDispatchTest(unittest.TestCase):
 def test_scalar_evaluation_and_global_component_truncation(self):
  a={'schema':'bounded-eta-chi-quadratic-v1','coords':['mc_source','eta','chi_eff'],
     'bounded_center':[.125,0.],'bounded_scale':[.1,.5],
     'bounded_standardized_bounds':[[-1.25,1.25],[-1.98,1.98]],
     'natural_parameters':[0.]*5,'bounded_log_integral_standardized':float(np.log(2.5*3.96)),
     'mc_conditional_intercept':10.,'mc_conditional_slopes':[0.,0.],'mc_conditional_variance':1.}
  b=a|{'mc_conditional_intercept':0.}
  f={'components':[a,b],'component_weights':[.5,.5]}
  self.assertTrue(np.isfinite(log_shape([10,.2,0],f)))
  self.assertTrue(np.isneginf(log_shape([10,.2,.995],f)))
  support={'mc_min':1.,'mc_max':20.,'q_min':.01,'q_max':1.,'amax':.99,
           'z_min':0.,'z_max':1.,'mc_det_min':1.,'mc_det_max':5.,
           'm1_det_min':0.,'m1_det_max':1000.,'m2_det_min':0.,'m2_det_max':1000.}
  limited=a|{'analysis_support':support}
  self.assertTrue(np.isneginf(log_shape([10,.2,0],limited)))
  self.assertTrue(np.isfinite(log_shape([10,.2,0],limited,physical=False)))
  self.assertTrue(np.isfinite(log_shape([3,.2,0],limited)))
  x,reject=sample(f,40000,seed=98)
  self.assertAlmostEqual(float((x[:,0]>5).mean()),2/3,delta=.015)
  self.assertAlmostEqual(reject,.25,delta=.01)

if __name__=='__main__':unittest.main()
