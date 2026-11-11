import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prior_folded_likelihood import log_shape,sample,log_normal_interval,sample_forward,intrinsic_logprior
from scipy.stats import truncnorm

class PriorFoldedDispatchTest(unittest.TestCase):
 def artifact(self):
  b={'mc_det_min':15.,'mc_det_max':100.,'q_min':.05,'q_max':1.,'m1_det_min':1.,'m1_det_max':1000.,'m2_det_min':1.,'m2_det_max':1000.,'z_min':.01,'z_max':2.,'amax':.99,'mc_min':5.,'mc_max':100/1.01}
  return {'center':[30,.2,0.],'scale':[.2,.005,.1],'bounds':b,'posterior_component_weights':[.5,.5],
    'log_likelihood_amplitudes':[np.log(.5),np.log(.5)-np.log(10)],
    'components':[{'theta':[0,0,4,-.5,-.5,-.5,0,0,0],'forward_log_normalizer':0},
                  {'theta':[0,0,-4,-.5,-.5,-.5,0,0,0],'forward_log_normalizer':np.log(10)}]}
 def test_likelihood_amplitude_is_not_posterior_fraction(self):
  f=self.artifact();x=np.array([[30,.2,.4],[30,.2,-.4]])
  self.assertAlmostEqual(float(np.diff(log_shape(x,f))[0]),-np.log(10),places=10)
 def test_flat_sampler_uses_gaussian_integrals_and_likelihood_amplitudes(self):
  x,reject=sample(self.artifact(),20000,24)
  self.assertLess(abs(np.mean(x[:,2]>0)-10/11),.01)
  self.assertLess(reject,.001)
 def test_eta_vertex_far_outside_boundary_samples_exact_truncation(self):
  f=self.artifact();f['components']=[{'theta':[0,5,0,-.5,-.125,-.5,0,0,0],'forward_log_normalizer':0}]
  f['posterior_component_weights']=[1.];f['log_likelihood_amplitudes']=[0.]
  x,_=sample(f,10000,28)
  lower=f['bounds']['q_min']/(1+f['bounds']['q_min'])**2
  expected=truncnorm.mean((lower-.3)/.01,(.25-.3)/.01,loc=.3,scale=.01)
  self.assertLess(abs(x[:,1].mean()-expected),.0001)
  self.assertTrue(np.isfinite(log_normal_interval(100,101)))
 def test_forward_sampler_matches_known_prior_reweighting(self):
  f=self.artifact();f['forward_proposal_mean']=[30,.2,0.];f['forward_proposal_cov']=np.diag([.2**2,.005**2,.4**2]).tolist()
  flat,_=sample(f,30000,34);lp=intrinsic_logprior(flat,f);w=np.exp(lp-lp.max());w/=w.sum();expected=w@flat[:,2]
  y,diag=sample_forward(f,15000,35,65536)
  self.assertLess(abs(y[:,2].mean()-expected),.015)
  self.assertLess(diag['integration_ess'],diag['integration_samples'])
  self.assertEqual(diag['output_resample_count'],15000)
 def test_wrong_fraction_dispatch_is_rejected(self):
  f=self.artifact();f['log_likelihood_amplitudes']=np.log([.5,.5]).tolist()
  with self.assertRaises(ValueError):log_shape(np.array([[30,.2,0]]),f)
 def test_eta_wall_has_finite_likelihood(self):
  self.assertTrue(np.isfinite(log_shape(np.array([[30,.25,0.]]),self.artifact())[0]))
 def test_source_support_is_explicit(self):
  self.assertFalse(np.isfinite(log_shape(np.array([[1000,.2,0]]),self.artifact())[0]))
if __name__=='__main__':unittest.main()
