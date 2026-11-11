import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from conditional_spins import sample_conditioned

class ConditionalSpinTest(unittest.TestCase):
    def test_constraint_support_and_equal_mass_symmetry(self):
        chi=np.full(60000,.45);q=np.ones(len(chi));s=sample_conditioned(chi,q,seed=47)
        np.testing.assert_allclose((s[:,2]+q*s[:,5])/(1+q),chi,rtol=0,atol=2e-15)
        self.assertLessEqual(np.linalg.norm(s[:,:3],axis=1).max(),.99+1e-14)
        self.assertLessEqual(np.linalg.norm(s[:,3:],axis=1).max(),.99+1e-14)
        self.assertAlmostEqual(s[:,2].mean(),.45,delta=.004)
        self.assertAlmostEqual(s[:,0].mean(),0,delta=.004)
    def test_agrees_with_unconditional_isotropic_bin(self):
        rng=np.random.default_rng(49);n=1000000;q=.5
        a=rng.uniform(0,.99,(n,2));z=a*rng.uniform(-1,1,(n,2));chi=(z[:,0]+q*z[:,1])/(1+q)
        target=z[abs(chi-.1)<.006,0];s=sample_conditioned(np.full(40000,.1),np.full(40000,q),seed=48)
        # Independent generative reference, rather than repeating the quadrature.
        for p in [.1,.25,.5,.75,.9]:self.assertAlmostEqual(np.quantile(s[:,2],p),np.quantile(target,p),delta=.018)
    def test_quadrature_order_stability_and_reject_unphysical(self):
        x=np.full(30000,-.8);q=np.full(len(x),.3)
        a=sample_conditioned(x,q,seed=50,order=64);b=sample_conditioned(x,q,seed=51,order=128)
        self.assertAlmostEqual(a[:,2].mean(),b[:,2].mean(),delta=.002)
        with self.assertRaises(ValueError):sample_conditioned([1.],[.5])

if __name__=='__main__':unittest.main()
