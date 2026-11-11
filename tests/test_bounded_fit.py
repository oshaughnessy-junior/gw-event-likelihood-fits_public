import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from bounded_fit import fit,log_shape,sample,validate

class FitRecovery(unittest.TestCase):
    def check_recovery(self,a,b):
        truth={'eta_a':a,'eta_b':b,'conditional_intercept':[20.,-.2],
               'conditional_slope':[5.,.4],'conditional_cov':[[1.,.03],[.03,.01]]}
        x,_=sample(truth,n=100000,seed=27);f=fit(x,np.ones(len(x)))
        y,_=sample(f,n=70000,seed=91)
        self.assertLess(max(validate(x,np.ones(len(x)),f)['marginal_ks']),.015)
        self.assertLess(max(validate(x,np.ones(len(x)),f)['projected_ks']),.02)
        # Validate the shape, not potentially non-identifiable outside-domain loc.
        eta=np.linspace(*np.quantile(x[:,1],[.01,.99]),31)
        grid=np.column_stack([20+5*eta/.25,eta,-.2+.4*eta/.25])
        delta=log_shape(grid,f)-log_shape(grid,truth)
        self.assertLess(np.std(delta),.25)
    def test_regular_gaussian(self):self.check_recovery(25.,-20.)
    def test_center_beyond_equal_mass_wall(self):self.check_recovery(70.,-25.)
    def test_bounded_positive_curvature(self):self.check_recovery(-3.,8.)
    def test_narrow_interior(self):self.check_recovery(8000.,-8000.)
    def test_non_gaussian_mixture_flagged(self):
        rng=np.random.default_rng(9);x=np.column_stack([rng.normal(25,1,30000),
             np.r_[rng.normal(.12,.003,15000),rng.normal(.235,.003,15000)],rng.normal(0,.1,30000)])
        f=fit(x,np.ones(len(x)));r=validate(x,np.ones(len(x)),f)
        self.assertEqual(r['status'],'requires_fallback')
    def test_mixture_truncation_changes_component_weights(self):
        # Equal pre-truncation masses: component 2 loses half its mass at Mc=0.
        a={'eta_a':0.,'eta_b':0.,'conditional_intercept':[10.,0.],
           'conditional_slope':[0.,0.],'conditional_cov':[[1.,0],[0,.01]]}
        b=a|{'conditional_intercept':[0.,0.]}
        x,reject=sample({'components':[a,b],'component_weights':[.5,.5]},60000,seed=80)
        self.assertAlmostEqual(float((x[:,0]>5).mean()),2/3,delta=.015)
        self.assertAlmostEqual(reject,.25,delta=.01)

    def test_bounds_and_coordinate_order(self):
        t={'eta_a':20.,'eta_b':-10.,'conditional_intercept':[20.,0.],
           'conditional_slope':[2.,.1],'conditional_cov':[[1.,0],[0,.01]]}
        self.assertTrue(np.isneginf(log_shape([20,.251,0],t)))
        self.assertTrue(np.isfinite(log_shape([20,.25,0],t)))

if __name__=='__main__':unittest.main()
