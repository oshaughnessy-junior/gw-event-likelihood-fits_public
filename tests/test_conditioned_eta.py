"""Regression for narrow eta near the equal-mass wall (GW250114_082203 failure)."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import truncnorm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bounded_fit import fit as fit_conditioned, validate


class ConditionedEtaTests(unittest.TestCase):
    def test_narrow_boundary_truncated_correlated_normal(self):
        rng=np.random.default_rng(490)
        # Underlying Gaussian eta vertex OUTSIDE the physical domain.
        mt,st=1.00015,.0004
        t=truncnorm.rvs(-mt/st,(1-mt)/st,loc=mt,scale=st,size=90000,random_state=rng)
        mc=28.7+30*(t-1)+rng.normal(0,.3,len(t))
        chi=-.035-2*(t-1)+.015*(mc-28.7)+rng.normal(0,.02,len(t))
        x=np.column_stack([mc,.25*t,chi]);w=np.ones(len(t))
        fit=fit_conditioned(x,w)
        self.assertLess(fit['gradient_max'],1e-8)
        self.assertLess(fit['conditioned_gradient_max'],1e-6)
        self.assertGreater(fit['gaussian_mu'][1],.25)
        self.assertEqual(validate(x,w,fit)['status'],'pass_screen')


if __name__=='__main__':unittest.main()
