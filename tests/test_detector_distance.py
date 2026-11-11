"""Synthetic joint likelihood-shape checks independent of catalogue moments."""
import copy
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import detector_distance as dd

BOUNDS = [[10., 30.], [.1, .25], [-.8, .8], [100., 1000.]]
PRIOR = 'Synthetic known p(dL) proportional to dL^-2, p(chi_eff) proportional to exp(2 chi_eff); other coordinates flat'


def exact_model(coordinate='inverse'):
    m = dd._template(BOUNDS, coordinate, PRIOR, None)
    # All six distinct cross terms are nonzero; positive eta coefficient pushes
    # the posterior against its physical boundary. Parameters are not moments.
    m['coefficients'] = [.25, 1.4, -.35, .2, -.7, .24, -.18, .3, -.4, .17, -.25, -.6, .22, -.5]
    m['log_offset'] = 23.7
    return m


def quadrature(n, model):
    counts = [n]*4 if isinstance(n, int) else list(n)
    nodes_weights = [leggauss(k) for k in counts]
    inds = np.asarray(list(itertools.product(*[range(k) for k in counts])))
    z = np.column_stack([nodes_weights[j][0][inds[:, j]] for j in range(4)])
    qt = np.prod(np.column_stack([nodes_weights[j][1][inds[:, j]] for j in range(4)]), axis=1)
    physical = z*np.asarray(model['scale'])+np.asarray(model['center'])
    # Transform inverse-distance quadrature back to physical dL.
    u = physical[:, 3].copy()
    physical[:, 3] = 1./u
    logjac = np.log(np.prod(model['scale']))-2*np.log(u)
    logprior = -2*np.log(physical[:, 3])+2*physical[:, 2]
    return physical, np.log(qt)+logjac+logprior


class DetectorDistanceTests(unittest.TestCase):
    def test_recovers_all_inverse_distance_cross_terms_from_loglikelihood(self):
        truth = exact_model()
        rng = np.random.default_rng(123)
        x = rng.uniform(np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1], (250, 4))
        y = dd.log_shape(x, truth)
        fit, diagnostic = dd.fit_loglikelihood(x, y, BOUNDS, nuisance_prior=PRIOR)
        # Conditioning changes natural coefficients; compare physical functions.
        np.testing.assert_allclose(dd.log_shape(x, fit), y, atol=1e-10)
        self.assertLess(diagnostic['weighted_loglikelihood_rms'], 1e-11)
        self.assertTrue(diagnostic['numerical_regression_gate'])
        self.assertTrue(diagnostic['numerical_optimizer_gate'])
        self.assertIn('no optimizer',diagnostic['numerical_gate_kind'])
        # Independent physical held-out points, including both eta rails.
        hold = rng.uniform(np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1], (100, 4))
        hold[:2, 1] = [.1, .25]
        np.testing.assert_allclose(dd.log_shape(hold, fit), dd.log_shape(hold, truth), atol=1e-10)

    def test_forward_fit_removes_known_prior_and_handles_joint_boundary(self):
        truth = exact_model()
        posterior, lp = quadrature(8, truth)
        integration, li = quadrature(10, truth)  # independent higher-order nodes
        # An enormous proposal weight outside the original eta domain must
        # contribute exactly zero, not distort the normalization.
        integration = np.vstack([integration, [20, .3, 0, 500]])
        li = np.r_[li, 1e6]
        posterior_logw = lp+dd.log_shape(posterior, truth)
        pw = np.exp(posterior_logw-posterior_logw.max())
        # The integration API computes a mean; deterministic quadrature weights
        # are represented as log(N*w*Jacobian*prior), with the same integral.
        fit, diagnostic = dd.fit_forward(posterior, integration, li+np.log(len(integration)),
                                        BOUNDS, nuisance_prior=PRIOR,
                                        posterior_weights=pw, maxiter=300)
        # Arbitrary likelihood offsets and training-only feature charts differ.
        fitted = dd.log_shape(posterior, fit)
        target_shape = dd.log_shape(posterior, truth)
        np.testing.assert_allclose(fitted-fitted[0], target_shape-target_shape[0], atol=2e-3)
        self.assertLess(diagnostic['maximum_abs_feature_residual'], 2e-7)
        self.assertGreater(diagnostic['integration_ess'], 100)
        self.assertEqual(diagnostic['integration_outside_support_draws'], 1)
        expected_center = (pw/pw.sum())@dd._transform(posterior, 'inverse')
        np.testing.assert_allclose(fit['center'], expected_center, atol=1e-12)
        self.assertEqual(diagnostic['scientific_acceptance'], 'not_assessed')
        # Moments of this boundary/prior-folded distribution are not natural
        # shape parameters. A normal whose moments are copied cannot replace it.
        pw /= pw.sum()
        mu = pw@posterior
        cov = (posterior-mu).T@(pw[:, None]*(posterior-mu))
        probe = np.asarray([[20, .245, .2, 400], [20, .18, .2, 400]])
        normal = -.5*np.einsum('ni,ij,nj->n', probe-mu, np.linalg.inv(cov), probe-mu)
        target = dd.log_shape(probe, truth)
        self.assertGreater(abs(np.diff(normal)[0]-np.diff(target)[0]), .1)

    def test_inverse_distance_training_conditioning_with_broad_declared_box(self):
        truth = exact_model()
        truth['bounds'][3] = [10., 10000.]  # preserve the original feature chart
        rng = np.random.default_rng(95)
        x = rng.uniform(np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1], (400, 4))
        x[:, 3] = rng.uniform(1000, 3000, len(x))
        y = dd.log_shape(x, truth)
        fitted, d = dd.fit_loglikelihood(x, y, truth['bounds'], nuisance_prior=PRIOR)
        np.testing.assert_allclose(fitted['center'], dd._transform(x, 'inverse').mean(0))
        self.assertLess(fitted['scale'][3], .001)
        np.testing.assert_allclose(dd.log_shape(x, fitted), y, atol=1e-10)
        self.assertLess(d['design_condition_number'], 100)

    def test_direct_distance_spline_and_intrinsic_correlations(self):
        rng = np.random.default_rng(391)
        x = rng.uniform(np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1], (900, 4))
        truth = dd._template(BOUNDS, 'log', PRIOR, None, distance_knots=3)
        w = np.ones(len(x))/len(x)
        dd._condition_on_training(truth, x, w)
        dd._setup_spline(truth, x, w)
        truth['coefficients'] = exact_model()['coefficients']+[.03, -.02, .01, .015]*4
        truth['log_offset'] = 8.3
        fitted, d = dd.fit_loglikelihood(x, dd.log_shape(x, truth), BOUNDS,
                                       distance_coordinate='log', nuisance_prior=PRIOR,
                                       distance_knots=3)
        hold = rng.uniform(np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1], (150, 4))
        np.testing.assert_allclose(dd.log_shape(hold, fitted), dd.log_shape(hold, truth), atol=1e-9)
        self.assertEqual(fitted['schema'], 'detector-distance-spline-v1')
        self.assertEqual(d['feature_rank'], 30)
        self.assertNotIn('negative_definite_quadratic', d)
        with self.assertRaises(ValueError):
            bad = copy.deepcopy(fitted)
            bad['coefficients'] = bad['coefficients'][:-1]
            dd.log_shape(hold, bad)

    def test_forward_distance_spline_with_nonflat_prior(self):
        truth = exact_model()
        truth['schema'] = dd.SPLINE_SCHEMA
        truth['spline_knots'] = [-.6, 0., .6]
        # A smooth cubic plus arbitrary full quadratic is in the spline family
        # regardless of the independently selected training quantile knots.
        truth['coefficients'] += [.04, 0., 0., 0.]+[0.]*12
        # Cubic hinges are C2 rather than analytic; resolve their distance
        # sufficient statistics with denser independent distance quadratures.
        posterior, lp = quadrature((8, 8, 8, 32), truth)
        integration, li = quadrature((10, 10, 10, 48), truth)
        lw = lp+dd.log_shape(posterior, truth)
        fitted, d = dd.fit_forward(posterior, integration, li+np.log(len(integration)),
                                  BOUNDS, nuisance_prior=PRIOR,
                                  posterior_weights=np.exp(lw-lw.max()), distance_knots=3)
        actual = dd.log_shape(posterior, fitted)
        expected = dd.log_shape(posterior, truth)
        np.testing.assert_allclose(actual-actual[0], expected-expected[0], atol=8e-3)
        self.assertLess(d['maximum_abs_feature_residual'], 1e-6)
        self.assertEqual(d['feature_count'], 30)
        self.assertEqual(d['feature_rank'], 30)

    def test_scalar_coordinate_substitution_does_not_add_jacobian(self):
        # Constant likelihood stays constant in physical dL for ALL feature
        # charts; a density Jacobian would incorrectly introduce distance shape.
        for coordinate in ['linear', 'log', 'inverse']:
            model = exact_model(coordinate)
            model['coefficients'] = [0.]*14
            x = np.asarray([[20, .2, .1, 150], [20, .2, .1, 950]])
            np.testing.assert_array_equal(dd.log_shape(x, model), [23.7, 23.7])
            self.assertEqual(dd.log_shape(x[0], model), 23.7)
            # Source masses only supply physical detector-mass arguments.
            source_mc, redshift = 12., .5
            point = [source_mc*(1+redshift), .2, .1, 300]
            self.assertEqual(dd.log_shape(point, model), 23.7)

    def test_original_physical_and_component_mass_support(self):
        model = exact_model()
        model['mass_support'] = {'m1': {'min': 15, 'max': 50}, 'm2': [5, 30], 'mtotal_max': 70}
        self.assertTrue(dd.support([20, .25, 0, 300], BOUNDS, model['mass_support']))
        x = np.asarray([[20, .251, 0, 300], [20, .2, 0, 99], [20, .2, 0, 1001],
                        [20, .2, np.nan, 300], [30, .1, 0, 300]])
        self.assertTrue(np.isneginf(dd.log_shape(x, model)).all())
        with self.assertRaises(ValueError):
            dd.support([20, .2, 0, 300], BOUNDS, {'mistyped_mass_bound': 10})

    def test_rejects_invalid_posterior_instead_of_dropping(self):
        post = np.asarray([[20, .2, 0, 300], [20, .26, 0, 300]])
        integration = np.asarray([[20, .2, 0, 300], [20, .22, .1, 400]])
        with self.assertRaises(ValueError):
            dd.fit_forward(post, integration, [0, 0], BOUNDS, nuisance_prior=PRIOR,
                           posterior_weights=[1, 0])
        with self.assertRaises(ValueError):
            dd.fit_loglikelihood(post, [1, 2], BOUNDS, nuisance_prior=PRIOR)
        with self.assertRaises(ValueError):
            dd.fit_forward(integration, integration, [np.inf, 0], BOUNDS, nuisance_prior=PRIOR)
        with self.assertRaises(ValueError):
            dd.fit_forward(integration, integration, [0, 0], BOUNDS, nuisance_prior='')

    def test_bounded_indefinite_shape_preserved_without_eigenvalue_floor(self):
        truth = exact_model('log')
        truth['coefficients'][4] = 2.  # positive curvature in detector chirp mass
        rng = np.random.default_rng(23)
        x = rng.uniform(np.asarray(BOUNDS)[:, 0], np.asarray(BOUNDS)[:, 1], (160, 4))
        fit, d = dd.fit_loglikelihood(x, dd.log_shape(x, truth), BOUNDS,
                                     distance_coordinate='log', nuisance_prior=PRIOR)
        self.assertFalse(d['negative_definite_quadratic'])
        self.assertGreater(max(d['transformed_log_shape_hessian_eigenvalues']), 0)
        np.testing.assert_allclose(dd.log_shape(x, fit), dd.log_shape(x, truth), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
