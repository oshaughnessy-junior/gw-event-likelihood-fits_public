"""Independent full-covariance and physical-measure controls for the 5D API."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eccentric_likelihood import (forward_coordinates, inverse_coordinates,
                                  log_likelihood, log_prior, sample, support)


class EccentricMeasureTest(unittest.TestCase):
    def setUp(self):
        # Off-diagonal mass/e and mass/z correlations are intentional controls.
        a = np.array([[.18, 0, 0, 0, 0], [.06, .4, 0, 0, 0],
                      [0, -.04, .25, 0, 0], [.10, 0, 0, .16, 0],
                      [.05, .08, 0, -.06, .35]])
        cov = a @ a.T
        self.fit = dict(
            schema="eccentric-joint-density-ratio-v1",
            support=dict(q_min=.05, q_max=1., amax=.99, ecc_min=0., ecc_max=.4,
                         z_min=.01, z_max=3., mc_det_min=1., mc_det_max=1000.),
            mass_shear=dict(a=157/40, b=.4),
            priors=dict(eccentricity_power=1, spin_regularizer=1e-7,
                        redshift_density=dict(z=[.01, 3.], log_density=[0., 0.])),
            density=dict(weights=[1.], means=[[np.log(20), .1, .05, .4, -.4]],
                         covariances=[cov.tolist()]),
            posterior_support_probability=1.,
        )
        self.point = np.array([18., .21, .15, .16, .7])

    def numeric_jacobian(self, point):
        columns = []
        for j in range(5):
            h = 1e-6 * max(abs(point[j]), .01)
            lo, hi = point.copy(), point.copy()
            lo[j] -= h
            hi[j] += h
            columns.append((forward_coordinates(hi, self.fit)[0] -
                            forward_coordinates(lo, self.fit)[0]) / (2*h))
        return np.stack(columns, axis=1)

    def test_density_ratio_matches_independent_pdf_and_finite_difference_jacobian(self):
        for power in [1, 2]:
            self.fit["priors"]["eccentricity_power"] = power
            t = forward_coordinates(self.point, self.fit)[0]
            reflected = t.copy()
            reflected[3] *= -1
            g = self.fit["density"]
            pdf = multivariate_normal(g["means"][0], g["covariances"][0])
            folded = logsumexp([pdf.logpdf(t), pdf.logpdf(reflected)])
            jac = np.linalg.slogdet(self.numeric_jacobian(self.point))[1]
            actual = log_likelihood(self.point, self.fit)[0] + log_prior(self.point, self.fit)[0]
            self.assertAlmostEqual(actual, folded + jac, places=7)

    def test_full_covariance_is_retained(self):
        diagonal = copy.deepcopy(self.fit)
        cov = np.asarray(diagonal["density"]["covariances"][0])
        diagonal["density"]["covariances"] = [np.diag(np.diag(cov)).tolist()]
        difference = abs(log_likelihood(self.point, self.fit)[0] -
                         log_likelihood(self.point, diagonal)[0])
        self.assertGreater(difference, .1)

    def test_eccentric_zero_limit_is_finite_for_both_native_priors(self):
        for power in [1, 2]:
            self.fit["priors"]["eccentricity_power"] = power
            x = np.tile(self.point, (3, 1))
            x[:, 3] = [0., 1e-10, 1e-8]
            values = log_likelihood(x, self.fit)
            self.assertTrue(np.isfinite(values).all())
            np.testing.assert_allclose(values, values[0], atol=1e-8)

    def test_coordinate_inverse_and_joint_draws(self):
        x = sample(self.fit, 300, seed=17)
        self.assertTrue(support(x, self.fit).all())
        np.testing.assert_allclose(inverse_coordinates(forward_coordinates(x, self.fit), self.fit),
                                   x, rtol=1e-10, atol=1e-10)
        self.assertEqual(x.shape, (300, 5))
        # A deterministic broad check catches accidental independent z resampling.
        self.assertGreater(abs(np.corrcoef(np.log(x[:, 0]), x[:, 4])[0, 1]), .1)

    def test_coupled_detector_mass_support(self):
        self.fit["support"]["mc_det_max"] = 35.
        x = np.tile(self.point, (2, 1))
        x[:, 4] = [.5, 2.]
        np.testing.assert_array_equal(support(x, self.fit), [True, False])
        self.assertTrue(np.isneginf(log_likelihood(x, self.fit)[1]))


if __name__ == "__main__":
    unittest.main()
