"""Synthetic controls for the native-spin base-measure 4D leaf; no observational coefficients."""
from pathlib import Path
import copy, hashlib, json, sys, unittest
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import native_spin_base_redshift as b
from native_posterior_redshift import native_pdf
from native_redshift_prior import log_native_prior, common_cosmology, dlprime
from aligned_redshift import log_mass_jacobian, masses_from_x

CENTER, SCALE = [.35, .05], [.14, .2]
THETA = [.3, -.2, -.5, .1, -.5]
COEF = [[30., -1.], [.5, .05], [.1, .04]]
COV = [[.25, .02], [.02, .25]]


def reference_grid(caps, support, n_delta=128, n_chi=1600):
    """Independent quadrature over (delta, chi): Gauss-Legendre in delta, midpoint in chi, pi normalized per delta node."""
    lo, hi = b.delta_bounds(support); nd, wd = np.polynomial.legendre.leggauss(n_delta)
    d = (lo + hi) / 2 + (hi - lo) / 2 * nd; wd = wd * (hi - lo) / 2; q = (1 - d) / (1 + d); e = (caps[0] + q * caps[1]) / (1 + q)
    t = (np.arange(n_chi) + .5) / n_chi * 2 - 1; chi = e[:, None] * t; dchi = 2 * e / n_chi
    pi = native_pdf(chi, np.broadcast_to(q[:, None], chi.shape), *caps, order=128); pi /= (pi.sum(1) * dchi)[:, None]
    return np.broadcast_to(d[:, None], chi.shape).ravel(), chi.ravel(), np.log(pi * (wd * dchi)[:, None]).ravel()


def synthetic_model(caps=(.99, .99), thetas=(THETA,), weights=(1.,), per_spin_support=True):
    support = dict(amax=max(caps), q_min=.05, q_max=1., z_min=1e-4, z_max=3.)
    if per_spin_support:
        support.update(a1max=caps[0], a2max=caps[1])
    d, chi, logbase = reference_grid(caps, support); comps = []
    for th in thetas:
        u = (d - CENTER[0]) / SCALE[0]; w = (chi - CENTER[1]) / SCALE[1]
        comps.append(dict(theta=list(th), logZ=float(logsumexp(logbase + np.stack([u, w, u * u, u * w, w * w], -1) @ np.asarray(th))), coef=COEF, cov=COV))
    contract = dict(analysis_support=support, spin_caps=list(caps), spin_quadrature_order=128, distance_prior=dict(kind='distance_squared'))
    return dict(schema=b.SCHEMA, coordinates=b.COORDINATES, density_chart=b.DENSITY_CHART, analysis_support=support, prior_contract=contract,
                spin_caps=list(caps), rho_embedded_in_scalar_likelihood=False, center=CENTER, scale=SCALE, components=comps, component_weights=list(weights))


class NativeSpinBaseTests(unittest.TestCase):
    def test_pointwise_prior_quotient(self):
        m = synthetic_model(); r = np.random.default_rng(5); n = 400
        x = np.column_stack([r.uniform(15, 30, n), r.uniform(.1, .25, n), r.uniform(-.9, .9, n), r.uniform(.05, 1., n)])
        delta = np.sqrt(1 - 4 * x[:, 1]); chi = x[:, 2]; z = x[:, 3]; mcd = x[:, 0] * (1 + z); c = m['components'][0]
        u = (delta - CENTER[0]) / SCALE[0]; w = (chi - CENTER[1]) / SCALE[1]; T = np.stack([u, w, u * u, u * w, w * w], -1)
        mean = np.column_stack([np.ones(n), u, w]) @ np.asarray(COEF); logN = multivariate_normal.logpdf(np.column_stack([mcd, np.log(z)]) - mean, mean=[0, 0], cov=COV)
        cos = common_cosmology(); dl = cos.luminosity_distance(z).value; distance = 2 * np.log(dl) + np.log(dlprime(z, cos))
        expected = T @ np.asarray(THETA) - c['logZ'] + logN - (np.log(mcd) - 1.2 * np.log(x[:, 1]) - np.log(2.)) - np.log(z) - distance
        actual = b.log_likelihood_x(x, m)
        self.assertLess(float(np.max(abs(actual - expected))), 2e-10)
        self.assertAlmostEqual(b.log_likelihood_x(x[0], m), actual[0], places=12)
        # Omitting the distance prior changes the shape, not only the event constant.
        diff = (actual + distance) - expected; diff -= np.median(diff); self.assertGreater(float(np.max(abs(diff))), .1)

    def test_bank_restores_native_posterior(self):
        """Bank weights times the full native prior (including pi) recover moments of pi*g*N from independent quadrature."""
        for caps, uf in [((.99, .99), .1), ((.6, .99), .1), ((.6, .99), .6)]:
            m = synthetic_model(caps, thetas=(THETA, [-.4, .5, -.8, 0., -.3]), weights=(.7, .3))
            y, lw, x, meta = b.integration_bank(m, n=20000, seed=73, grid_size=128, uniform_fraction=uf)
            self.assertEqual(x.shape, (20000, 4)); self.assertAlmostEqual(float(logsumexp(lw)), 0., places=10)
            np.testing.assert_allclose(y, masses_from_x(x)); self.assertFalse(meta['rho_embedded'])
            nw = lw + log_native_prior(x, m['prior_contract'], native_pdf, order=128) - log_mass_jacobian(x); nw = np.exp(nw - logsumexp(nw))
            delta = np.sqrt(1 - 4 * x[:, 1]); mcd = x[:, 0] * (1 + x[:, 3])
            d, chi, logbase = reference_grid(caps, m['analysis_support']); u = (d - CENTER[0]) / SCALE[0]; w = (chi - CENTER[1]) / SCALE[1]; T = np.stack([u, w, u * u, u * w, w * w], -1)
            mix = np.stack([np.log(p) + logbase + T @ np.asarray(c['theta']) - c['logZ'] for c, p in zip(m['components'], m['component_weights'])])
            pg = np.exp(logsumexp(mix, 0)); self.assertAlmostEqual(float(pg.sum()), 1., places=6)
            lin = np.column_stack([np.ones(len(d)), u, w]) @ np.asarray(COEF)
            known = [pg @ d, pg @ chi, pg @ lin[:, 0], pg @ lin[:, 1]]
            got = [nw @ delta, nw @ x[:, 2], nw @ mcd, nw @ np.log(x[:, 3])]
            for g_, k_, tol in zip(got, known, [.01, .01, .03, .02]):
                self.assertLess(abs(g_ - k_), tol, (caps, got, known))
            # Second moments in (delta, chi) resolve proposal-weight errors that leave means within tolerance.
            self.assertLess(abs(nw @ x[:, 2] ** 2 / (pg @ chi ** 2) - 1), .05); self.assertLess(abs(nw @ delta ** 2 / (pg @ d ** 2) - 1), .02)
            # Dropping pi from the restoring prior must visibly fail: pi does not cancel there.
            bad = lw + log_native_prior(x, m['prior_contract'], native_pdf, order=128) - log_mass_jacobian(x) - np.log(native_pdf(x[:, 2], (1 - delta) / (1 + delta), *caps))
            bad = np.exp(bad - logsumexp(bad)); self.assertGreater(abs(bad @ x[:, 2] - known[1]) + abs(bad @ delta - known[0]), .02)

    def test_proposal_density_is_normalized(self):
        """Mean of fitted/proposal over bank draws, times acceptance, equals the quadrature integral of sum_j p_j g_j (no pi)."""
        for caps in [(.99, .99), (.6, .99)]:
            m = synthetic_model(caps, thetas=(THETA, [-.4, .5, -.8, 0., -.3]), weights=(.7, .3))
            for uf in [.1, .5]:
                _, _, x, meta = b.integration_bank(m, n=20000, seed=11, grid_size=128, uniform_fraction=uf)
                ratio = np.exp(b.log_fitted(x, m) - b.proposal_logpdf(x, m, grid_size=128, uniform_fraction=uf)) * meta['physical_acceptance']
                lo, hi = b.delta_bounds(m['analysis_support']); nd, wd = np.polynomial.legendre.leggauss(128); d = (lo + hi) / 2 + (hi - lo) / 2 * nd; wd = wd * (hi - lo) / 2
                q = (1 - d) / (1 + d); e = (caps[0] + q * caps[1]) / (1 + q); t = (np.arange(1600) + .5) / 1600 * 2 - 1; chi = e[:, None] * t; dA = (wd * 2 * e / 1600)[:, None]
                u = (d[:, None] - CENTER[0]) / SCALE[0]; w = (chi - CENTER[1]) / SCALE[1]; T = np.stack(np.broadcast_arrays(u, w, u * u, u * w, w * w), -1)
                K = sum(p * float(np.sum(np.exp(T @ np.asarray(c['theta']) - c['logZ']) * dA)) for c, p in zip(m['components'], m['component_weights']))
                err = ratio.std() / np.sqrt(len(ratio)); self.assertLess(abs(ratio.mean() - K), max(5 * err, 1e-3 * K), (caps, uf, ratio.mean(), K, err))
                self.assertLess(abs(ratio.mean() / K - 1), .01, (caps, uf, ratio.mean(), K))

    def test_spin_endpoints_finite_and_support_closed(self):
        caps = (.6, .99); m = synthetic_model(caps, per_spin_support=False); eta = np.array([.2, .2, .25]); delta = np.sqrt(1 - 4 * eta); q = (1 - delta) / (1 + delta); e = (caps[0] + q * caps[1]) / (1 + q)
        inside = np.column_stack([[25.] * 3, eta, [e[0], -e[1], e[2]], [.3] * 3])
        self.assertTrue(np.isfinite(b.log_likelihood_x(inside, m)).all())
        outside = inside.copy(); outside[:, 2] *= 1.001
        self.assertTrue(np.all(b.log_likelihood_x(outside, m) == -np.inf))
        self.assertEqual(b.log_likelihood_x(np.array([25., .2, 0., 3.5]), m), -np.inf)

    def test_rejects_ambiguous_models(self):
        good = synthetic_model(); b.validate_model(good)
        edits = [lambda m: m.update(schema='aligned-redshift-native-spin-cdf-posterior-v1'), lambda m: m.update(rho_embedded_in_scalar_likelihood=True),
                 lambda m: m.update(spin_caps=[.9, .99]), lambda m: m.update(density_chart=['mc_detector', 'delta', 's', 'logz']),
                 lambda m: m.update(component_weights=[.5]), lambda m: m.update(scale=[.14, 0.]),
                 lambda m: m['components'][0].update(cov=[[1., 2.], [2., 1.]]), lambda m: m['components'][0].update(theta=[0., 0., 0.])]
        for edit in edits:
            m = copy.deepcopy(good); edit(m)
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                b.log_likelihood_x(np.array([25., .2, 0., .3]), m)
        with self.assertRaises(ValueError):
            b.integration_bank(good, n=10, uniform_fraction=0.)


    def test_public_data_gate_detects_base_coefficients(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from check_public_data import coefficient_product
        self.assertTrue(coefficient_product(dict(event='e', fit=synthetic_model())))
        self.assertFalse(coefficient_product(dict(event='e', thresholds=dict(cov=1.), theta='x')))


class BaseCollectionDispatchTest(unittest.TestCase):
    def setUp(self):
        import test_collection_registry as registry_fixture
        from public_fixture_certificate import install
        self.f = registry_fixture.RegistryTest('test_complete_identity_and_pairs'); self.f.setUp(); self.addCleanup(self.f.doCleanups); f = self.f
        f.coords = b.COORDINATES; f.manifest.update(coordinates=f.coords, likelihood_backend='native_spin_base_redshift'); self.model = synthetic_model()
        canonical = hashlib.sha256(json.dumps(self.model, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        f.fit.update(coordinates=f.coords, fit=self.model, upstream_model_sha256='a' * 64, model_canonical_sha256=canonical); fit = f.write('fit.json', f.fit)
        f.manifest['entries'][0]['fit'] = fit; f.manifest['entries'][0]['acceptance']['status'] = 'public_source_verified_candidate'
        original = f.write('original-validation.json', dict(event='event-1', model_sha256='a' * 64, status='passed_declared_recovery_checks', metrics={}))
        proof = dict(schema='native-redshift-candidate-validation-v1', status='candidate_unvalidated', event='event-1', analysis_id='analysis-1', fit_sha256=fit['sha256'],
                     coordinates=f.coords, upstream_model_sha256='a' * 64, model_canonical_sha256=canonical, original_validation=original)
        f.manifest['entries'][0]['validation'] = f.write('proof.json', proof); install(f.root, fit['sha256'], 'event-1', source_model_sha256='a' * 64); f.save()

    def test_dispatch_matches_leaf_and_requires_candidate_flag(self):
        c = self.f.load(); x = np.array([[25., .2, .03, .4], [20., .24, -.1, .2]])
        with self.assertRaisesRegex(ValueError, 'not accepted'):
            c.log_likelihood('event-1', x)
        np.testing.assert_allclose(c.log_likelihood('event-1', x, allow_candidate=True), b.log_likelihood_x(x, self.model), atol=1e-12)
        self.assertEqual(c.verify()['analyses'], 1)

    def test_public_certificate_still_required(self):
        (self.f.root / 'provenance/public_data_certificates.json').unlink()
        with self.assertRaisesRegex(ValueError, 'certificate'):
            self.f.load().log_likelihood('event-1', [25., .2, 0., .3], allow_candidate=True)

    def test_candidate_evidence_cannot_grant_acceptance(self):
        self.f.manifest['entries'][0]['acceptance']['status'] = 'accepted_for_declared_target'; self.f.save()
        with self.assertRaisesRegex(ValueError, 'passing recovery'):
            self.f.load().fit('event-1')


if __name__ == '__main__':
    unittest.main()
