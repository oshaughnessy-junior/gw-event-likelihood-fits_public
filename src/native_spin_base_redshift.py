"""Native-posterior 4D leaf with the native effective-spin prior as base measure.

Density chart v=(Mc_detector, delta, chi_eff, log z), delta=sqrt(1-4 eta).
The fitted native posterior density in v is

    f(v) = pi(chi_eff|q) * sum_j p_j g_j(delta, chi_eff) N_j(Mc_detector, log z | delta, chi_eff),
    g_j  = exp(theta_j . T(delta, chi_eff) - logZ_j),

with T the quadratic features (u, w, u^2, u w, w^2) of the standardized
u=(delta-center_0)/scale_0, w=(chi_eff-center_1)/scale_1, and logZ_j normalizing
pi*exp(theta_j . T) over delta and chi_eff. N_j is a bivariate normal whose mean is
linear in (u, w). pi is the native conditional effective-spin prior for independent
uniform spin magnitudes below the recorded caps and isotropic tilts.

The native retained prior in v is pi(chi_eff|q) * J_detector_delta * z * p_DL * dDL/dz
(up to an event constant), so pi cancels exactly in the scalar likelihood: no
density is divided by a vanishing spin prior and the chi_eff dependence of L is
the quadratic exponent. Mass and distance prior factors are divided out as in
native_posterior_redshift. Each logZ_j is a fitted constant stored with the model.
"""
import numpy as np
from scipy.special import logsumexp

from aligned_redshift import masses_from_x, support
from native_posterior_redshift import distance_logfactor

SCHEMA = 'aligned-redshift-native-spin-base-posterior-v1'
COORDINATES = ['mc_source', 'eta', 'chi_eff', 'z']
DENSITY_CHART = ['mc_detector', 'delta', 'chi_eff', 'logz']


def chart(x):
    x = np.asarray(x, float); delta = np.sqrt(np.maximum(1 - 4 * x[:, 1], 0)); q = (1 - delta) / (1 + delta)
    return x[:, 0] * (1 + x[:, 3]), delta, x[:, 2], np.log(x[:, 3]), q


def edge(q, caps):
    """Largest |chi_eff| allowed at mass ratio q by spin-magnitude caps (a1, a2)."""
    return (caps[0] + q * caps[1]) / (1 + q)


def feats(delta, chi, center, scale):
    u = (delta - center[0]) / scale[0]; w = (chi - center[1]) / scale[1]
    return np.stack([u, w, u * u, u * w, w * w], -1)


def delta_bounds(analysis_support):
    lo, hi = [analysis_support.get(k) if analysis_support.get(k) is not None else d for k, d in [('q_min', 0.), ('q_max', 1.)]]
    eb = [lo / (1 + lo) ** 2, hi / (1 + hi) ** 2]
    return float(np.sqrt(max(0, 1 - 4 * eb[1]))), float(np.sqrt(max(0, 1 - 4 * eb[0])))


def validate_model(model):
    """Reject ambiguous charts, priors or parameters before evaluating a model."""
    if model.get('schema') != SCHEMA or list(model.get('density_chart', [])) != DENSITY_CHART or list(model.get('coordinates', [])) != COORDINATES:
        raise ValueError('Unsupported native-spin base-measure leaf')
    caps = np.asarray(model['spin_caps'], float)
    if caps.shape != (2,) or not np.isfinite(caps).all() or np.any(caps <= 0) or np.any(caps > 1):
        raise ValueError('Spin caps must be two finite values in (0,1]')
    if not np.array_equal(caps, np.asarray(model['prior_contract']['spin_caps'], float)):
        raise ValueError('Native spin prior and base-measure caps disagree')
    if model.get('rho_embedded_in_scalar_likelihood', False) is not False:
        raise ValueError('Scalar likelihood must not embed a population redshift law')
    center = np.asarray(model['center'], float); scale = np.asarray(model['scale'], float)
    if center.shape != (2,) or scale.shape != (2,) or not np.isfinite(center).all() or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('Invalid feature standardization')
    weights = np.asarray(model['component_weights'], float)
    if weights.shape != (len(model['components']),) or not len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid base-measure mixture weights')
    for c in model['components']:
        theta = np.asarray(c['theta'], float); coef = np.asarray(c['coef'], float); cov = np.asarray(c['cov'], float)
        if theta.shape != (5,) or coef.shape != (3, 2) or cov.shape != (2, 2) or not np.isfinite(c['logZ']):
            raise ValueError('Invalid base-measure component shape')
        if not (np.isfinite(theta).all() and np.isfinite(coef).all() and np.isfinite(cov).all()) or not np.allclose(cov, cov.T):
            raise ValueError('Invalid base-measure component parameters')
        np.linalg.cholesky(cov)


def log_conditional(uv, Y, coef, cov):
    r = Y - np.column_stack([np.ones(len(uv)), uv]) @ coef
    return -.5 * np.einsum('ni,ij,nj->n', r, np.linalg.inv(cov), r) - np.log(2 * np.pi) - .5 * np.linalg.slogdet(cov)[1]


def component_terms(x, model):
    """log p_j + log g_j + log N_j per row and component; pi omitted (common factor)."""
    mcd, d, chi, lz, _ = chart(x); c0, s0 = np.asarray(model['center']), np.asarray(model['scale'])
    T = feats(d, chi, c0, s0); uv = T[:, :2]; Y = np.column_stack([mcd, lz]); out = []
    for c, p in zip(model['components'], model['component_weights']):
        out.append(np.log(p) + T @ np.asarray(c['theta']) - c['logZ'] + log_conditional(uv, Y, np.asarray(c['coef']), np.asarray(c['cov'])))
    return np.stack(out, 1)


def log_fitted(x, model):
    """log sum_j p_j g_j N_j: native posterior density over pi(chi_eff|q), in the density chart."""
    return logsumexp(component_terms(x, model), 1)


def log_likelihood_x(x, model):
    """Scalar log likelihood at x=[Mc_source, eta, chi_eff, z]; arbitrary event constant."""
    validate_model(model)
    x = np.asarray(x, float); scalar = x.ndim == 1; a = x.reshape(-1, 4); good = support(a, model); out = np.full(len(a), -np.inf)
    if good.any():
        b = a[good]; y = masses_from_x(b); q = y[:, 1] / y[:, 0]; valid = abs(b[:, 2]) <= edge(q, model['spin_caps'])
        logJ = np.log(b[:, 0] * (1 + b[:, 3])) - 1.2 * np.log(b[:, 1]) - np.log(2.)
        value = log_fitted(b, model) - logJ - np.log(b[:, 3]) - distance_logfactor(b[:, 3], model['prior_contract'])
        out[good] = np.where(valid, value, -np.inf)
    return float(out[0]) if scalar else out.reshape(x.shape[:-1])


def proposal_tables(model, grid_size=256):
    """Per-component cell probabilities of g_j (no pi) on a (delta, t=chi/edge) grid, and their delta marginals."""
    caps = model['spin_caps']; lo, hi = delta_bounds(model['analysis_support']); c0, s0 = np.asarray(model['center']), np.asarray(model['scale'])
    de = np.linspace(lo, hi, grid_size + 1); te = np.linspace(-1, 1, grid_size + 1); dm = (de[:-1] + de[1:]) / 2; tm = (te[:-1] + te[1:]) / 2
    D, Tt = np.meshgrid(dm, tm, indexing='ij'); E = edge((1 - D) / (1 + D), caps); Fg = feats(D.ravel(), (Tt * E).ravel(), c0, s0)
    tables = []
    for c in model['components']:
        lp = Fg @ np.asarray(c['theta']) + np.log(E.ravel()); prob = .9995 * np.exp(lp - logsumexp(lp)) + .0005 / len(lp)
        tables.append((prob, prob.reshape(grid_size, grid_size).sum(1)))
    return de, te, tables


def mixture_probabilities(model):
    p = np.asarray(model['component_weights']); return .9999 * p / p.sum() + .0001 / len(p)


def proposal_logpdf(x, model, grid_size=256, uniform_fraction=.1, tables=None):
    """Log density of the integration_bank proposal in the chart (Mc_det, delta, chi_eff, log z)."""
    de, te, tables = proposal_tables(model, grid_size) if tables is None else tables
    caps = model['spin_caps']; c0, s0 = np.asarray(model['center']), np.asarray(model['scale']); p = mixture_probabilities(model); area = (de[1] - de[0]) * (te[1] - te[0])
    mcd, d, chi, lz, q = chart(x); e = edge(q, caps); uv = feats(d, chi, c0, s0)[:, :2]; Y = np.column_stack([mcd, lz])
    di = np.clip(np.searchsorted(de, d, side='right') - 1, 0, grid_size - 1); ti = np.clip(np.searchsorted(te, chi / e, side='right') - 1, 0, grid_size - 1); terms = []
    for c, (prob, pd), wt in zip(model['components'], tables, p):
        lc = log_conditional(uv, Y, np.asarray(c['coef']), np.asarray(c['cov'])) + np.log(wt)
        a = np.log1p(-uniform_fraction) + np.log(prob[di * grid_size + ti]) - np.log(area * e)
        b = np.log(uniform_fraction) + np.log(pd[di]) - np.log(de[1] - de[0]) - np.log(2 * e)
        terms.append(lc + np.logaddexp(a, b))
    return logsumexp(np.stack(terms), 0)


def integration_bank(model, n=32768, seed=1064, grid_size=256, uniform_fraction=.1):
    """Known proposal: per component, (delta, t=chi/edge) from a table of g_j (no pi), or chi uniform; (Mc_det, logz) Gaussian.

    Returns (y, logw, x, metadata), y=[m1_source, m2_source, chi_eff, z], with weights for
    integration in dm1_source dm2_source dchi_eff dz. No population redshift law is included.
    """
    validate_model(model)
    if not 0 < uniform_fraction < 1 or n < 1 or grid_size < 2:
        raise ValueError('Invalid integration proposal settings')
    rng = np.random.default_rng(seed); caps = model['spin_caps']; c0, s0 = np.asarray(model['center']), np.asarray(model['scale']); p = mixture_probabilities(model)
    de, te, tables = proposal_tables(model, grid_size)
    xs = []; count = drawn = 0
    while count < n:
        kk = max(1024, int(np.ceil(1.2 * (n - count)))); lab = rng.choice(len(p), kk, p=p); flat = rng.random(kk) < uniform_fraction; x = np.empty((kk, 4))
        for j, c in enumerate(model['components']):
            ids = np.flatnonzero(lab == j)
            if not len(ids):
                continue
            prob, pd = tables[j]; fl = flat[ids]; d = np.empty(len(ids)); t = np.empty(len(ids))
            cells = rng.choice(len(prob), int((~fl).sum()), p=prob); di, ti = cells // grid_size, cells % grid_size
            d[~fl] = de[di] + rng.random(len(cells)) * (de[1] - de[0]); t[~fl] = te[ti] + rng.random(len(cells)) * (te[1] - te[0])
            di = rng.choice(grid_size, int(fl.sum()), p=pd); d[fl] = de[di] + rng.random(len(di)) * (de[1] - de[0]); t[fl] = 2 * rng.random(fl.sum()) - 1
            chi = t * edge((1 - d) / (1 + d), caps); uv = feats(d, chi, c0, s0)[:, :2]
            mean = np.column_stack([np.ones(len(ids)), uv]) @ np.asarray(c['coef']); Yv = mean + rng.normal(size=(len(ids), 2)) @ np.linalg.cholesky(np.asarray(c['cov'])).T
            z = np.exp(Yv[:, 1]); x[ids] = np.column_stack([Yv[:, 0] / (1 + z), (1 - d * d) / 4, chi, z])
        good = support(x, model); xs.append(x[good]); count += int(good.sum()); drawn += kk
        if drawn > max(100000, 100 * n):
            raise RuntimeError('Base-measure proposal support unresolved')
    x = np.vstack(xs)[:n]
    logq = proposal_logpdf(x, model, grid_size, uniform_fraction, (de, te, tables))
    ld = distance_logfactor(x[:, 3], model['prior_contract'])
    # Fitted density over pi and the native mass/distance prior, times the physical-mass chart Jacobian.
    lw = log_fitted(x, model) - logq - 2 * np.log1p(x[:, 3]) - ld - np.log(n)
    if not np.isfinite(lw).all():
        raise RuntimeError('Nonfinite base-measure likelihood bank')
    lw -= logsumexp(lw)
    return masses_from_x(x), lw, x, dict(n=n, seed=seed, rho_embedded=False, uniform_physical_chi_fraction=uniform_fraction, physical_acceptance=count / drawn,
                                         density_chart='Mc_detector,delta,chi_eff,logz', integration_measure='dm1_source dm2_source dchi_eff dz',
                                         event_normalization='Arbitrary fixed event constant; no absolute evidence')
