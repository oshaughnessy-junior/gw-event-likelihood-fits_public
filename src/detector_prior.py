"""Declared detector-frame retained prior and uncapped integration proposals.

The likelihood retains native conditional nuisance priors. This is not the
full-state log_prior column. Coordinates are Mc_detector, eta, chi_eff, dL.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from prior_folded_likelihood import _chi_prior
from detector_distance import canonical_mass_support, _mass_bounds


def validate_config(config):
    b = np.asarray(config['bounds'], float)
    if b.shape != (4, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]):
        raise ValueError('bounds must be finite increasing 4x2 intervals')
    if b[0, 0] <= 0 or b[1, 0] <= 0 or b[1, 1] > .25 or b[3, 0] <= 0:
        raise ValueError('Nonphysical mass, eta or distance bounds')
    q0, q1 = config['q_range']
    if not 0 < q0 < q1 <= 1 or not 0 < config['amax'] <= 1:
        raise ValueError('Invalid q_range or amax')
    if config.get('mass_measure') != 'uniform_detector_component_masses':
        raise ValueError('Only verified uniform detector component-mass priors are supported')
    if config.get('spin_measure') != 'uniform_magnitudes_isotropic':
        raise ValueError('Only verified isotropic, uniform spin magnitudes are supported')
    for key in ['prior_evidence', 'nuisance_prior']:
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('Missing explicit ' + key)
    for key, interval in _mass_bounds(config.get('mass_support')).items():
        if key not in ['m1', 'm2', 'mtotal'] or not 0 <= interval[0] < interval[1]:
            raise ValueError('Invalid detector mass constraint ' + key)
    d, p = distance_table(config)
    if b[3, 0] < d[0] or b[3, 1] > d[-1]:
        raise ValueError('Distance table must cover declared support')
    holes = (p[:-1] == 0) & (p[1:] == 0) & (d[1:] > b[3, 0]) & (d[:-1] < b[3, 1])
    if holes.any():
        raise ValueError('Positive-width zero-prior distance gaps cannot be represented by the model support')
    if int(config.get('spin_quadrature_order', 64)) < 16:
        raise ValueError('Spin quadrature order must be at least 16')
    qlo = max(q0, eta_to_q(b[1, 0]))
    qhi = min(q1, eta_to_q(b[1, 1]))
    if qlo >= qhi or max(b[2, 0], -config['amax']) >= min(b[2, 1], config['amax']):
        raise ValueError('Empty retained support')
    return config


def effective_model_support(config):
    """Intersection of every retained native cut for saved scalar evaluation.

    Only box cuts are absorbed here; coupled mass cuts remain explicit. Keeping
    q=1 in the closure preserves a finite scalar likelihood at eta=1/4.
    """
    validate_config(config)
    b = np.array(config['bounds'], dtype=float, copy=True)
    qlo, qhi = config['q_range']
    b[1, 0] = max(b[1, 0], qlo/(1+qlo)**2)
    b[1, 1] = min(b[1, 1], qhi/(1+qhi)**2)
    b[2, 0] = max(b[2, 0], -config['amax'])
    b[2, 1] = min(b[2, 1], config['amax'])
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError('Empty native retained support')
    return dict(bounds=b.tolist(), mass_support=canonical_mass_support(config.get('mass_support')))


def distance_table(config):
    tab = config['distance_prior']
    d, p = np.asarray(tab['d_l'], float), np.asarray(tab['density'], float)
    if d.ndim != 1 or len(d) < 2 or d.shape != p.shape or not np.isfinite(d).all() or not np.isfinite(p).all() or np.any(np.diff(d) <= 0) or np.any(p < 0):
        raise ValueError('Distance prior requires increasing d_l and nonnegative density')
    integral = np.trapezoid(p, d) if hasattr(np, 'trapezoid') else np.trapz(p, d)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError('Distance density has zero integral')
    return d, p / integral


def eta_to_q(eta):
    eta = np.asarray(eta)
    return 2 * eta / (1 - 2 * eta + np.sqrt(np.maximum(0, 1 - 4 * eta)))


def retained_logprior(x, config):
    """Log induced prior in dMc_det dEta dChi_eff dD_L, with coupled cuts.

    Overall mass/spin normalization is irrelevant to likelihood shape fitting.
    The eta=1/4 endpoint has measure zero and is excluded from evaluation.
    """
    a = np.asarray(x, float)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError('Expected Nx4 paired joint coordinates')
    b = np.asarray(config['bounds'])
    mc, eta, chi, dl = a.T
    ok = np.isfinite(a).all(1) & ((a >= b[:, 0]) & (a <= b[:, 1])).all(1)
    ok &= (mc > 0) & (eta > 0) & (eta < .25) & (abs(chi) < config['amax'])
    idx = np.flatnonzero(ok)
    q = eta_to_q(eta[idx])
    m1 = mc[idx] * (1 + q)**.2 / q**.6
    masses = {'m1': m1, 'm2': m1*q, 'mtotal': m1*(1+q), 'total': m1*(1+q)}
    keep = (q >= config['q_range'][0]) & (q <= config['q_range'][1])
    for key, (lo, hi) in _mass_bounds(config.get('mass_support')).items():
        keep &= (masses[key] >= lo) & (masses[key] <= hi)
    idx, q = idx[keep], q[keep]
    out = np.full(len(a), -np.inf)
    if len(idx):
        pc = np.concatenate([_chi_prior(chi[idx[j:j+4096]], q[j:j+4096], config['amax'], int(config.get('spin_quadrature_order', 64))) for j in range(0, len(idx), 4096)])
        grid, density = distance_table(config)
        pd = np.interp(dl[idx], grid, density, left=0, right=0)
        with np.errstate(divide='ignore'):
            out[idx] = np.log(mc[idx]) - 1.2*np.log(eta[idx]) - .5*np.log1p(-4*eta[idx]) + np.log(pc) + np.log(pd)
    return out


def _proposal(posterior_x, config):
    a = np.asarray(posterior_x, float)
    if a.ndim != 2 or a.shape[1] != 4 or len(a) < 5 or not np.isfinite(a).all():
        raise ValueError('Proposal requires at least five finite paired Nx4 rows')
    b = np.asarray(config['bounds'], float)
    center = np.mean(a, axis=0)
    scale = np.std(a, axis=0)
    floor = (b[:, 1]-b[:, 0])*1e-5
    scale = np.maximum(scale, floor)
    z = (a-center)/scale
    corr = np.cov(z.T) + np.eye(4)*1e-4
    cov = corr*np.outer(scale, scale)
    lo = b[:, 0].copy(); hi = b[:, 1].copy()
    lo[1] = max(config['q_range'][0], eta_to_q(b[1, 0]))
    hi[1] = min(config['q_range'][1], eta_to_q(b[1, 1]))
    lo[2] = max(lo[2], -config['amax']); hi[2] = min(hi[2], config['amax'])
    return center, cov, lo, hi


def proposal_logpdf(x, proposal):
    """Exact .4/.3/.3 Student-t, broad Student-t, uniform-in-q mixture."""
    a = np.asarray(x, float); c, cov, lo, hi = [np.asarray(proposal[k]) for k in ['center','cov','box_lower_mc_q_chi_dl','box_upper_mc_q_chi_dl']]
    logs = [np.log(.4)+multivariate_t.logpdf(a, loc=c, shape=cov, df=5), np.log(.3)+multivariate_t.logpdf(a, loc=c, shape=16*cov, df=5)]
    eta = a[:, 1]; q = eta_to_q(np.clip(eta, 0, .25))
    transformed = a.copy(); transformed[:, 1] = q
    inside = (eta > 0) & (eta < .25) & ((transformed >= lo) & (transformed <= hi)).all(1)
    box = np.full(len(a), -np.inf)
    with np.errstate(divide='ignore'):
        box[inside] = np.log(.3)-np.log(hi-lo).sum()+3*np.log1p(q[inside])-np.log1p(-q[inside])
    return logsumexp(np.stack(logs+[box]), axis=0)


def build_integration_bank(posterior_x, config, n=32768, seed=29):
    """Independent proposal rows; weights are never capped or resampled.

    Uniform q gives integrable coverage of the eta=1/4 Jacobian divergence.
    Proposal moments are numerical integration aids, not fitted likelihoods.
    """
    validate_config(config)
    if n < 1:
        raise ValueError('n must be positive')
    c, cov, lo, hi = _proposal(posterior_x, config)
    p = {'center':c.tolist(), 'cov':cov.tolist(), 'box_lower_mc_q_chi_dl':lo.tolist(), 'box_upper_mc_q_chi_dl':hi.tolist(), 'mixture_weights':[.4,.3,.3], 'student_df':5, 'broad_scale':4, 'seed':int(seed)}
    rng = np.random.default_rng(seed); labels = rng.choice(3, n, p=[.4,.3,.3]); x = np.empty((n,4))
    for j in [0,1]:
        ix = labels == j
        if ix.any():
            x[ix] = multivariate_t.rvs(loc=c, shape=cov*(16 if j else 1), df=5, size=int(ix.sum()), random_state=rng).reshape(-1,4)
    ix = labels == 2
    x[ix] = rng.uniform(lo, hi, size=(int(ix.sum()),4))
    q = x[ix,1].copy(); x[ix,1] = q/(1+q)**2
    logq = proposal_logpdf(x,p)
    if not np.isfinite(logq).all():
        raise ValueError('Nonfinite mixture proposal density')
    return {'x':x, 'log_prior_minus_proposal':retained_logprior(x,config)-logq, 'proposal':p}
