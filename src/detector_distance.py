"""Full 4D bounded quadratic *scalar likelihood* models.

The default 14 natural coefficients multiply four linear and ten upper-triangle
quadratic features, with no factor of two on cross terms. A positive quadratic
direction is permitted on the finite physical box; this is not necessarily a
Gaussian. Optional distance cubic splines add distance flexibility and intrinsic
linear interactions. Changing distance coordinate changes the feature map, never adds a
density Jacobian to a scalar likelihood. Hidden nuisance priors remain part of
the declared marginal likelihood target.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

COORDINATES = ['mc_det', 'eta', 'chi_eff', 'd_l']
PAIRS = tuple((i, j) for i in range(4) for j in range(i, 4))
SCHEMA = 'detector-distance-quadratic-v1'
SPLINE_SCHEMA = 'detector-distance-spline-v1'


def _points(x):
    a = np.asarray(x, dtype=float)
    scalar = a.ndim == 1
    if scalar:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError('Coordinates must have shape (N,4) or (4,)')
    return a, scalar


def _bounds(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (4, 2) or not np.isfinite(b).all() or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError('Require finite increasing 4x2 physical bounds')
    if b[0, 0] < 0 or b[1, 0] < 0 or b[1, 1] > .25 or b[2, 0] < -1 or b[2, 1] > 1 or b[3, 0] <= 0:
        raise ValueError('Bounds violate physical mass, eta, spin or distance support')
    return b


def canonical_mass_support(mass_support):
    """Normalize legacy total aliases without silently overriding a cut."""
    if mass_support is None:
        return None
    if not isinstance(mass_support, dict):
        raise ValueError('mass_support must be a dictionary')
    result = {}
    def comparable(key, value):
        if key.endswith(('_min', '_max')):
            return float(value)
        if isinstance(value, dict):
            return (float(value.get('min', 0.)), float(value.get('max', np.inf)))
        return tuple(map(float, value))
    for key, value in mass_support.items():
        canonical = 'mtotal'+key[5:] if key in ('total', 'total_min', 'total_max') else key
        if canonical in result and comparable(canonical, result[canonical]) != comparable(canonical, value):
            raise ValueError('Conflicting total/mtotal mass-support aliases')
        result[canonical] = value
    return result


def _mass_bounds(mass_support):
    if mass_support is None:
        return {}
    mass_support = canonical_mass_support(mass_support)
    valid = {k+s for k in ['m1', 'm2', 'mtotal'] for s in ['', '_min', '_max']}
    if set(mass_support)-valid:
        raise ValueError('Unknown detector mass-support key')
    result = {}
    for k in ['m1', 'm2', 'mtotal']:
        lo, hi = 0., np.inf
        if k in mass_support:
            v = mass_support[k]
            if isinstance(v, dict):
                if set(v)-{'min', 'max'}:
                    raise ValueError('Unknown mass-bound key')
                lo, hi = v.get('min', lo), v.get('max', hi)
            else:
                if len(v) != 2:
                    raise ValueError('Mass bound requires [min,max]')
                lo, hi = v
        lo, hi = float(mass_support.get(k+'_min', lo)), float(mass_support.get(k+'_max', hi))
        if np.isnan(lo) or np.isnan(hi) or lo < 0 or hi <= lo:
            raise ValueError('Invalid detector mass bounds')
        if k in mass_support or k+'_min' in mass_support or k+'_max' in mass_support:
            result[k] = (lo, hi)
    return result


def support(x, bounds, mass_support=None):
    """Physical support in original detector coordinates, before any transform."""
    a, scalar = _points(x)
    b = _bounds(bounds)
    mask = np.isfinite(a).all(axis=1) & (a >= b[:, 0]).all(axis=1) & (a <= b[:, 1]).all(axis=1)
    mask &= (a[:, 0] > 0) & (a[:, 1] > 0) & (a[:, 1] <= .25) & (np.abs(a[:, 2]) <= 1) & (a[:, 3] > 0)
    mb = _mass_bounds(mass_support)
    if mb:
        eta = np.where(mask, a[:, 1], .25)
        mt = np.where(mask, a[:, 0], 1.) * eta**(-3./5.)
        delta = np.sqrt(np.maximum(1.-4.*eta, 0.))
        masses = {'m1': mt*(1.+delta)/2., 'm2': mt*(1.-delta)/2., 'mtotal': mt}
        for name, (lo, hi) in mb.items():
            mask &= np.isfinite(masses[name]) & (masses[name] >= lo) & (masses[name] <= hi)
    return bool(mask[0]) if scalar else mask


def _transform(x, coordinate):
    a = np.array(x, dtype=float, copy=True)
    if coordinate == 'inverse':
        a[:, 3] = 1./a[:, 3]
    elif coordinate == 'log':
        a[:, 3] = np.log(a[:, 3])
    elif coordinate != 'linear':
        raise ValueError('distance_coordinate must be inverse, linear or log')
    return a


def _features(x, model):
    z = (_transform(x, model['distance_coordinate'])-np.asarray(model['center']))/np.asarray(model['scale'])
    columns = [z] + [z[:, i]*z[:, j] for i, j in PAIRS]
    if model['schema'] == SPLINE_SCHEMA:
        distance = [z[:, 3]**3]+[np.maximum(z[:, 3]-k, 0.)**3 for k in model['spline_knots']]
        for basis in distance:
            columns.extend([basis]+[basis*z[:, j] for j in range(3)])
    return np.column_stack(columns)


def _template(bounds, distance_coordinate, nuisance_prior, mass_support, distance_knots=0):
    if isinstance(distance_knots, bool) or not isinstance(distance_knots, (int, np.integer)) or not 0 <= distance_knots <= 20:
        raise ValueError('distance_knots must be an integer from 0 through 20')
    b = _bounds(bounds)
    mass_support = canonical_mass_support(mass_support)
    _mass_bounds(mass_support)
    if not isinstance(nuisance_prior, str) or not nuisance_prior.strip():
        raise ValueError('An explicit nuisance_prior description is required')
    corners = _transform(b.T, distance_coordinate)
    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
    model = {'schema': SPLINE_SCHEMA if distance_knots else SCHEMA, 'coordinates': COORDINATES.copy(), 'bounds': b.tolist(),
            'mass_support': mass_support, 'distance_coordinate': distance_coordinate,
            'nuisance_prior': nuisance_prior, 'center': ((lo+hi)/2).tolist(),
            'scale': ((hi-lo)/2).tolist(), 'coefficients': [0.]*14, 'log_offset': 0.,
            'coefficient_order': ['linear_'+str(i) for i in range(4)]+['quadratic_%d_%d'%p for p in PAIRS]}
    if distance_knots:
        model['spline_knots'] = [0.]*distance_knots  # replaced with TRAIN quantiles before use
        model['spline_basis'] = 'z_d^3, (z_d-k)_+^3; each alone then multiplied by z_0,z_1,z_2'
        model['coefficients'] += [0.]*(4*(distance_knots+1))
        for j in range(distance_knots+1):
            model['coefficient_order'] += ['distance_cubic_%d_times_%s'%(j, k) for k in ['1', 'z0', 'z1', 'z2']]
    return model


def _validate_model(model):
    if model.get('schema') not in [SCHEMA, SPLINE_SCHEMA] or model.get('coordinates') != COORDINATES:
        raise ValueError('Unsupported detector-distance model schema or coordinates')
    _template(model['bounds'], model['distance_coordinate'], model['nuisance_prior'], model.get('mass_support'))
    size = 14
    if model['schema'] == SPLINE_SCHEMA:
        knots = np.asarray(model.get('spline_knots'), float)
        if knots.ndim != 1 or not 1 <= len(knots) <= 20 or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0):
            raise ValueError('Spline knots must be finite, strictly increasing training quantiles')
        size += 4*(len(knots)+1)
    for key, size in [('center', 4), ('scale', 4), ('coefficients', size)]:
        v = np.asarray(model[key], dtype=float)
        if v.shape != (size,) or not np.isfinite(v).all() or (key == 'scale' and np.any(v <= 0)):
            raise ValueError('Invalid model '+key)
    if not np.isfinite(float(model['log_offset'])):
        raise ValueError('Invalid model log_offset')


def log_shape(x, model):
    """Evaluate scalar log likelihood; outside ORIGINAL declared support is -inf."""
    _validate_model(model)
    a, scalar = _points(x)
    valid = support(a, model['bounds'], model.get('mass_support'))
    out = np.full(len(a), -np.inf)
    if valid.any():
        out[valid] = _features(a[valid], model)@np.asarray(model['coefficients']) + float(model['log_offset'])
    return float(out[0]) if scalar else out


def _weights(weights, n, name):
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not w.sum() > 0:
        raise ValueError('Invalid '+name)
    return w/w.sum()


def _condition_on_training(model, x, weights):
    """Numerical feature chart only, NEVER fitted Gaussian parameters.

    Training samples set the origin and units so a broad declared support does
    not render inverse-distance features almost collinear on posterior support.
    Integration and held-out rows never determine this conditioning chart.
    """
    t = _transform(x, model['distance_coordinate'])
    center = weights@t
    scale = np.sqrt(weights@((t-center)**2))
    box_span = 2*np.asarray(model['scale'])
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= box_span*1e-12):
        raise ValueError('Training rows do not resolve all four coordinate directions')
    model['center'] = center.tolist()
    model['scale'] = scale.tolist()
    model['conditioning'] = 'training-only transformed-coordinate mean and standard deviation; numerical feature units, not Gaussian shape parameters'


def _setup_spline(model, x, weights):
    if model['schema'] != SPLINE_SCHEMA:
        return
    z = (_transform(x, model['distance_coordinate'])[:, 3]-model['center'][3])/model['scale'][3]
    order = np.argsort(z)
    values, w = z[order], weights[order]
    keep = w > 0
    values, w = values[keep], w[keep]
    quantiles = np.arange(1, len(model['spline_knots'])+1)/(len(model['spline_knots'])+1.)
    knots = np.interp(quantiles, np.cumsum(w)-w/2, values)
    if np.any(np.diff(knots) <= 0):
        raise ValueError('Training distance sample cannot identify distinct spline knots')
    model['spline_knots'] = knots.tolist()
    model['spline_knot_selection'] = 'weighted quantiles of training-only standardized distance coordinate'


def _feature_whitening(features, weights):
    mean = weights@features
    centered = features-mean
    _, singular, vt = np.linalg.svd(centered*np.sqrt(weights)[:, None], full_matrices=False)
    keep = singular > singular[0]*1e-10
    if not keep.any():
        raise ValueError('Training data do not identify likelihood-shape features')
    whitening = vt[keep].T/singular[keep]
    return mean, whitening, singular, int(keep.sum())


def _family_diagnostics(model):
    if model['schema'] == SPLINE_SCHEMA:
        return {'family': 'bounded-distance-spline-with-intrinsic-cross-terms',
                'scientific_acceptance': 'not_assessed'}
    return _curvature_diagnostics(model['coefficients'])


def _curvature_diagnostics(coefficients):
    h = np.zeros((4, 4))
    for value, (i, j) in zip(coefficients[4:], PAIRS):
        h[i, j] = h[j, i] = value if i != j else 2*value
    eig = np.linalg.eigvalsh(h)
    return {'transformed_log_shape_hessian_eigenvalues': eig.tolist(),
            'negative_definite_quadratic': bool(np.all(eig < 0)),
            'family': 'bounded-full-quadratic', 'scientific_acceptance': 'not_assessed'}


def fit_loglikelihood(x, logl, bounds, *, distance_coordinate='inverse', nuisance_prior,
                      mass_support=None, weights=None, distance_knots=0):
    """Direct weighted regression of supplied scalar/marginal log likelihoods.

    The caller must establish that each logl has the nuisance target described
    by nuisance_prior. Hidden full-state logL values are not marginalized merely
    by omitting their hidden coordinates in this regression.
    """
    model = _template(bounds, distance_coordinate, nuisance_prior, mass_support, distance_knots)
    a, _ = _points(x)
    y = np.asarray(logl, dtype=float)
    if y.shape != (len(a),) or not np.isfinite(y).all() or not np.all(support(a, bounds, mass_support)):
        raise ValueError('Nonfinite or out-of-support direct likelihood rows')
    w = _weights(weights, len(a), 'regression weights')
    _condition_on_training(model, a, w)
    _setup_spline(model, a, w)
    f = _features(a, model)
    target, whitening, singular, rank = _feature_whitening(f, w)
    if rank < f.shape[1]:
        raise ValueError('Direct likelihood design does not identify all requested shape terms')
    design = (f-target)@whitening
    y0 = float(w@y)
    beta = (design*np.sqrt(w)[:, None]).T@((y-y0)*np.sqrt(w))
    coeff = whitening@beta
    model['coefficients'] = coeff.tolist()
    model['log_offset'] = float(y0-target@coeff)
    residual = y-log_shape(a, model)
    diag = {'method': 'direct-loglikelihood-weighted-least-squares', 'n_rows': len(a),
            'design_rank': int(rank+1), 'feature_rank': rank, 'feature_count': f.shape[1],
            'design_condition_number': float(singular[0]/singular[-1]),
            'optimization_feature_condition_number': 1.,
            'feature_whitening': 'training-only weighted centered SVD',
            'weighted_loglikelihood_rms': float(np.sqrt(w@(residual**2))),
            'maximum_abs_loglikelihood_residual': float(np.max(np.abs(residual))),
            'regression_weight_ess': float(1./(w@w))}
    solved = bool(rank == f.shape[1] and np.isfinite(coeff).all()
                  and np.isfinite(model['log_offset']) and np.isfinite(residual).all())
    diag.update(numerical_regression_gate=solved, numerical_optimizer_gate=solved,
                numerical_gate_kind='full-rank finite direct regression; no optimizer is involved')
    diag.update(_family_diagnostics(model))
    return model, diag


def fit_forward(posterior_x, integration_x, integration_log_weights, bounds, *,
                distance_coordinate='inverse', nuisance_prior, mass_support=None,
                posterior_weights=None, maxiter=500, distance_knots=0):
    """Fit exp(log_shape)*known prior to posterior samples using a fixed integral.

    distance_knots>0 adds training-quantile cubic distance splines and intrinsic
    linear interactions. All optimizer features are whitened using training
    samples only; stored coefficients are converted back to the declared basis.

    integration_x must be SEPARATE known-proposal draws. Their log weights are
    log(prior density / proposal density), in the ORIGINAL physical coordinates,
    and may include -inf for zero prior. No Jacobian is added by this module.
    Samples outside declared support contribute zero to the integral. Posterior
    rows outside support (including zero-weight rows) raise instead of dropping.
    Optimizer success and integration ESS are numerical diagnostics, not a
    scientific validation; repeat with independent proposals and held-out PE.
    """
    model = _template(bounds, distance_coordinate, nuisance_prior, mass_support, distance_knots)
    post, _ = _points(posterior_x)
    integ, _ = _points(integration_x)
    if not len(post) or not np.all(support(post, bounds, mass_support)):
        raise ValueError('Nonfinite or out-of-support posterior rows')
    if not len(integ) or not np.isfinite(integ).all():
        raise ValueError('Nonfinite or empty integration proposal')
    lw = np.asarray(integration_log_weights, dtype=float)
    if lw.shape != (len(integ),) or np.isnan(lw).any() or np.isposinf(lw).any():
        raise ValueError('Invalid integration log weights')
    valid = support(integ, bounds, mass_support) & np.isfinite(lw)
    if not valid.any():
        raise ValueError('No positive supported integration weight')
    wpost = _weights(posterior_weights, len(post), 'posterior weights')
    _condition_on_training(model, post, wpost)
    _setup_spline(model, post, wpost)
    post_features = _features(post, model)
    target, whitening, singular, rank = _feature_whitening(post_features, wpost)
    f = _features(integ[valid], model)
    optimization_features = (f-target)@whitening
    base = lw[valid]
    logn = np.log(len(integ))
    evaluations = [0]
    def objective(theta):
        t = base+optimization_features@theta
        lz = logsumexp(t)
        wn = np.exp(t-lz)
        evaluations[0] += 1
        return float(lz-logn), wn@optimization_features
    res = minimize(objective, np.zeros(rank), jac=True, method='BFGS',
                   options={'maxiter': int(maxiter), 'gtol': 1e-8})
    if not np.isfinite(res.x).all():
        raise ValueError('Optimizer returned nonfinite natural parameters')
    coefficients = whitening@res.x
    logw = base+f@coefficients
    logz = float(logsumexp(logw)-logn)
    wn = np.exp(logw-logsumexp(logw))
    model['coefficients'] = coefficients.tolist()
    model['log_offset'] = -logz
    objective_value, gradient = objective(res.x)
    diag = {'method': 'prior-forward-normalized-likelihood-shape', 'optimizer_success': bool(res.success),
            'optimizer_message': str(res.message), 'optimizer_iterations': int(res.nit),
            'numerical_optimizer_gate': bool(res.success and rank == f.shape[1] and np.max(np.abs(gradient)) < 1e-6),
            'objective_evaluations': evaluations[0], 'objective': objective_value,
            'maximum_abs_feature_residual': float(np.max(np.abs(gradient))),
            'maximum_abs_unwhitened_feature_residual': float(np.max(np.abs(wn@f-target))),
            'feature_rank': rank, 'feature_count': f.shape[1],
            'training_feature_rank_deficient': bool(rank < f.shape[1]),
            'training_feature_condition_number': float(singular[0]/singular[rank-1]),
            'optimization_feature_condition_number': 1.,
            'feature_whitening': 'training-only weighted centered SVD; coefficients converted back to stored feature basis',
            'integration_draws': len(integ), 'integration_supported_positive_draws': int(valid.sum()),
            'integration_outside_support_draws': int((~support(integ, bounds, mass_support)).sum()),
            'integration_ess': float(1./(wn@wn)), 'integration_maximum_weight_fraction': float(wn.max()),
            'integration_log_normalizer_before_offset': logz,
            'posterior_rows': len(post), 'posterior_weight_ess': float(1./(wpost@wpost)),
            'integration_measure': 'original physical coordinates; supplied log prior/proposal ratio'}
    diag.update(_family_diagnostics(model))
    return model, diag
