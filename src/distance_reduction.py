"""Reduce a detector-frame likelihood using an explicit, normalized redshift measure.

Likelihoods are scalar functions: substitution of Mc_det=(1+z)*Mc_source
does not multiply their value by a coordinate-density Jacobian.
"""
import numpy as np


def validate_quadrature(z, d_l, log_weights):
    z, d_l, lw = (np.asarray(a, dtype=float) for a in (z, d_l, log_weights))
    if any(a.ndim != 1 for a in (z, d_l, lw)) or not (len(z) == len(d_l) == len(lw)) or not len(z):
        raise ValueError("quadrature arrays must be nonempty matching one-dimensional arrays")
    if not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(np.diff(z) <= 0):
        raise ValueError("z must be finite, nonnegative and strictly increasing")
    if not np.all(np.isfinite(d_l)) or np.any(d_l <= 0) or np.any(np.diff(d_l) <= 0):
        raise ValueError("d_l must be finite, positive and strictly increasing")
    if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ValueError("log_weights must be finite or negative infinity")
    log_mass = np.logaddexp.reduce(lw)
    if not np.isfinite(log_mass) or abs(log_mass) > 1e-8:
        raise ValueError("log_weights must describe an already normalized proper redshift measure")
    return z, d_l, lw


def validate_measure(measure):
    if not isinstance(measure, dict):
        raise ValueError("measure metadata must be a JSON object")
    for key in ("cosmology", "nuisance_measure"):
        if not measure.get(key) or not isinstance(measure[key], (str, dict)):
            raise ValueError("explicit cosmology and nuisance_measure metadata are required")
    return measure


def _source_points(source_x):
    x = np.asarray(source_x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError("source_x must be a finite N by 3 array (mc_source, eta, chi_eff)")
    return x


def joint_source(source_x, model, z, d_l):
    """Evaluate the joint likelihood at paired source coordinates and fixed z,dL.

    z,d_l may be scalars or arrays broadcastable to the number of source points.
    Values outside the detector model's original support remain -infinity.
    """
    from detector_distance import log_shape
    x = _source_points(source_x)
    z, d = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(d_l, dtype=float))
    if np.any(~np.isfinite(z)) or np.any(z < 0) or np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("z must be nonnegative and distance positive, both finite")
    out = np.empty((len(x), 4))
    out[:, :3] = x
    out[:, 0] *= np.broadcast_to(1 + z, (len(x),))
    out[:, 3] = np.broadcast_to(d, (len(x),))
    bounds = np.asarray(model["bounds"], dtype=float)
    if bounds.shape != (4, 2) or np.any(~np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("model bounds must be finite increasing 4 by 2 original support")
    inside = np.all((out >= bounds[:, 0]) & (out <= bounds[:, 1]), axis=1)
    vals = np.asarray(log_shape(out, model), dtype=float)
    if vals.shape != (len(x),) or np.any(np.isnan(vals)) or np.any(np.isposinf(vals)):
        raise ValueError("detector likelihood returned invalid values")
    return np.where(inside, vals, -np.inf)


def marginalize_source(source_x, model, z, d_l, log_weights, chunk_size=256):
    """Integrate a scalar likelihood; never renormalize support-truncated weights.

    log_weights already contain rho_z dz, normalized over the declared proper
    redshift prior. Loss of that measure outside model support is reported.
    """
    x = _source_points(source_x)
    z, d, lw = validate_quadrature(z, d_l, log_weights)
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    result, supported = np.full(len(x), -np.inf), np.zeros(len(x))
    for start in range(0, len(x), chunk_size):
        block = x[start:start + chunk_size]
        repeated = np.repeat(block, len(z), axis=0)
        val = joint_source(repeated, model, np.tile(z, len(block)), np.tile(d, len(block)))
        val = val.reshape(len(block), len(z))
        result[start:start + len(block)] = np.logaddexp.reduce(val + lw[None, :], axis=1)
        supported[start:start + len(block)] = np.exp(np.logaddexp.reduce(
            np.where(np.isfinite(val), lw[None, :], -np.inf), axis=1))
    diagnostics = {"supported_prior_mass": supported,
                   "lost_prior_mass": np.maximum(0., 1. - supported),
                   "quadrature_nodes": len(z),
                   "support_policy": "original detector support; missing mass is not renormalized",
                   "likelihood_jacobian": "none: scalar likelihood evaluated by substitution"}
    return result, diagnostics


def compare_reductions(log_likelihood, check_log_likelihood, max_relative_difference=.02):
    """Convergence against an independent finer quadrature on the same source points.

    Relative differences use the finer result as denominator. Two unsupported
    points are unresolved: two grids can miss the same thin nonempty support.
    """
    a, b = (np.asarray(v, dtype=float) for v in (log_likelihood, check_log_likelihood))
    if a.shape != b.shape or a.ndim != 1 or np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("comparison requires matching one-dimensional valid log likelihoods")
    if np.any(np.isposinf(a)) or np.any(np.isposinf(b)):
        raise ValueError("positive infinite log likelihood is invalid")
    if not np.isfinite(max_relative_difference) or not 0 < max_relative_difference < 1:
        raise ValueError("relative convergence tolerance must be between zero and one")
    diff = np.zeros(len(a))
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(over="ignore"):
        diff[finite] = np.abs(np.expm1(a[finite] - b[finite]))
    disagreement = np.isfinite(a) != np.isfinite(b)
    # A support disagreement is always a failure, even when the finer likelihood
    # is finite and the coarse estimate is zero.
    diff[disagreement] = np.inf
    both_unsupported = np.isneginf(a) & np.isneginf(b)
    passed = (diff <= max_relative_difference) & ~disagreement & ~both_unsupported
    return {"relative_likelihood_difference": diff, "quadrature_converged": passed,
            "nonfinite_disagreement": disagreement,
            "status": "pass" if np.all(passed) else "fail",
            "max_relative_difference": max_relative_difference,
            "relative_difference_definition": "abs(L_coarse/L_fine - 1); support disagreement fails",
            "both_unsupported_count": int(np.sum(np.isneginf(a) & np.isneginf(b)))}


def marginalize_source_adaptive(source_x, model, measure, nodes=256):
    """Integrate on each source point's continuous detector-support intersection.

    The local quadrature retains the GLOBAL proper redshift-prior normalization.
    Currently supports only builder-declared flat LambdaCDM and known measures.
    Empty intersections are proven geometrically, not inferred from missed nodes.
    """
    from astropy.cosmology import FlatLambdaCDM
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from detector_distance import _mass_bounds
    validate_measure(measure)
    c, prior = measure["cosmology"], measure["nuisance_measure"]
    if not isinstance(c, dict) or not isinstance(prior, dict) or c.get("class") != "astropy.cosmology.FlatLambdaCDM" or c.get("Tcmb0_K") != 0:
        raise ValueError("adaptive integration requires builder FlatLambdaCDM with Tcmb0=0")
    kind = prior.get("kind")
    if kind not in ("flat_z", "source_volume", "comoving_volume"):
        raise ValueError("adaptive integration requires a known proper redshift measure")
    zmin, zmax = map(float, prior["bounds"])
    if not np.isfinite([zmin,zmax]).all() or zmin < 0 or zmax <= zmin:
        raise ValueError("invalid declared redshift support")
    if not isinstance(nodes, (int, np.integer)) or nodes < 2:
        raise ValueError("at least two adaptive quadrature nodes are required")
    cosmo = FlatLambdaCDM(H0=c["H0_km_s_Mpc"], Om0=c["Om0"], Tcmb0=0)
    if c.get("distance_unit") != "Mpc" or not 0 < c["Om0"] <= 1 or c["H0_km_s_Mpc"] <= 0:
        raise ValueError("invalid adaptive cosmology or distance units")
    def density(z):
        if kind == "flat_z":
            return np.ones_like(z, dtype=float)
        value = 4*np.pi*cosmo.differential_comoving_volume(z).value
        return value/(1+np.asarray(z)) if kind == "source_volume" else value
    global_norm = quad(lambda z: float(density(z)), zmin, zmax, epsabs=1e-10, epsrel=1e-10)[0]
    if not np.isfinite(global_norm) or global_norm <= 0:
        raise ValueError("invalid global redshift-prior normalization")
    x = _source_points(source_x)
    bounds = np.asarray(model["bounds"], dtype=float)
    if bounds.shape != (4,2) or not np.isfinite(bounds).all() or np.any(bounds[:,1] <= bounds[:,0]):
        raise ValueError("invalid original detector bounds")
    distance = lambda z: float(cosmo.luminosity_distance(z).value)
    dlo, dhi = distance(zmin), distance(zmax)
    distance_empty = bounds[3,0] >= dhi or bounds[3,1] <= dlo
    low_dz, high_dz = zmin, zmax
    if not distance_empty:
        if bounds[3,0] > dlo:
            low_dz = brentq(lambda z: distance(z)-bounds[3,0], zmin, zmax, xtol=1e-13)
        if bounds[3,1] < dhi:
            high_dz = brentq(lambda z: distance(z)-bounds[3,1], zmin, zmax, xtol=1e-13)
    mass_bounds = _mass_bounds(model.get("mass_support"))
    t, w = np.polynomial.legendre.leggauss(nodes)
    result = np.full(len(x), -np.inf)
    supported = np.zeros(len(x)); empty = np.ones(len(x), dtype=bool)
    intervals = np.full((len(x),2), np.nan)
    for i, (mc, eta, spin) in enumerate(x):
        if distance_empty or mc <= 0 or not 0 < eta <= .25 or not bounds[1,0] <= eta <= bounds[1,1] or not bounds[2,0] <= spin <= bounds[2,1]:
            continue
        lo = max(low_dz, bounds[0,0]/mc-1)
        hi = min(high_dz, bounds[0,1]/mc-1)
        total = mc*eta**(-3/5)
        delta = np.sqrt(max(0.,1-4*eta))
        source_masses = {"mtotal": total, "m1":total*(1+delta)/2,
                         "m2":total*2*eta/(1+delta)}
        for key, (lower, upper) in mass_bounds.items():
            lo = max(lo, lower/source_masses[key]-1)
            hi = min(hi, upper/source_masses[key]-1)
        if hi <= lo:
            continue
        empty[i] = False; intervals[i] = [lo,hi]
        z = lo + (t+1)*(hi-lo)/2
        d = cosmo.luminosity_distance(z).value
        lw = np.log(w*(hi-lo)/2) + np.log(density(z)) - np.log(global_norm)
        vals = joint_source(np.repeat(x[i:i+1], nodes, axis=0), model, z, d)
        result[i] = np.logaddexp.reduce(vals+lw)
        supported[i] = quad(lambda z: float(density(z)), lo, hi, epsabs=1e-10, epsrel=1e-10)[0]/global_norm
    return result, {"supported_prior_mass":supported, "lost_prior_mass":np.maximum(0.,1-supported),
        "geometrically_empty_support":empty, "redshift_support_intervals":intervals,
        "quadrature_nodes":nodes, "method":"source-specific continuous-support Gauss-Legendre",
        "global_prior_normalization":global_norm, "support_policy":"original detector support; no local renormalization",
        "likelihood_jacobian":"none: scalar likelihood evaluated by substitution"}
