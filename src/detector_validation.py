"""Held-out recovery checks for experimental distance-resolved likelihoods.

The normalizer banks are independent of the fitting bank. Kish ESS is based on
the original weights, never the number of visualization resamples. Thresholds
are reproducible regression gates, not bounds on population-inference bias.
"""
import numpy as np
from scipy.special import logsumexp, rel_entr
from scipy.stats import gaussian_kde
from detector_distance import log_shape


def normalized(w):
    w = np.asarray(w, float)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0) or not w.sum() > 0:
        raise ValueError('Require finite nonnegative weights with positive sum')
    return w / w.sum()


def ks(a, wa, b, wb):
    ia, ib = np.argsort(a), np.argsort(b)
    grid = np.unique(np.r_[a, b])
    ca, cb = np.r_[0., np.cumsum(wa[ia])], np.r_[0., np.cumsum(wb[ib])]
    return float(np.max(abs(ca[np.searchsorted(a[ia], grid, side='right')] -
                            cb[np.searchsorted(b[ib], grid, side='right')])))


def js(a, wa, b, wb, rng, count=4000):
    aa = a[rng.choice(len(a), count, p=wa)]
    bb = b[rng.choice(len(b), count, p=wb)]
    lo, hi = np.quantile(np.r_[aa, bb], [.0005, .9995])
    if hi <= lo:
        raise ValueError('Degenerate marginal for JS evaluation')
    grid = np.linspace(lo, hi, 512)
    pa, pb = gaussian_kde(aa)(grid), gaussian_kde(bb)(grid)
    pa, pb = pa / pa.sum(), pb / pb.sum()
    mid = (pa + pb) / 2
    return float(.5 * np.sum(rel_entr(pa, mid) + rel_entr(pb, mid)) / np.log(2))


def modifier(x, kind, bounds):
    if kind == 'native':
        return np.zeros(len(x))
    if kind == 'spin_gaussian':
        return -.5 * (x[:, 2] / .2)**2
    if kind == 'distance_tilt':
        # Broad Student-t proposals can have negative distances; these carry
        # zero model support, rather than NaNs in an otherwise valid integral.
        return np.log(np.maximum(x[:, 3],np.finfo(float).tiny) / np.mean(np.asarray(bounds)[3]))
    raise ValueError('Unknown retained-prior modifier')


def integration_weights(x, log_ratio, model, kind):
    x=np.asarray(x,float);log_ratio=np.asarray(log_ratio,float)
    if x.ndim!=2 or x.shape[1]!=4 or not len(x) or not np.isfinite(x).all() or log_ratio.shape!=(len(x),) or np.isnan(log_ratio).any() or np.isposinf(log_ratio).any():
        raise ValueError('Require finite Nx4 integration rows and matching log weights')
    lw = log_shape(x, model) + log_ratio + modifier(x, kind, model['bounds'])
    if not np.isfinite(logsumexp(lw)):
        raise ValueError('Independent integration bank misses all model support')
    z = logsumexp(lw)
    w = np.exp(lw-z)
    ess = float(1 / (w @ w))
    return w, dict(log_normalizer=float(z-np.log(len(x))), integration_ess=ess,
                   relative_mc_se=float(np.sqrt(max(0., 1 / ess - 1 / len(x)))))


def _recovery(ref, wx, bank, weights, rng, *, js_bar, axis_ks_bar,
              projected_ks_bar, z_grid=None, d_l_grid=None):
    """Identical shape checks for either independently generated audit bank."""
    axes = [ks(ref[:,j],wx,bank[:,j],weights) for j in range(4)]
    jss = [js(ref[:,j],wx,bank[:,j],weights,rng) for j in range(4)]
    center = wx @ ref
    scale = np.sqrt(wx @ (ref-center)**2)
    if np.any(scale <= 0):
        raise ValueError('Degenerate held-out coordinate')
    directions = np.random.default_rng(812).normal(size=(24,4))
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    projected = [ks(((ref-center)/scale)@v,wx,((bank-center)/scale)@v,weights) for v in directions]
    conditional = []
    for axis in [0,2]:
        edges = np.quantile(ref[:,axis],[0,1/3,2/3,1])
        for j in range(3):
            r = (ref[:,axis]>=edges[j]) & (ref[:,axis] <= edges[j+1] if j==2 else ref[:,axis]<edges[j+1])
            s = (bank[:,axis]>=edges[j]) & (bank[:,axis] <= edges[j+1] if j==2 else bank[:,axis]<edges[j+1])
            if wx[r].sum()<=0 or weights[s].sum()<=0:
                conditional.append(dict(axis=axis,bin=j,status='insufficient_support'))
                continue
            rw, sw = normalized(wx[r]), normalized(weights[s])
            ess = min(float(1/(rw@rw)),float(1/(sw@sw)))
            value = ks(ref[r,3],rw,bank[s,3],sw)
            conditional.append(dict(axis=axis,bin=j,minimum_ess=ess,distance_ks=value,
                                    status='pass' if ess>=100 and value<=.10 else 'fail'))
    reasons=[]
    if max(jss)>js_bar: reasons.append('marginal_JS')
    if max(axes)>axis_ks_bar: reasons.append('axis_KS')
    if max(projected)>projected_ks_bar: reasons.append('projected_joint_KS')
    if any(c['status']!='pass' for c in conditional): reasons.append('conditional_distance_recovery')
    result=dict(posterior_js_bits=dict(zip(['mc_det','eta','chi_eff','d_l'],jss)),
                axis_ks=axes,projected_joint_ks=projected,conditional_distance=conditional)
    if z_grid is not None:
        smc=ref[:,0]/(1+np.interp(ref[:,3],d_l_grid,z_grid))
        tmc=bank[:,0]/(1+np.interp(bank[:,3],d_l_grid,z_grid))
        sj=js(smc,wx,tmc,weights,rng); sk=ks(smc,wx,tmc,weights)
        result['source_frame_mc_posterior']={'js_bits':sj,'ks':sk}
        if sj>js_bar or sk>axis_ks_bar: reasons.append('source_frame_recovery')
    return dict(result,status='pass' if not reasons else 'fail',reasons=reasons)


def compare(model, reference_x, reference_weights, audit_x, audit_log_weights,
            second_x, second_log_weights, *, z_grid=None, d_l_grid=None,
            seed=319, min_ess=1000, js_bar=.03, axis_ks_bar=.05,
            projected_ks_bar=.07, relative_mc_se_bar=.02, normalizer_difference_bar=.05):
    ref = np.asarray(reference_x, float)
    audit_x, second_x = np.asarray(audit_x,float), np.asarray(second_x,float)
    if ref.ndim != 2 or ref.shape[1] != 4 or not len(ref) or not np.isfinite(ref).all():
        raise ValueError('Reference must contain paired finite four-dimensional rows')
    if not np.isfinite(log_shape(ref, model)).all():
        raise ValueError('Held-out reference outside declared model support')
    wr = normalized(reference_weights)
    if len(wr) != len(ref):
        raise ValueError('Reference weight length mismatch')
    for bank in [audit_x,second_x]:
        if bank.ndim!=2 or bank.shape[1]!=4 or not len(bank) or not np.isfinite(bank).all():
            raise ValueError('Both audit banks must contain finite Nx4 rows')
    if (z_grid is None) != (d_l_grid is None):
        raise ValueError('Supply both z and distance grids')
    if z_grid is not None:
        z_grid, d_l_grid = np.asarray(z_grid), np.asarray(d_l_grid)
        if z_grid.ndim!=1 or d_l_grid.ndim!=1 or len(z_grid)<2 or len(z_grid)!=len(d_l_grid) or not np.isfinite(np.r_[z_grid,d_l_grid]).all() or np.any(np.diff(z_grid)<=0) or np.any(np.diff(d_l_grid)<=0) or z_grid[0]<0 or d_l_grid[0]<=0:
            raise ValueError('Cosmology grid must be finite, positive and strictly increasing')
        for bank in [audit_x,second_x]:
            supported=np.isfinite(log_shape(bank,model))
            if not supported.any():
                raise ValueError('No supported audit coordinates')
            if min(ref[:,3].min(),bank[supported,3].min()) < d_l_grid[0] or max(ref[:,3].max(),bank[supported,3].max()) > d_l_grid[-1]:
                raise ValueError('Cosmology interpolation must cover reference and BOTH audit banks')
    # Keep the original first-bank random stream and fields stable for plotting.
    rng, second_rng = np.random.default_rng(seed), np.random.default_rng(seed+1)
    rows = []
    for kind in ['native', 'spin_gaussian', 'distance_tilt']:
        wx = normalized(wr * np.exp(modifier(ref, kind, model['bounds'])))
        wy, d1 = integration_weights(audit_x, audit_log_weights, model, kind)
        wz, d2 = integration_weights(second_x, second_log_weights, model, kind)
        discrepancy = float(abs(np.expm1(np.clip(d2['log_normalizer']-d1['log_normalizer'], -700, 700))))
        arguments=dict(js_bar=js_bar,axis_ks_bar=axis_ks_bar,projected_ks_bar=projected_ks_bar,z_grid=z_grid,d_l_grid=d_l_grid)
        first=_recovery(ref,wx,audit_x,wy,rng,**arguments)
        second=_recovery(ref,wx,second_x,wz,second_rng,**arguments)
        reasons=[]
        source_ess=float(1/(wx@wx))
        if source_ess<min_ess: reasons.append('original_reference_weight_ESS')
        if min(d1['integration_ess'],d2['integration_ess'])<min_ess: reasons.append('integration_ESS')
        if max(d1['relative_mc_se'],d2['relative_mc_se'])>relative_mc_se_bar: reasons.append('normalizer_MCSE')
        if discrepancy>normalizer_difference_bar: reasons.append('independent_normalizer_disagreement')
        reasons.extend(first['reasons'])
        if second['status']!='pass': reasons.append('second_audit_bank_recovery')
        # Existing first-bank keys remain at their original paths. The second
        # bank receives the SAME shape gates, not merely a normalizer check.
        row=dict(first,prior=kind,status='pass' if not reasons else 'fail',reasons=reasons,
                 first_audit_bank_recovery_status=first['status'],
                 second_audit_bank_recovery=dict(second,integration=d2),
                 original_reference_weight_ess=source_ess,integrations=[d1,d2],
                 independent_normalizer_relative_difference=discrepancy)
        rows.append(row)
    return dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',views=rows,
                gates=dict(min_ess=min_ess,js_bits=js_bar,axis_ks=axis_ks_bar,
                           projected_ks=projected_ks_bar,relative_mc_se=relative_mc_se_bar,
                           normalizer_difference=normalizer_difference_bar,conditional_distance_ks=.10,
                           shape_recovery_banks=2),
                scope='Held-out paired PE recovery in BOTH independent audit banks with retained-coordinate prior modifiers; not an independent likelihood or hidden-nuisance-prior test.',
                sampling_limit='Kish weight ESS does not establish independent PE chain information.')
