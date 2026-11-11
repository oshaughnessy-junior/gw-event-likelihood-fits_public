# Native-posterior reconstruction with the spin prior as base measure

`src/native_spin_base_redshift.py` evaluates the scalar four-dimensional likelihood
of schema `aligned-redshift-native-spin-base-posterior-v1`. It imports only this
repository's `aligned_redshift` support and mass utilities, the distance factor of
`native_posterior_redshift`, NumPy and SciPy. Collections select it with
`likelihood_backend: native_spin_base_redshift`.

Inputs are `x=[Mc_source, eta, chi_eff, z]`, with the fixed source-time cosmology of
`NATIVE_POSTERIOR_REDSHIFT.md`. The density chart is
`v=(Mc_detector, delta, chi_eff, log z)`, `delta=sqrt(1-4 eta)`.

The fitted native posterior density in `v` is

    f(v) = pi(chi_eff|q) sum_j p_j g_j(delta, chi_eff) N_j(Mc_detector, log z | delta, chi_eff),
    g_j  = exp(theta_j . T - logZ_j).

`T` holds the quadratic features `(u, w, u^2, u w, w^2)` of the standardized
`u=(delta-center_0)/scale_0` and `w=(chi_eff-center_1)/scale_1`. `logZ_j` normalizes
`pi exp(theta_j . T)` over `delta` and `chi_eff`; it is fitted and stored with each
component. `N_j` is a bivariate normal whose mean is linear in `(u, w)`. `pi` is the
native conditional effective-spin prior for independent uniform spin magnitudes below
the two recorded caps and isotropic tilts.

The native retained prior in `v` is `pi(chi_eff|q) J_detector_delta z p_native(dL) dDL/dz`
up to an event constant. `pi` therefore cancels in the scalar likelihood, which is
`sum_j p_j g_j N_j` divided by the remaining mass and distance factors. The
`chi_eff` dependence of `L` is the fitted quadratic exponent, with a finite value at
the spin caps and `-inf` beyond them. The runtime never evaluates `pi`.

`integration_bank(model, n, seed, grid_size, uniform_fraction)` returns
`(y, logw, x, metadata)` with `y=[m1_source, m2_source, chi_eff, z]` and weights for
`dm1_source dm2_source dchi_eff dz`. The proposal draws `(delta, chi_eff/edge)` from a
table of each `g_j` without `pi`, mixed with draws uniform in `chi_eff`, and
`(Mc_detector, log z)` from `N_j`. `proposal_logpdf` returns its density, which
enters the weights. Bank weights hold no population redshift law, and a fixed event
normalization is removed.

Synthetic controls:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_native_spin_base_redshift.py -v
```

They check the pointwise prior quotient against an independent formula, that the bank
times the full native prior (including `pi`) recovers first and second moments of a
known two-component posterior, that the proposal density integrates correctly, spin
endpoints, model rejection, Collection dispatch and the public-data gate. Each
check fails when the corresponding term of the runtime is removed.

`tools/check_public_base_runtime.py` compares a collection with the exact leaf source
that produced its models, pointwise and bank-for-bank at equal seed.
`tools/build_public_base_collection.py` publishes fits only when they pass the
producer's declared checks on the published split and on every replication split, and
re-issues each existing public proof for the new fit bytes.
