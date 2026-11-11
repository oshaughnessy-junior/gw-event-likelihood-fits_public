# Native-posterior reconstruction with redshift

`src/native_posterior_redshift.py` evaluates a positive joint four-dimensional
likelihood from a native-posterior mixture. It imports only this repository's
`aligned_redshift` kernel and `native_redshift_prior` utilities, plus NumPy,
SciPy and Astropy. This source-only addition contains no observational
coefficients or public-source certification.

Inputs are `x=[Mc_source, eta, chi_eff, z]`: chirp masses are in solar masses,
redshift is dimensionless, and luminosity distance is in Mpc. The cosmology
mapping these source coordinates to detector masses/distance is fixed explicitly
at FlatLambdaCDM(H0=67.9 km/s/Mpc, Om0=.3065, Tcmb0=0 K). The event's native
distance-prior cosmology is separately retained and may differ.

The density chart is `v=(Mc_detector,delta,s,log(z))`, where
`delta=sqrt(1-4*eta)` and `s=2*a*F_native(chi_eff|q)-a`.
`F_native` assumes independent uniform spin magnitudes with the recorded two
caps and isotropic tilts. The effective-spin prior is conditional on q.
A different native spin family requires a different CDF; it cannot be replaced
by a pooled spin distribution. The numerical CDF uses split Gauss quadrature,
with an analytic leading logarithmic term. Its inverse uses bisection.
Finite floating-point resolution near the physical spin caps is not a generic
accuracy guarantee for arbitrarily concentrated edge distributions.

Fit the original paired native posterior weights in this chart. Do not multiply
row weights by a coordinate Jacobian or inverse prior when fitting this native
posterior density. The model uses schema
`aligned-redshift-native-spin-cdf-posterior-v1`; each component carries its bounded
quadratic normalization and conditional detector-Mc/log-z Gaussian parameters.
Full coupled support and the conditional nuisance-prior inventory must accompany
observational products. A retained-prior quotient cannot change hidden-spin or
inclination priors that were marginalized out.

For `J_detector_delta=Mc_detector*eta^(-6/5)/2`, the native prior in this chart is
`J_detector_delta*z*p_native(dL)*dDL/dz/(2*a)`, up to an event constant.
`log_likelihood_x(x, model)` returns the scalar likelihood after dividing the
fitted chart density by this prior. There is no residual `p(chi_eff|q)` divisor,
so scalar evaluation has a finite spin-boundary limit. The runtime preserves
native mass, distance/redshift and spin support. It does not infer likelihoods
outside the original prior support or recover absolute event evidence.

`native_redshift_prior.log_distance_density` supports distance-squared,
UniformSourceFrame and UniformComovingVolume priors, together with the explicitly
recorded supported cosmologies. Cosmological priors are densities in **dL**;
their native dz/ddL factor is included. `common_source_time` explicitly denotes
the fixed source-time cosmology above. Unsupported declarations raise an error.
These densities are defined up to an event constant; analytic bounds belong to
the model's declared support.

`integration_bank(model, n, seed, grid_size, uniform_fraction)` returns
`(y, logw, x, metadata)`, with `y=[m1_source,m2_source,chi_eff,z]` and weights
for integration in `dm1_source dm2_source dchi_eff dz`. The known proposal mixes
CDF-coordinate draws with uniform physical effective-spin draws. Its complete
mixture density enters the weights, including the CDF Jacobian. Bank weights
contain no population redshift measure: add the chosen cosmological population
law exactly once. A fixed event normalization is removed from each bank.

Run the synthetic controls with:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests \
  -p test_native_posterior_redshift.py -v
```

They check a known correlated normalized posterior, pointwise prior division,
full-proposal native-posterior restoration, a missing-distance negative control,
conditional-CDF Jacobians and finite spin/equal-mass endpoints. Real fits require
separate held-out native and joint population-prior recovery, proposal precision,
and public-source provenance checks. Small native one-dimensional JS values do
not certify population recovery.
