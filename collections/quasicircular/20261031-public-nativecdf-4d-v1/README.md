# Public aligned-spin and redshift candidates

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

These five replacements are fitted only to checksum-verified public posterior products.
They retain four physical coordinates: source chirp mass, eta, effective aligned
spin and redshift. The native-CDF representation reconstructs a scalar likelihood
by explicitly removing the source analysis's native mass, spin and distance prior.
Original conditional nuisance priors and native support remain part of its scope.

Public source eligibility is verified independently of numerical acceptance.
Every entry is a **candidate**, including entries that pass all declared stress
checks. Native and predeclared forward-prior diagnostics are preserved; no universal
retained-prior or population-recovery certification is granted. The collection is
an explicitly enumerated replacement subset, not a complete catalog.

O3 inputs use the public GWTC2.1 NoCosmo XPHM groups with native distance-squared
priors. The cosmologically reweighted sibling products are not substituted while
retaining their serialized distance-squared prior. O4 inputs use the public GWTC4.1
XPHM-SpinTaylor groups with their own native priors. GWTC4.1's version-specific
Zenodo record 20275769 has no human version field, so provenance pins
`record:20275769`, its DOI and exact file checksum; it does not infer a version
from a filename or metadata revision number.

Use `Collection(root, "quasicircular/20261031-public-nativecdf-4d-v1")`.
The explicit `native_posterior_redshift` backend expects points ordered as
`mc_source, eta, chi_eff, z`. Evaluation requires `allow_candidate=True` and the
existing source-issue acknowledgment. Neither flag bypasses public certificates.
The old 3D catalog is not silently used when a model or source is unavailable.

The manifest, source ledger, public API snapshots and fit-bound proof JSON identify
exact inputs. Original public-data validation reports are preserved alongside
candidate wrappers. Their diagnostic mass ratio q is not relabeled as eta.
`population_5` is a predeclared generic test prior, not a reused private
hyperposterior sample set. No posterior draws or integration banks are published.

The largest native one-dimensional JS values below are in bits and describe
`Mc_source, q, chi_eff, z`. They are recovery diagnostics, not full-joint or
population certification. Predeclared test-prior results retain their original
pass/fail status in the per-event reports.

| Event | Public posterior rows | Largest native 1D JS, bits | Largest predeclared target 5 JS, bits | All declared checks pass |
|---|---:|---:|---:|---|
| GW190929_012149 | 153,782 | 0.00302 | 0.01184 | no |
| GW190805_211137 | 169,040 | 0.00162 | 0.00662 | yes |
| GW230928_215827 | 19,897 | 0.00807 | 0.08964 | no |
| GW230820_212515 | 15,811 | 0.00768 | 0.07305 | no |
| GW190719_215514 | 153,538 | 0.00347 | 0.01023 | yes |
