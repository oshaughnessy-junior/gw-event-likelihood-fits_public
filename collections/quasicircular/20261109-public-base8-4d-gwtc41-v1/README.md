# Public base-measure candidates for seven GWTC-4 events from GWTC-4.1

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

This edition publishes base-measure fits for GW230726, GW230729, GW230731, GW231020,
GW231110, GW231113 and GW231123. The schema is `aligned-redshift-native-spin-base-posterior-v1`
and the backend is `native_spin_base_redshift` (`NATIVE_SPIN_BASE_REDSHIFT.md`). None of these
events is in an earlier 4D edition.

"base8" in the edition name is the base-measure family, not the component count. These fits
use k = 16, as `20261109-public-base8-4d-v1` does; `20261109-public-base8-4d-jointv2-v1` uses k = 8.

Use `Collection(root, "quasicircular/20261109-public-base8-4d-gwtc41-v1")` with points
ordered `mc_source, eta, chi_eff, z`. Evaluation requires `allow_candidate=True` and the
source-issue acknowledgment. Neither flag bypasses public certificates.

## Public source

The paper-2 joint_v2 references for these events came from bilby runs that no public release
holds: the GWTC-4.0 release reweighted a different input. Each reference here was exported
instead from the GWTC-4.1 release file (Zenodo 20275769, group `C00:IMRPhenomXPHM-SpinTaylor`).
Every proof uses the exact-file route.

The exporter read each file from an internal mirror of the release. It used a mirror file only
when its md5 equalled the Zenodo checksum. `source_metadata/` records the file name, and the proof
binds the file's sha256 to the release file, as in `20261109-public-base8-4d-jointv2-v1`.

The proofs were drafted at commit `fea1cf91`, tag `drafts/gwtc41-seven-public-proofs-20261109`
(`provenance/drafts/20261109-gwtc41-seven-public-sources/`), for the exact model bytes.
`tools/build_public_base_collection_drafts.py` makes only the declared promotion changes: `verified`
status, `fit_sha256` rebound to the published artifact, the Zenodo snapshot moved into this collection,
and `leaf_revision`, `source_model_sha256` and `promoted_from` added.

## Validation

No population analysis uses these fits; paper 2 used the joint_v2 fits for these events. The edition
was built with `--select-by-checks`: a fit is released only if it passes the declared checks on its
published split and on every declared replication seed, and each entry states this.

Every entry is a **candidate**. The configuration (k = 16, EM tolerance 2e-5, at most 2000
iterations) was adopted from `20261109-public-base8-4d-v1`, where it was the fewest components
passing on three splits. Here each k16 fit passed the producer's declared checks on
intrinsic-group split seed 73 and on replication seeds 101 and 202. A k = 8 fit ran on seed 73
only, and passed for all seven.

Published fits are seed 73. "Worst" is the worst value over seeds 73, 101 and 202.

| Event | EM iterations | Max marginal JS, bits | Max q given z JS, bits | Min reference ESS | Worst marginal JS | Worst q given z JS | Worst reference ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| GW230726_002940 | 169 | 0.010 | 0.032 | 1217 | 0.013 | 0.032 | 1208 |
| GW230729_082317 | 71 | 0.027 | 0.040 | 777 | 0.027 | 0.040 | 777 |
| GW230731_215307 | 124 | 0.012 | 0.017 | 1338 | 0.012 | 0.018 | 1331 |
| GW231020_142947 | 223 | 0.018 | 0.031 | 927 | 0.018 | 0.031 | 927 |
| GW231110_040320 | 162 | 0.021 | 0.049 | 765 | 0.021 | 0.049 | 765 |
| GW231113_200417 | 143 | 0.019 | 0.034 | 832 | 0.024 | 0.034 | 832 |
| GW231123_135430 | 104 | 0.008 | 0.028 | 2419 | 0.008 | 0.028 | 2419 |

Limits: marginal JS 0.03, q given z JS 0.06, reference ESS 80.

We did not run the tail gate or the bias share of `eval_refit.py` on these fits. This is not
population-recovery certification.

## Provenance

- Export: RIFT_roboto_paper `export_public_replacements.py` at commit `f8c7e098`.
- Fits and replications: archived outside this repository
  (`SHA256SUMS` sha256 `7480e37a3cd624f3…`), HTCondor cluster 571783503.
- Records: RIFT_roboto_paper `analyses/paper2_aligned_redshift_20261030/RESULTS_public_sources_20261109.md`
  and `RESULTS_gwtc41_seven_refit_20261109.md`.
- `runtime_parity.json` compares this collection with the producing leaf (`posterior_base.py`
  sha256 `bea8929d…`), pointwise and bank-for-bank.
