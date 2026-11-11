# Paper-2 base8 fits with public-source proofs

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

This edition releases the base8 fits (k = 8, at most 100 EM iterations) that the paper-2
population analysis uses, for the 38 joint_v2 events whose fit inputs equal Zenodo release rows.
Each fit has the coefficients of the model the population manifest names
(`population_base8_refit_v1_20261109`, sha256 `0b28b11fcc8932c2…`); the private working repository
holds those exact bytes, whose sha256 `EXPORT_MANIFEST.json` gives. The schema is
`aligned-redshift-native-spin-base-posterior-v1` and the backend is `native_spin_base_redshift`
(`NATIVE_SPIN_BASE_REDSHIFT.md`). None overlaps the two earlier 4D editions.

Use `Collection(root, "quasicircular/20261109-public-base8-4d-jointv2-v1")` with points
ordered `mc_source, eta, chi_eff, z`. Evaluation requires `allow_candidate=True` and the
source-issue acknowledgment. Neither flag bypasses public certificates.

## Public source

Each proof was drafted at commit `6b9e3f4d`, tag `drafts/jointv2-public-proofs-20261109`
(`provenance/drafts/20261109-joint-v2-public-sources/`), for the exact model bytes. A sample-level
audit found the reference rows equal to those of a Zenodo release file. `tools/build_public_base_collection_drafts.py`
makes these changes to each draft, and no others:

- sets `status` to `verified` and removes `draft_note`;
- rebinds `fit_sha256` from the model to the published artifact;
- moves the Zenodo metadata snapshot into this collection and updates `public_metadata`;
- adds `leaf_revision`, `source_model_sha256` and `promoted_from` (the draft's path, sha256, commit and tag).

The drafts are kept, unchanged, in that directory. The runtime refuses a proof that differs from
its draft in any other field, or whose reference, model, producer or leaf differs from the fit.
`tools/check_public_data.py` also compares each draft with the tagged commit when git can reach it.

| Release | Zenodo record | Route | Events |
|---|---|---|---:|
| GWTC-2.1 v2 | 6513631 | exact file | 13 |
| GWTC-3 | 22685054 | exact file | 8 |
| GWTC-3 v1 | 5546663 | posterior and prior equivalence | 2 |
| GWTC-4.0 | 16053484, 17014085 | exact file | 15 |

`source_metadata/` holds each fit's input metadata. Its `source.path`, the file the fit read, is
reduced to `<host-path>/` and the file name. For GWTC-4.0 events that file was in an internal LVK
mirror, and for older events in an LVK PE directory. Each proof shows that the file read has the same sha256 as the Zenodo release
file, so the bytes fitted are the public bytes.

## Validation

Every entry is a **candidate**. The producer's declared checks are recorded per event, in
`validation/<analysis>.json`, in each entry's scope and in `acceptance.declared_checks`; they did
not select events. The loader refuses an entry whose stated outcome differs from its records.
Limits: marginal JS 0.03 bits, q given z JS 0.06 bits, reference ESS 80.

| Outcome | Events |
|---|---:|
| Passes on seed 73 and on replication seeds 101 and 202 | 25 |
| Passes on seed 73, fails q given z JS on a replication seed (GW170729, GW230606, GW230609) | 3 |
| Fails on seed 73 (replication not run) | 10 |

The ten that fail on seed 73 are GW191127, GW230704_212616, GW230819, GW231001, GW231004,
GW231119, GW231127, GW231221, GW231223 and GW231230. Three of them (GW230819, GW231001 and
GW231004) fail the reference ESS check, which depends on the held-out PE rows and not on the fit.
For 23 events the split is by row, since the references carry no intrinsic-group identity; for 15 it is by intrinsic group.
EM stopped at the iteration limit for 26 events. This is not population-recovery certification.

Refits of GW170729, GW230606 and GW230609 at k = 16, tol 2e-5, also fail at least one of the
three splits. They are not released;
the paper uses the base8 fits.

Not included: 30 events whose rows match only the GWTC-2 DCC release (the verifier accepts
Zenodo only), and 60 events equal to a release only after the GWTC-4.0 cosmology reweighting.

## Provenance

- Fits: archived outside this repository (`SHA256SUMS` sha256 `60ca907ae185e586…`).
- Replications: archived outside this repository (`SHA256SUMS` sha256 `1619fc3de9711d27…`), HTCondor cluster 571783502.
- Record: RIFT_roboto_paper `analyses/fit_refit_20261108/RESULTS_public28_20261109.md`.
- `runtime_parity.json` compares this collection with the producing leaf (`posterior_base.py` sha256 `bea8929d…`), pointwise and bank-for-bank.
