# Public aligned-spin and redshift candidates, spin prior as base measure

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

This edition refits two events of `20261031-public-nativecdf-4d-v1` from the same
public-certified references and metadata (identical sha256). It uses schema
`aligned-redshift-native-spin-base-posterior-v1` and the backend
`native_spin_base_redshift` (`NATIVE_SPIN_BASE_REDSHIFT.md`). The native effective-spin
prior is the base measure of the fit, so it cancels in the scalar likelihood and the
`chi_eff` tail of `L` follows the fitted quadratic exponent.

Use `Collection(root, "quasicircular/20261109-public-base8-4d-v1")` with points
ordered `mc_source, eta, chi_eff, z`. Evaluation requires `allow_candidate=True` and
the source-issue acknowledgment. Neither flag bypasses public certificates.

Each public proof is the proof of the earlier edition, re-issued for the new fit
bytes. Release, file, checksum, posterior group and reference hashes are unchanged.
`reissued_from` names the earlier proof. No posterior draws or integration banks are
published.

Every entry is a **candidate**. Each passed the producer's declared checks on the
published intrinsic-group split (seed 73) and on two replication splits (seeds 101
and 202). This is not population-recovery certification. Both events keep their
passing native-CDF candidates in the earlier edition; this edition replaces nothing.

| Event | Components | Largest marginal JS, bits | Largest q given z JS, bits | Splits passed |
|---|---:|---:|---:|---:|
| GW190719_215514 | 16 | 0.012 | 0.026 | 3 of 3 |
| GW190805_211137 | 16 | 0.014 | 0.037 | 3 of 3 |

GW190929_012149, GW230820_212515 and GW230928_215827 are listed as unresolved in the
manifest, each with its reason. GW230820 and GW230928 fail the reference ESS check,
which no fit of these references can pass.

Fits, validation and replication fits are archived outside this repository (`SHA256SUMS` sha256
`c03e4a095bd5b4a2…`). Record: RIFT_roboto_paper
`analyses/fit_refit_20261108/RESULTS_public5_20261109.md`. `runtime_parity.json`
compares this collection with the exact producing leaf, pointwise and bank-for-bank.
