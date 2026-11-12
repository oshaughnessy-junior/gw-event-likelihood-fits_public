# Paper-2 base8 fits with DCC, Zenodo and reissued proofs

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

This edition releases 48 base8 fits (k = 8, at most 100 EM iterations) that the paper-2
population analysis uses. Each fit has the coefficients of the model the population manifest names
(`population_base8_refit_v1_20261109`, sha256 `0b28b11fcc8932c2…`); the private working repository
holds those exact bytes, whose sha256 `EXPORT_MANIFEST.json` gives. Selection follows RO's
decision of 2026-11-10: release where a public-source proof covers the training inputs, and record
validation outcomes without selecting on them. Candidates were the 115 paper-2 fits not yet
released byte-exact.

Use `Collection(root, "quasicircular/20261110-public-base8-4d-paper2-v1")` with points ordered
`mc_source, eta, chi_eff, z`. Evaluation requires `allow_candidate=True` and the source-issue
acknowledgment. Neither flag bypasses public certificates.

## Public source

| Route | Proof method | Release | Events |
|---|---|---|---:|
| GWTC-2 DCC archive member | `public_dcc_archive_member_match` | P2000223-v7 | 30 |
| Zenodo file (N. Manning SEOBNRv5PHM) | `public_release_file_checksum_match` | 24094073, version 1.0.0 | 13 |
| Certified public export, proof reissued | `public_release_file_checksum_match` | 6513631, 20275769 | 5 |

The DCC and Zenodo bytes were downloaded anonymously over HTTPS on CIT on 2026-11-10 by
`tools/retrieve_paper2_public_sources.py`. For each DCC event, the sha256 of the archive member
equals the source file sha256 recorded in the fit's source metadata. For each Zenodo event, the
md5 of the downloaded file equals the released md5, and its sha256 equals the source file sha256.
The posterior group the fit used is present in each public file. DCC archives carry no
publisher checksum, so the witness records observed digests only. GW190412_053044 is published
as `GW190412.tar`; the verifier pins that one short name.

The five reissued proofs come from `20261031-public-nativecdf-4d-v1`, whose references and
metadata these fits used (equal sha256). `reissued_from` names each earlier proof.

## Validation

Every entry is a candidate. Declared checks are recorded per event in `validation/`, in each
entry's scope and in `acceptance.declared_checks`. Limits: marginal JS 0.03 bits, q given z JS
0.06 bits, reference ESS 80. No replication split was run for these fits.

| Route | Events | Pass on seed 73 | EM limit reached |
|---|---:|---:|---:|
| DCC | 30 | 27 | 23 |
| Zenodo | 13 | 12 | 11 |
| Reissued | 5 | 0 | 1 |
| All | 48 | 39 | 35 |

Fails on seed 73: GW190517_055101, GW190519_153544, GW190620_030421, GW191109_010717,
GW190719_215514, GW190805_211137, GW190929_012149, GW230820_212515 and GW230928_215827.
The DCC splits are by row; the others are by intrinsic group.

## Not released

The manifest lists 67 events as unresolved, each with its audit reason.

- 60: the release equals the training rows only after the GWTC-4.0 cosmology reweighting. The
  training rows are the pre-transform source and hold more rows than the release.
- 6: the release reweighted a different run that shares no rows with the training source.
- 1 (GW231123_135430): the release input holds a subset of the training rows.

## Provenance

- Retrieval record: archived outside this repository (`SHA256SUMS` sha256
  `f9f0601defcfacb1…`, `retrieval.json` sha256 `ce43b822c34711fb…`).
- Fits: archived outside this repository (`SHA256SUMS` sha256 `60ca907ae185e586…`).
- Audit classification: archived outside this repository, `classification_20261109.json`
  (sha256 `e1c88bc37b0b6f00…`).
- `runtime_parity.json` compares this collection with the producing leaf (`posterior_base.py`
  sha256 `bea8929d…`), pointwise and bank-for-bank; the largest error is 0.0.
- Build: `tools/build_public_paper2_collection.py`.
