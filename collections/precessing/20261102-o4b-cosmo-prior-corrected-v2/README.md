# Corrected O4b public recovery candidates

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

This edition corrects our native distance-prior conversion for 37 public
SEOBNRv5PHM inputs. The official RIFT records explicitly declare
`cosmo_sourceframe`; our old reader incorrectly used a LALInference
`DL_squared` fallback. The learned posterior-density coefficients and exact
original source/reference/weight/group bindings are preserved. All six cases
were rerun using the exact uncapped weighted CDF and unchanged gates, followed
by a separate source/model/proof audit. The other 67 coefficient wrappers and
validation records are reused from the prior edition.

All104 public-PE events have prior-consistent paired-native candidates;
GW240406_062847 remains explicit as lacking public PE (105-event search census).
At FAR<0.25/year,94 of95 search events have public PE. Per-event native/common-z/
flat9D scopes remain mandatory. This is not a waveform, PE-chain, arbitrary-prior
or absolute-evidence certificate. The RIFT-SEOB reference-PE coordinate concern
remains separate; this correction does not generate new reference PE.

Use `O4bRecoveryArchive(root, collection="collections/precessing/20261102-o4b-cosmo-prior-corrected-v2")` from
`src/o4b_recovery.py`, explicitly acknowledging candidate and source limitations.
External distance grids are hash-bound and not distributed; this edition does not
record where they are kept. The historical collection retains its 37 quarantines unchanged.

The through-O4a stable `current` pointer is unchanged. Raw PE, raw source records,
distance grids and integration banks are not distributed here. Public Zenodo
metadata and exact fit/source certificates are required for every wrapper.
