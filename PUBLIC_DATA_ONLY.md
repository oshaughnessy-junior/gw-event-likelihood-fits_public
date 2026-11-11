# Public input eligibility

Current active distribution: **255 public-source-certified four-dimensional
candidates covering 238 events, and 104 public-source-certified, prior-consistent nine-dimensional
recovery candidates; zero universally/generically accepted models**. The corrected
9D edition repairs our erroneous `DL_squared` conversion for 37 public RIFT
`cosmo_sourceframe` inputs and preserves the other 67 models. All37 corrected
products retain exact public-source/reference certificates and rerun scoped
validation. Public eligibility, numerical recovery and reference-PE robustness
remain separate. The historical104-product edition, including37 blocked prior
conversions, is preserved; duplicate editions do not add events to coverage.
See `collections/precessing/20261102-o4b-cosmo-prior-corrected-v2/`.
The legacy 3D catalog remains unavailable. The five-event 4D collection is at
`collections/quasicircular/20261031-public-nativecdf-4d-v1/`. A second 4D edition,
`collections/quasicircular/20261109-public-base8-4d-v1/`, refits two of those events
from the same certified references; it adds no events to coverage. A third 4D edition,
`collections/quasicircular/20261109-public-base8-4d-jointv2-v1/`, adds the 38 base8 fits
that paper 2 uses whose drafted Zenodo proofs bind the exact fit bytes; 10 fail the
producer declared checks, as recorded per event. A fourth,
`collections/quasicircular/20261109-public-base8-4d-gwtc41-v1/`, adds 7 GWTC-4 events
exported from GWTC-4.1 release files and refit; no population analysis uses them, and
each passes the declared checks on three splits.

Only fits derived from verifiably public data may be distributed in this tree.
Recovery accuracy, collaborator authorization and a planned public release are
separate from public availability. Unverified inputs fail closed.

Each fit needs a certificate in `provenance/public_data_certificates.json`, bound
to its event and exact coefficient SHA256. The certificate names a hash-verified
`public-fit-source-proof-v1` JSON evidence file. The runtime validates its release
DOI/version, public file URL, released and observed file checksums, exact file and
reference hashes, posterior group and producer revision.

Supported evidence routes are an exact public-release file match, a versioned DCC archive-member witness, or a reproducible
public posterior-and-prior equivalence witness. A separate public reproduction
validator is not implemented; that route fails closed.
An equivalence witness must identify every used column and weight and bind equal
canonical hashes for the original/public samples and priors. Changes in file
packaging alone need not force a refit. Nonidentical inference inputs require a
new fit edition and independent scientific validation.

Provenance records may describe withdrawn products; they do not grant access to
missing coefficients. No opt-in candidate flag bypasses public eligibility.
Synthetic unit tests use explicitly fabricated fixtures, never observational
coefficients. New release paths and DOI labels must not be substituted for proof.

When official Zenodo metadata supplies no human version, the only supported
alternative is an exact `record:<record_id>` version with
`release_version_source=zenodo_record_id`, bound to the actual record ID, DOI and
file checksum. Inferred filename versions and `latest` aliases are rejected.


## GWTC-2 DCC archive members

The long-EM 4D edition supports the pinned public document `P2000223-v7`.
A saved public landing page must list the archive URL.
The builder downloads each archive through anonymous HTTPS and checks its SHA256.
It reads the named regular HDF5 member and compares its hash to the fit source.
The runtime binds the page, archive, member, posterior group, reference and model.
The witness labels its checksum authority as observed HTTPS retrieval.
It does not claim a publisher-issued checksum or invent a Zenodo record.
Other DCC documents and versions fail closed.

The current long-EM edition adds 203 candidates with native shape checks on three splits.
Broader-prior verdicts remain recorded and do not filter this edition.
The consumer selection names the current preferred public fits for each paper event.
