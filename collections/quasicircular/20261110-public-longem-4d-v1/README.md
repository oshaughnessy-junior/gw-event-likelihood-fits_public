# Current public long-EM four-dimensional fits

In this repository each absolute path on a cluster host, with its site prefix, is replaced by `<host-path>/` and the file name, and each sha256 binding names the cleaned file. Coefficients, outcomes and other values are unchanged. Lineage hashes (parent fit, parent certificate, source model, earlier proof) name the files that were used. `EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

Collection: `quasicircular/20261110-public-longem-4d-v1`.

This edition contains 203 public-source-verified native-posterior refits.
Every model passes native shape checks on three intrinsic-group splits.
All three splits pass the broader declared tests for 129 events.
The other 74 retain their failed checks in the evidence records.
GW191219 remains unresolved after producer or validation failures.

The primary fit uses split seed 73 and eight mixture components.
The fitter permits up to 2,000 EM iterations, with tolerance 2e-5.
Splits 101 and 202 provide replication evidence. Previous editions remain immutable.

Source verification covers 201 Zenodo release files and two DCC archive members.
The DCC witnesses bind fresh anonymous HTTPS downloads from GWTC-2 v7.
They record observed archive and member hashes, without claiming publisher-issued checksums.
Each proof also binds the reference, model, metadata and producer.
Raw PE and integration banks remain outside Git.

The population-paper selection of 2026-11-10 is kept in the private working repository and
names fits by their source sha256 (`EXPORT_MANIFEST.json`). It selects fits for all 179 all-mass events and 159 of 160 high-mass events.
It uses native recovery to select completed refits, retaining broader-prior outcomes.
Fit-option comparisons support the appendix stability checks.

Load the named `Collection` with `allow_candidate=True` and
`acknowledge_source_issues=True`. These flags preserve the public-source gate.
Validation records state the recovery scope of each event.

Source verification was produced by `tools/publish_public_catalog_refits.py`.
Check public eligibility with `tools/check_public_data.py`.
Verify this collection with
`tools/check_collections.py --collection quasicircular/20261110-public-longem-4d-v1`.
`tools/check_public_base_runtime.py` compares likelihoods and banks to the producing leaf.
The source campaign is archived outside this repository.
