# Gravitational-wave event likelihood fits

Fitted likelihood models for gravitational-wave events, built only from public LVK
posterior releases. Each fit is bound to its public source file by a certificate in
`provenance/public_data_certificates.json`.

No posterior samples, distance grids or integration banks are distributed.
A fit's validation record states whether it passed its declared checks. Failures are kept.

The records are exported from a private working repository. Host paths in them are replaced by
`<host-path>/` and the file name, and sha256 bindings name the cleaned files.
`EXPORT_MANIFEST.json` gives the source sha256 of each cleaned file.

Promoted proofs name the draft proofs they came from (`promoted_from`); the drafts are included
and the verifier compares each promoted proof with its draft. This repository has no git history
from the working repository, so a draft cannot be traced to the commit named in `promoted_from`.

## Collections

| Collection | Fits |
|---|---:|
| [precessing/20261102-o4b-cosmo-prior-corrected-v2](collections/precessing/20261102-o4b-cosmo-prior-corrected-v2/README.md) | 104 |
| [quasicircular/20261031-public-nativecdf-4d-v1](collections/quasicircular/20261031-public-nativecdf-4d-v1/README.md) | 5 |
| [quasicircular/20261109-public-base8-4d-gwtc41-v1](collections/quasicircular/20261109-public-base8-4d-gwtc41-v1/README.md) | 7 |
| [quasicircular/20261109-public-base8-4d-jointv2-v1](collections/quasicircular/20261109-public-base8-4d-jointv2-v1/README.md) | 38 |
| [quasicircular/20261109-public-base8-4d-v1](collections/quasicircular/20261109-public-base8-4d-v1/README.md) | 2 |
| [quasicircular/20261110-public-longem-4d-v1](collections/quasicircular/20261110-public-longem-4d-v1/README.md) | 203 |

## Loading a fit

Run from the repository root. Each collection README states its
coordinates, priors, scope and the acknowledgments its reader requires.

```python
import sys
sys.path.insert(0, 'src')
from collection_registry import Collection
c = Collection('.', 'quasicircular/20261110-public-longem-4d-v1')
event = c.manifest['entries'][0]['event']
# points: array of shape (n, len(c.coordinates)), in manifest coordinate order
logL = c.log_likelihood(event, points, allow_candidate=True, acknowledge_source_issues=True)
```

```python
import sys
sys.path.insert(0, 'src')
from o4b_recovery import O4bRecoveryArchive
a = O4bRecoveryArchive('.', collection='collections/precessing/20261102-o4b-cosmo-prior-corrected-v2')
event = next(e for e, row in a.events.items() if 'fit' in row)
artifact = a.artifact(event, allow_candidate=True, acknowledge_source_issues=True)
```

Restoring a full 9D model needs distance grids that are not distributed here.

## Verifying

```sh
python tools/check_public_data.py
python tools/check_collections.py --collection quasicircular/20261031-public-nativecdf-4d-v1
python tools/check_collections.py --collection quasicircular/20261109-public-base8-4d-gwtc41-v1
python tools/check_collections.py --collection quasicircular/20261109-public-base8-4d-jointv2-v1
python tools/check_collections.py --collection quasicircular/20261109-public-base8-4d-v1
python tools/check_collections.py --collection quasicircular/20261110-public-longem-4d-v1
python -m unittest discover -s tests
```

`EXPORT_MANIFEST.json` lists the sha256 of every file and the source commit.
Requires Python 3 with numpy and scipy.

## Citation and license

Exported from commit `65a13772f675ecf27726ac25662d60ad1e860060` (tag `public-export-v1`) of the private working repository.

Cite with `CITATION.cff`. DOI: pending.
License: CC BY 4.0 (`LICENSE`).
