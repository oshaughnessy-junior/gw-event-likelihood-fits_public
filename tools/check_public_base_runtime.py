#!/usr/bin/env python3
"""Compare a public base-measure Collection with the exact leaf source that produced each model.

Checks, per event, on seeded model-generated points: scalar log likelihood through the
Collection against the producing leaf, and the integration bank (draws and weights) of the
runtime against the producing leaf at equal seed. The producing leaf's directory must hold
the modules it imports; their hashes are recorded.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from collection_registry import Collection
import native_spin_base_redshift as runtime


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]); p.add_argument('--collection-id', required=True)
    p.add_argument('--reference-source', type=Path, required=True, help='posterior_base.py that produced the models'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--points', type=int, default=256); p.add_argument('--bank-points', type=int, default=2048); a = p.parse_args()
    if a.out.exists():
        raise ValueError('Existing parity evidence is immutable')
    sys.path.insert(0, str(a.reference_source.parent)); spec = importlib.util.spec_from_file_location('producing_base_leaf', a.reference_source)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    dependencies = {name: sha(a.reference_source.parent / name) for name in ['posterior_joint.py', 'joint_adapter.py', 'prior_contract.py']}
    c = Collection(a.root, a.collection_id); rows = []
    for event in [r['event'] for r in c.manifest['entries']]:
        artifact = c.artifact(event); model = artifact['fit']
        if model['source']['leaf_sha256'] != sha(a.reference_source):
            raise ValueError('Model was not produced by this leaf source: ' + event)
        _, _, x, _ = runtime.integration_bank(model, n=a.points, seed=581, grid_size=64)
        expected = reference.log_likelihood_x(x, model); actual = c.log_likelihood(event, x, allow_candidate=True, acknowledge_source_issues=True)
        point_error = float(np.max(abs(actual - expected)))
        ry, rlw, rx, _ = reference.integration_bank(model, a.bank_points, 1064, 256, .1); y, lw, bx, _ = runtime.integration_bank(model, a.bank_points, 1064, 256, .1)
        bank_error = dict(x=float(np.max(abs(bx - rx))), y=float(np.max(abs(y - ry))), logw=float(np.max(abs(lw - rlw))))
        if not (point_error < 1e-10 and max(bank_error.values()) < 1e-10):
            raise ValueError('Runtime parity failure: %s %r %r' % (event, point_error, bank_error))
        rows.append(dict(event=event, n_model_generated_points=len(x), max_scalar_log_likelihood_error=point_error, n_bank_points=a.bank_points, max_bank_error=bank_error,
                         fit_sha256=c.record(event)['fit']['sha256'], upstream_model_sha256=artifact['upstream_model_sha256']))
    result = dict(status='pass', events=rows, reference_source_sha256=sha(a.reference_source), reference_dependency_sha256=dependencies,
                  runtime_source_sha256=sha(Path(runtime.__file__)), producer_sha256=sha(__file__),
                  scope='Pointwise scalar likelihood and integration-bank parity of the public Collection runtime with the exact leaf source that produced each model, '
                        'on seeded model-generated points. No PE rows or private samples used. This checks the software port, not scientific fit recovery.')
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
