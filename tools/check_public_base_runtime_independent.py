#!/usr/bin/env python3
"""Compare public runtime to the producing leaf in a separate interpreter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--collection-id',required=True);p.add_argument('--reference-source',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--reference-output',type=Path)
    p.add_argument('--bank-points',type=int,default=256);a=p.parse_args()
    root=a.root.resolve();base=root/'collections'/a.collection_id
    if a.reference_output is not None:
        # No owning-runtime module has been imported in this interpreter.
        sys.path.insert(0,str(a.reference_source.resolve().parent))
        import posterior_base as leaf
        dependencies={}
        for name in ['posterior_base','posterior_joint','joint_adapter','prior_contract']:
            path=Path(sys.modules[name].__file__).resolve()
            assert path.parent==a.reference_source.resolve().parent
            dependencies[name+'.py']=sha(path)
        manifest=json.loads((base/'manifest.json').read_text())
        for entry in manifest['entries']:
            model=json.loads((base/entry['fit']['path']).read_text())['fit']
            assert model['source']['leaf_sha256']==dependencies['posterior_base.py']
            y,lw,x,_=leaf.integration_bank(model,a.bank_points,1064,256,.1)
            values=leaf.log_likelihood_x(x,model)
            assert np.isfinite(values).all()
            np.savez(a.reference_output/(entry['analysis_id']+'.npz'),x=x,y=y,logw=lw,logL=values)
        (a.reference_output/'dependencies.json').write_text(json.dumps(dependencies))
        return
    if a.out is None or a.out.exists():
        raise ValueError('A new immutable parity-output path is required')
    sys.path.insert(0,str(root/'src'))
    from collection_registry import Collection
    import native_spin_base_redshift as runtime
    c=Collection(root,a.collection_id);c.verify();rows=[]
    with tempfile.TemporaryDirectory()as directory:
        temporary=Path(directory)
        subprocess.run([sys.executable,__file__,'--root',str(root),'--collection-id',a.collection_id,
            '--reference-source',str(a.reference_source),'--reference-output',str(temporary),
            '--bank-points',str(a.bank_points)],check=True)
        dependencies=json.loads((temporary/'dependencies.json').read_text())
        for entry in c.manifest['entries']:
            event=entry['event'];model=c.fit(event,allow_candidate=True,acknowledge_source_issues=True)
            with np.load(temporary/(entry['analysis_id']+'.npz'))as expected:
                actual=c.log_likelihood(event,expected['x'],allow_candidate=True,acknowledge_source_issues=True)
                scalar=float(np.max(abs(actual-expected['logL'])))
                y,lw,x,_=runtime.integration_bank(model,a.bank_points,1064,256,.1)
                banks=dict(x=float(np.max(abs(x-expected['x']))),y=float(np.max(abs(y-expected['y']))),
                    logw=float(np.max(abs(lw-expected['logw']))))
            if scalar>=1e-10 or max(banks.values())>=1e-10:
                raise ValueError('Independent runtime parity failed: '+event)
            rows.append(dict(event=event,fit_sha256=entry['fit']['sha256'],
                max_scalar_log_likelihood_error=scalar,max_bank_error=banks,bank_points=a.bank_points))
    d=dict(status='pass',events=rows,reference_dependency_sha256=dependencies,
        runtime_source_sha256=sha(Path(runtime.__file__)),producer_sha256=sha(__file__),
        scope='Separate reference interpreter verifies its imported source paths and hashes. Same seeded proposal draws and scalar likelihoods. This checks the software port, not source-PE robustness.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(status=d['status'],events=len(rows),maximum_scalar_error=max(r['max_scalar_log_likelihood_error']for r in rows))),flush=True)


if __name__=='__main__':main()
