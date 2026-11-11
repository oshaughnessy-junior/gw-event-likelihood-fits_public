#!/usr/bin/env python3
"""Compare public Collection scores to the exact pinned producing leaf source."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from collection_registry import Collection
import native_posterior_redshift as native


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--collection-id',required=True);p.add_argument('--reference-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Existing parity evidence is immutable')
    sys.path.insert(0,str(a.reference_source.parent));spec=importlib.util.spec_from_file_location('producing_native_leaf',a.reference_source);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    c=Collection(a.root,a.collection_id);rows=[]
    for event in c.manifest['requested_events']:
        artifact=c.artifact(event);model=artifact['fit'];assert model['source']['adapter_sha256']==sha(a.reference_source)
        _,_,x,_=native.integration_bank(model,n=64,seed=581,grid_size=64)
        expected=reference.log_likelihood_x(x,model);actual=c.log_likelihood(event,x,allow_candidate=True,acknowledge_source_issues=True)
        error=float(np.max(abs(actual-expected)));assert error<1e-10
        rows.append(dict(event=event,n_model_generated_points=len(x),max_scalar_log_likelihood_error=error,fit_sha256=c.record(event)['fit']['sha256'],upstream_model_sha256=artifact['upstream_model_sha256']))
    result=dict(status='pass',events=rows,reference_source_sha256=sha(a.reference_source),runtime_source_sha256=sha(Path(native.__file__)),producer_sha256=sha(__file__),scope='Pointwise scalar likelihood parity of the public Collection runtime with the exact source that produced each model, on independent seeded model-generated points. No original PE rows, integration banks or private hyperposterior samples used. This checks the software port, not scientific fit recovery.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
