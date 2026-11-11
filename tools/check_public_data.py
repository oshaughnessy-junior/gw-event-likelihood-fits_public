#!/usr/bin/env python3
"""Require public-source certificates for every tracked observational coefficient JSON."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from public_provenance import require_public_fit,require_promotion_at_commit
KEYS={'conditional_intercept','conditional_cov','natural_parameters','joint_conditional_intercept','gaussian_mu'}


def coefficient_product(v):
    if isinstance(v,dict):
        if any(k in v and isinstance(v[k],list)for k in KEYS):return True
        if 'means' in v and 'covariances' in v and isinstance(v['means'],list) and isinstance(v['covariances'],list):return True
        if 'mu'in v and 'cov'in v and isinstance(v['mu'],list)and isinstance(v['cov'],list):return True
        if all(isinstance(v.get(k),list)for k in ['theta','coef','cov']):return True  # native-spin base-measure component
        return any(coefficient_product(x)for x in v.values())
    return isinstance(v,list)and any(coefficient_product(x)for x in v)


def source_metadata(p,data):
    """The hash-bound source metadata a collection names for this artifact, or None."""
    coll=p.parent.parent;sources=coll/'sources.json';fit=data.get('fit')if isinstance(data.get('fit'),dict)else{}
    expected=(fit.get('source')or{}).get('metadata_sha256')if isinstance(fit.get('source'),dict)else None
    if not expected or not sources.is_file():return None
    rows=[a for a in json.loads(sources.read_text()).get('analyses',[])if a.get('analysis_id')==data.get('analysis_id')]
    name=rows[0].get('priors',{}).get('metadata_source')if len(rows)==1 else None
    if not name:return None
    path=(coll/name).resolve()
    if not path.is_relative_to(coll.resolve())or not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError("Source metadata does not match the fit's metadata_sha256: "+str(p))
    return json.loads(path.read_text())


def main():
    root=Path(__file__).resolve().parents[1];checked=[];promotions=dict(verified_at_commit=0,commit_unreachable=0)
    for p in root.rglob('*.json'):
        if '.git'in p.parts:continue
        try:data=json.loads(p.read_text())
        except (ValueError,UnicodeError):continue
        if coefficient_product(data):
            digest=hashlib.sha256(p.read_bytes()).hexdigest();isdict=isinstance(data,dict)
            proof=require_public_fit(root,digest,data.get('event','')if isdict else'',artifact=data if isdict else None,source_metadata=source_metadata(p,data)if isdict else None)
            if 'promoted_from'in proof:promotions['verified_at_commit'if require_promotion_at_commit(root,proof)else'commit_unreachable']+=1
            checked.append(str(p.relative_to(root)))
    for extension in ['*.npz','*.npy','*.h5','*.hdf5']:
        if any(root.rglob(extension)):raise ValueError('Unclassified binary data product in repository: '+extension)
    print(json.dumps(dict(status='pass',public_certified_coefficient_products=len(checked),promoted_proofs=promotions,files=checked)))


if __name__=='__main__':main()
