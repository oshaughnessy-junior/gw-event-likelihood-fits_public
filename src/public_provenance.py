"""Fail-closed public-source eligibility, independent of scientific fit acceptance."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse,unquote

REGISTRY='provenance/public_data_certificates.json'
METHODS={'public_release_file_checksum_match','public_posterior_and_prior_equivalence','public_dcc_archive_member_match'}


def _digest(value,length=64):
    return isinstance(value,str)and re.fullmatch('[0-9a-f]{%d}'%length,value)is not None


def _bindings(artifact):
    """(proof key, artifact value, where) pairs that a proof must match for the loaded fit artifact."""
    out=[];fit=artifact.get('fit')if isinstance(artifact.get('fit'),dict)else{};fsrc=fit.get('source')if isinstance(fit.get('source'),dict)else{}
    asrc=artifact.get('source')if isinstance(artifact.get('source'),dict)else{}
    for value,where in [(fsrc.get('reference_sha256'),'fit.source.reference_sha256'),(fit.get('reference_sha256'),'fit.reference_sha256'),(asrc.get('reference_sha256'),'source.reference_sha256')]:
        if value is not None:out.append(('reference_sha256',value,where))
    if fsrc.get('fitter_sha256')is not None:out.append(('producer_revision',fsrc['fitter_sha256'],'fit.source.fitter_sha256'))
    if fsrc.get('leaf_sha256')is not None:out.append(('leaf_revision',fsrc['leaf_sha256'],'fit.source.leaf_sha256'))
    if artifact.get('upstream_model_sha256')is not None:out.append(('source_model_sha256',artifact['upstream_model_sha256'],'upstream_model_sha256'))
    if artifact.get('upstream_fit_sha256')is not None:out.append(('upstream_fit_sha256',artifact['upstream_fit_sha256'],'upstream_fit_sha256'))
    if asrc.get('source_sha256')is not None:out.append(('source_file_sha256',asrc['source_sha256'],'source.source_sha256'))
    return out


def require_proof_binds_artifact(proof,artifact):
    """The proof must name the reference, model, producer and source that the loaded fit artifact records."""
    for key,value,where in _bindings(artifact):
        if proof.get(key)!=value:raise ValueError('Public proof %s differs from the fit artifact %s: %s'%(key,where,proof.get('event')))


PROMOTION_REPLACED={'status','draft_note','fit_sha256','public_metadata'}
PROMOTION_ADDED={'leaf_revision','source_model_sha256','promoted_from'}


def _contained_file(root,relative,what):
    relative=Path(relative)
    if relative.is_absolute()or '..'in relative.parts:raise ValueError(what+' path must be contained')
    path=(root/relative).resolve()
    if not path.is_relative_to(root)or not path.is_file():raise ValueError(what+' missing')
    return path


def require_promotion(root,proof):
    """A promoted proof equals its in-tree draft on every field except the declared promotion changes."""
    origin=proof['promoted_from'];path=_contained_file(root,origin.get('path',''),'Draft proof')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=origin.get('sha256'):raise ValueError('Draft proof checksum differs from promoted_from: '+proof['event'])
    draft=json.loads(path.read_text())
    if draft.get('schema')!='public-fit-source-proof-v1'or draft.get('status')!='draft'or draft.get('event')!=proof['event']:raise ValueError('promoted_from must name a draft proof of this event')
    changed=sorted(k for k in set(draft)-PROMOTION_REPLACED if proof.get(k)!=draft[k])
    extra=sorted(set(proof)-set(draft)-PROMOTION_ADDED)
    if changed or extra:raise ValueError('Promoted proof departs from its draft (%s): %s'%(', '.join(changed+extra),proof['event']))
    if proof.get('source_model_sha256')!=draft.get('fit_sha256'):raise ValueError('Promoted proof names another model than its draft: '+proof['event'])


def require_promotion_at_commit(root,proof):
    """True if the draft at promoted_from.commit has promoted_from.sha256; None if git or the commit is unavailable."""
    import subprocess
    origin=proof['promoted_from']
    try:out=subprocess.run(['git','-C',str(root),'show','%s:%s'%(origin['commit'],origin['path'])],capture_output=True,timeout=30)
    except (OSError,subprocess.SubprocessError):return None
    if out.returncode:return None
    if hashlib.sha256(out.stdout).hexdigest()!=origin['sha256']:raise ValueError('Draft proof at %s differs from promoted_from: %s'%(origin['commit'][:12],proof['event']))
    return True


def require_metadata_binding(proof,metadata):
    """The fit's hash-bound source metadata must name the proof's posterior group and used columns, where it records them."""
    dataset=(metadata.get('source')or{}).get('dataset')
    if dataset is not None and dataset!=proof.get('posterior_group'):raise ValueError('Public proof posterior_group differs from the fit source metadata: '+proof['event'])
    columns=(metadata.get('public_source_proof')or{}).get('used_columns')
    if columns is not None and sorted(columns)!=sorted(proof.get('used_columns')or[]):raise ValueError('Public proof used_columns differ from the fit source metadata: '+proof['event'])


def require_public_fit(root,fit_sha256,event,artifact=None,source_metadata=None):
    root=Path(root).resolve();path=root/REGISTRY
    if not path.is_file():raise ValueError('Public-data provenance certificate required: '+event)
    registry=json.loads(path.read_text())
    if registry.get('schema')!='public-fit-certificates-v1':raise ValueError('Unsupported public provenance registry')
    matches=[r for r in registry['certificates']if r.get('fit_sha256')==fit_sha256 and r.get('event')==event]
    if len(matches)!=1:raise ValueError('Fit withdrawn pending exact public-data proof: '+event)
    cert=matches[0];ref=cert.get('evidence',{});relative=Path(ref.get('path',''))
    if relative.is_absolute()or '..'in relative.parts:raise ValueError('Public evidence path must be contained')
    evidence=(root/relative).resolve()
    if not evidence.is_relative_to(root)or not evidence.is_file():raise ValueError('Public evidence file missing')
    if hashlib.sha256(evidence.read_bytes()).hexdigest()!=ref.get('sha256'):raise ValueError('Public evidence checksum mismatch')
    proof=json.loads(evidence.read_text())
    if proof.get('schema')!='public-fit-source-proof-v1'or proof.get('status')!='verified':raise ValueError('Public source proof is not verified')
    if proof.get('fit_sha256')!=fit_sha256 or proof.get('event')!=event:raise ValueError('Public proof does not bind these fit bytes/event')
    if not isinstance(proof.get('public_file_name'),str)or event not in proof['public_file_name']:raise ValueError('Public file name does not name the event: '+event)
    if proof.get('method')not in METHODS:raise ValueError('Unsupported public-source proof method')
    for key in ['release_id','release_version','posterior_group','producer_revision']:
        if not isinstance(proof.get(key),str)or not proof[key].strip():raise ValueError('Public proof lacks '+key)
    if proof['method']!='public_dcc_archive_member_match' and not re.fullmatch(r'10\.\d{4,9}/\S+',proof.get('doi','')):raise ValueError('Public source DOI required')
    if urlparse(proof.get('public_file_url','')).scheme!='https':raise ValueError('Public file URL required')
    for key in ['source_file_sha256','reference_sha256','public_metadata_sha256','public_file_sha256']:
        if not _digest(proof.get(key)):raise ValueError('Public proof lacks exact '+key)
    checksum=proof.get('public_file_checksum',{})
    if checksum.get('algorithm')not in ['md5','sha256']or not _digest(checksum.get('value'),32 if checksum.get('algorithm')=='md5'else 64):raise ValueError('Public release file checksum required')
    metadata_ref=proof.get('public_metadata',{});mp=Path(metadata_ref.get('path',''))
    if mp.is_absolute()or '..'in mp.parts:raise ValueError('Public metadata path must be contained')
    mp=(root/mp).resolve()
    if not mp.is_relative_to(root)or not mp.is_file():raise ValueError('Public metadata snapshot missing')
    mdigest=hashlib.sha256(mp.read_bytes()).hexdigest()
    if mdigest!=proof['public_metadata_sha256']or mdigest!=metadata_ref.get('sha256'):raise ValueError('Public metadata checksum mismatch')
    metadata=json.loads(mp.read_text())
    if proof['method']=='public_dcc_archive_member_match':
        from public_dcc_provenance import require_dcc_witness
        require_dcc_witness(root,proof,metadata)
        if artifact is not None:require_proof_binds_artifact(proof,artifact)
        if source_metadata is not None:require_metadata_binding(proof,source_metadata)
        return proof
    if not isinstance(metadata.get('id'),int)or metadata['id']<=0:raise ValueError('Public record identity missing')
    if proof['release_id']!=str(metadata['id']):raise ValueError('Public release identity mismatch')
    version=metadata.get('metadata',{}).get('version')
    if version is None:
        if proof.get('release_version_source')!='zenodo_record_id'or proof['release_version']!='record:'+str(metadata['id']):raise ValueError('Missing version requires an exact Zenodo record pin')
    elif proof['release_version']!=version:raise ValueError('Public release version mismatch')
    if metadata.get('access',{}).get('files')!='public'and metadata.get('metadata',{}).get('access_right')!='open':raise ValueError('Release metadata does not declare open/public file access')
    if metadata.get('doi',metadata.get('metadata',{}).get('doi'))!=proof['doi']:raise ValueError('Public metadata DOI mismatch')
    files=[f for f in metadata.get('files',[])if f.get('key')==proof.get('public_file_name')]
    if len(files)!=1 or files[0].get('checksum')!=checksum['algorithm']+':'+checksum['value']:raise ValueError('Released filename/checksum not present in public metadata')
    url=urlparse(proof['public_file_url'])
    if url.hostname!='zenodo.org':raise ValueError('Current public metadata verifier supports Zenodo releases only')
    allowed=list(files[0].get('links',{}).values())+[f"https://zenodo.org/records/{metadata.get('id')}/files/{proof['public_file_name']}"]
    if unquote(url.path)not in [unquote(urlparse(u).path)for u in allowed if isinstance(u,str)]:raise ValueError('Public file URL not bound to metadata file')
    if proof.get('observed_file_checksum')!=checksum:raise ValueError('Observed public-file checksum differs from released checksum')
    if proof['method']=='public_release_file_checksum_match'and proof['public_file_sha256']!=proof['source_file_sha256']:raise ValueError('Fit source is not the authenticated public file')
    if proof['method']=='public_posterior_and_prior_equivalence':
        for key in ['public_used_columns_sha256','original_used_columns_sha256','public_prior_sha256','original_prior_sha256']:
            if not _digest(proof.get(key)):raise ValueError('Equivalence proof lacks '+key)
        if proof['public_used_columns_sha256']!=proof['original_used_columns_sha256']or proof['public_prior_sha256']!=proof['original_prior_sha256']:raise ValueError('Public posterior/prior equivalence fails')
        if not isinstance(proof.get('used_columns'),list)or not proof['used_columns']:raise ValueError('Equivalence proof must name every used column and weight')
    if 'promoted_from'in proof:require_promotion(root,proof)
    if artifact is not None:require_proof_binds_artifact(proof,artifact)
    if source_metadata is not None:require_metadata_binding(proof,source_metadata)
    return proof
