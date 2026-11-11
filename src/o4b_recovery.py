"""Public-certified 9D coefficient archive; explicit candidates, external grids.

This does not add a generic likelihood/Collection backend or certify source PE.
The coefficient model is restored exactly for the recorded CIT evaluator.
"""
import hashlib,json,math
from pathlib import Path
from public_provenance import require_public_fit

COLLECTION='collections/precessing/20261102-o4b-public-recovery-v1'
CORRECTED_COLLECTION='collections/precessing/20261102-o4b-cosmo-prior-corrected-v2'


COORDINATES=['mc_source','eta','s1x','s1y','s1z','s2x','s2y','s2z','redshift']
OBSERVABLES={'mc_detector','eta','s1x','s1y','s1z','s2x','s2y','s2z','chi_eff','chi_p','spin1_magnitude','spin2_magnitude','geometric4_radius','sin_geometric4_azimuth','cos_geometric4_azimuth','geometric4_residual_parallel','geometric4_residual_perpendicular','inverse_luminosity_distance','luminosity_distance','mc_source','redshift'}
PRIORS=['native','generic_source8_common_source_time_z','generic_flat_source9d']
def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def _at_least(value,limit):return _finite(value) and value>=limit
def _bounded(value,limit):return _finite(value) and 0<=value<=limit

def _case_passes(case):
    if case.get('status')!='pass_scoped_validation' or case.get('reasons')!=[]:return False
    try:
        info=case['source_original_group_ESS']
        if any(not _at_least(info[k],1000) for k in ['full','training','heldout','actual_fitting_groups']):return False
        if not _at_least(info['tuning'],0):return False
        normals=case['normalizers']
        if len(normals)!=2 or any(not isinstance(n['seed'],int) or isinstance(n['seed'],bool) or n['seed']<0 for n in normals) or len({n['seed'] for n in normals})!=2:return False
        if any(not _finite(n['logZ_up_to_native_prior_event_constant']) or not _at_least(n['integration_ESS'],1000) or not _bounded(n['relative_MCSE'],.01) for n in normals):return False
        delta=abs(math.expm1(normals[1]['logZ_up_to_native_prior_event_constant']-normals[0]['logZ_up_to_native_prior_event_constant']))
        if not _bounded(case['normalizer_relative_difference'],.02) or delta>.02 or not math.isclose(delta,case['normalizer_relative_difference'],rel_tol=1e-9,abs_tol=1e-12):return False
        if set(case['tests'])!={'full','heldout_original_groups'}:return False
        for test in case['tests'].values():
            if test['reasons']!=[] or set(test['KS'])!=OBSERVABLES or set(test['JS_bits'])!=OBSERVABLES or len(test['projected_KS'])!=32:return False
            if any(not _bounded(x,.05) for x in test['KS'].values()) or any(not _bounded(x,.03) for x in test['JS_bits'].values()) or any(not _bounded(x,.07) for x in test['projected_KS']):return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):return False

def _validated_scope(cases):
    expected={(p,m) for p in PRIORS for m in ['none','spin_gaussian']}
    keys=[(c['prior'],c['modifier']) for c in cases]
    if len(keys)!=6 or set(keys)!=expected:raise ValueError('Six distinct declared prior/modifier cases required')
    passed={p:all(_case_passes(c) for c in cases if c['prior']==p) for p in PRIORS}
    if not passed['native']:raise ValueError('Paired native recovery numerical gates fail')
    if passed[PRIORS[1]] and passed[PRIORS[2]]:return 'generic_flat_joint9d_candidate'
    if passed[PRIORS[1]]:return 'generic_source8_common_z_candidate'
    return 'native_forward_only_candidate'

def _model_passes(model):
    try:
        if model['density']['EM_converged'] is not True or not _finite(model['source_support_coverage']) or not .999<=model['source_support_coverage']<=1+1e-10:return False
        coverage=model['source_chart_coverage']['weighted_coverage_by_original_split']
        if any(not _finite(coverage[k]) or not .999<=coverage[k]<=1+1e-10 for k in ['full','training','tuning','heldout']):return False
        support=model['support_normalization'];banks=support['independent_banks']
        if len(banks)!=2 or any(not isinstance(b['seed'],int) or isinstance(b['seed'],bool) or b['seed']<0 for b in banks) or len({b['seed'] for b in banks})!=2:return False
        if any(not _at_least(b['normalization'],1e-300) or not _at_least(b['true_importance_ESS'],1000) or not _bounded(b['relative_MCSE'],.01) for b in banks):return False
        delta=abs(banks[1]['normalization']/banks[0]['normalization']-1)
        if not _bounded(support['relative_difference'],.02) or delta>.02 or not math.isclose(delta,support['relative_difference'],rel_tol=1e-9,abs_tol=1e-12):return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):return False

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _inside(base,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Artifact path must be contained')
    p=(base/p).resolve()
    if not p.is_relative_to(base.resolve()):raise ValueError('Artifact escapes archive')
    return p

def _read_bound(base,record):
    p=_inside(base,record['path']);raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Artifact checksum mismatch')
    return json.loads(raw)

class O4bRecoveryArchive:
    def __init__(self,root,*,collection=COLLECTION):
        self.root=Path(root).resolve();self.directory=_inside(self.root,collection)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if self.manifest['schema']!='o4b-public-coefficient-archive-v1':raise ValueError('Unknown archive schema')
        rows=self.manifest['events'];self.events={r['event']:r for r in rows}
        if len(rows)!=len(self.events) or len(rows)!=self.manifest['expected_events']:raise ValueError('Archive completeness mismatch')
        if sum('fit' in r for r in rows)!=self.manifest.get('public_coefficient_products',self.manifest['public_coefficient_candidates']):raise ValueError('Product count mismatch')
        if sum('fit' in r and r.get('availability','candidate')=='candidate' for r in rows)!=self.manifest['public_coefficient_candidates']:raise ValueError('Candidate count mismatch')
        prior=_read_bound(self.directory,self.manifest['native_prior_audit']);self.prior_audit={r['event']:r for r in prior['events']}
        if len(prior['events'])!=len(self.prior_audit) or set(self.prior_audit)!={r['event'] for r in rows if 'fit' in r}:raise ValueError('Native-prior audit cohort mismatch')
    def artifact(self,event,*,allow_candidate=False,acknowledge_source_issues=False,allow_diagnostic=False):
        row=self.events[event]
        if 'fit' not in row:raise ValueError('No public PE/fit for '+event)
        if not allow_candidate:raise ValueError('Explicit candidate access required; no generic acceptance')
        if not acknowledge_source_issues:raise ValueError('Acknowledge source PE robustness and prior-scope limitations')
        certificate=require_public_fit(self.root,row['fit']['sha256'],event)
        artifact=_read_bound(self.directory,row['fit'])
        if artifact['event']!=event or artifact['schema']!='public-geometric4-coefficient-archive-v1':raise ValueError('Artifact identity/schema mismatch')
        proof=_read_bound(self.directory,row['validation'])
        if proof['fit_sha256']!=artifact['upstream_fit_sha256'] or proof['fit_sha256']!=row['upstream_fit_sha256']:raise ValueError('Upstream validation binding mismatch')
        model=artifact['fit']
        if proof.get('event')!=event or model.get('event')!=event:raise ValueError('Nested model/validation event mismatch')
        if artifact.get('coordinates')!=COORDINATES or model.get('coordinates')!=COORDINATES:raise ValueError('Source9D coordinate mismatch')
        if model['reference_sha256']!=row['reference_sha256'] or model['reference_sha256']!=certificate['reference_sha256']:raise ValueError('Reference binding mismatch')
        if proof.get('reference_sha256',model['reference_sha256'])!=model['reference_sha256']:raise ValueError('Validation reference mismatch')
        if proof.get('source_record_sha256',model['source_record_sha256'])!=model['source_record_sha256']:raise ValueError('Validation source-record mismatch')
        for source in [artifact['source'],row['source']]:
            if source['source_sha256']!=certificate['source_file_sha256'] or source['source_group']!=certificate['posterior_group']:raise ValueError('Public source/group binding mismatch')
            if source['source_record_sha256']!=model['source_record_sha256']:raise ValueError('Source-record binding mismatch')
        if certificate['upstream_fit_sha256']!=artifact['upstream_fit_sha256'] or certificate['producer_revision']!='sha256:'+proof['producer_sha256']:raise ValueError('Public producer/upstream binding mismatch')
        if not _model_passes(model):raise ValueError('EM/support/chart numerical gates fail')
        scope=_validated_scope(proof['cases'])
        if artifact['scope']!=scope or row['scope']!=scope:raise ValueError('Declared scope differs from validated prior cases')
        prior=self.prior_audit[event]
        if prior['source_record_sha256']!=model['source_record_sha256'] or prior['source_file_sha256']!=certificate['source_file_sha256'] or prior['source_group']!=certificate['posterior_group'] or prior['model_distance_kind']!=model['bounds']['distance']['kind']:raise ValueError('Native-prior audit binding mismatch')
        expected='source_time' if 'UniformSourceFrame(' in str(prior.get('recorded_analytic_distance_prior')) or 'cosmo_sourceframe' in str(prior.get('recorded_RIFT_distance_option')) else None
        consistent=prior['status']=='consistent_with_public_metadata' and expected is not None and expected==prior['expected_distance_kind_from_public_metadata']==model['bounds']['distance']['kind']
        if consistent != (row.get('availability','candidate')=='candidate'):raise ValueError('Native-prior availability mismatch')
        if not consistent and not allow_diagnostic:raise ValueError('Native prior contract conflicts; diagnostic inspection only, no likelihood restoration')
        return artifact
    def restore_model(self,event,grid_directory,**options):
        if options.get('allow_diagnostic'):raise ValueError('Diagnostic acknowledgment cannot enable likelihood restoration')
        artifact=self.artifact(event,**options);asset=artifact['external_distance_grid'];p=_inside(Path(grid_directory),asset['filename']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=asset['sha256']:raise ValueError('External distance-grid checksum mismatch')
        grid=json.loads(raw)
        if canonical_sha(grid)!=asset['canonical_sha256'] or set(grid)!={'dl','log_pdf_dl'}:raise ValueError('External distance-grid content mismatch')
        model=json.loads(json.dumps(artifact['fit']));model['bounds']['distance'].update(grid)
        digest=canonical_sha(model)
        if digest!=artifact['upstream_model_canonical_sha256'] or digest!=artifact['relocation_witness']['restored_model_canonical_sha256']:raise ValueError('Model restoration differs from validated upstream model')
        return model
