import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from o4b_recovery import O4bRecoveryArchive,COLLECTION,canonical_sha
COORDS=['mc_source','eta','s1x','s1y','s1z','s2x','s2y','s2z','redshift']
OBS=['mc_detector','eta','s1x','s1y','s1z','s2x','s2y','s2z','chi_eff','chi_p','spin1_magnitude','spin2_magnitude','geometric4_radius','sin_geometric4_azimuth','cos_geometric4_azimuth','geometric4_residual_parallel','geometric4_residual_perpendicular','inverse_luminosity_distance','luminosity_distance','mc_source','redshift']
from public_fixture_certificate import install

class ArchiveControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);d=self.root/COLLECTION;d.mkdir(parents=True)
        self.grid=self.root/'grids';self.grid.mkdir();g={'dl':[1.,2.],'log_pdf_dl':[0.,0.]};p=self.grid/'synthetic.json';p.write_text(json.dumps(g))
        full={'event':'GWsynthetic','bounds':{'distance':dict(g,kind='source_time')},'reference_sha256':'c'*64,'density':{'means':[[0.]],'covariances':[[[1.]]]}};full.update(coordinates=COORDS,source_record_sha256='e'*64,source_support_coverage=1.,source_chart_coverage={'weighted_coverage_by_original_split':{k:1. for k in ['full','training','tuning','heldout']}},support_normalization={'relative_difference':0.,'independent_banks':[{'seed':i,'normalization':1.,'true_importance_ESS':10000.,'relative_MCSE':.001} for i in [801,802]]});full['density']['EM_converged']=True;self.full=full
        f=json.loads(json.dumps(full));f['bounds']['distance']={'kind':'source_time'}
        a={'schema':'public-geometric4-coefficient-archive-v1','event':'GWsynthetic','fit':f,'coordinates':COORDS,'source':{'source_sha256':'b'*64,'source_group':'synthetic/posterior_samples','source_record_sha256':'e'*64},'scope':'native_forward_only_candidate','upstream_fit_sha256':'1'*64,'upstream_model_canonical_sha256':canonical_sha(full),'external_distance_grid':{'filename':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'canonical_sha256':canonical_sha(g)},'relocation_witness':{'restored_model_canonical_sha256':canonical_sha(full)}}
        self.fit=d/'fit.json';self.fit.write_text(json.dumps(a));digest=hashlib.sha256(self.fit.read_bytes()).hexdigest();self.certify(digest)
        tests={k:{'KS':{n:.01 for n in OBS},'JS_bits':{n:.001 for n in OBS},'projected_KS':[.01]*32,'reasons':[]} for k in ['full','heldout_original_groups']}
        native=[{'prior':'native','modifier':m,'status':'pass_scoped_validation','reasons':[],'source_original_group_ESS':{k:2000. for k in ['full','training','tuning','heldout','actual_fitting_groups']},'normalizers':[{'seed':i,'logZ_up_to_native_prior_event_constant':0.,'integration_ESS':10000.,'relative_MCSE':.001} for i in [803,804]],'normalizer_relative_difference':0.,'tests':tests} for m in ['none','spin_gaussian']]
        extra=[{'prior':p,'modifier':m,'status':'inconclusive_information'} for p in ['generic_source8_common_source_time_z','generic_flat_source9d'] for m in ['none','spin_gaussian']]
        v={'event':'GWsynthetic','reference_sha256':'c'*64,'source_record_sha256':'e'*64,'producer_sha256':'a'*64,'fit_sha256':'1'*64,'cases':native+extra};p=d/'validation.json';p.write_text(json.dumps(v))
        row={'event':'GWsynthetic','fit':{'path':'fit.json','sha256':digest},'validation':{'path':'validation.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'upstream_fit_sha256':'1'*64,'reference_sha256':'c'*64,'source':a['source'],'scope':'native_forward_only_candidate'}
        audit=d/'native-prior-audit.json';audit.write_text(json.dumps({'events':[{'event':'GWsynthetic','source_record_sha256':'e'*64,'source_file_sha256':'b'*64,'source_group':'synthetic/posterior_samples','model_distance_kind':'source_time','recorded_analytic_distance_prior':'UniformSourceFrame(synthetic unit fixture)','expected_distance_kind_from_public_metadata':'source_time','status':'consistent_with_public_metadata'}]}));self.manifest=d/'manifest.json';self.manifest.write_text(json.dumps({'schema':'o4b-public-coefficient-archive-v1','expected_events':1,'public_coefficient_candidates':1,'native_prior_audit':{'path':audit.name,'sha256':hashlib.sha256(audit.read_bytes()).hexdigest()},'events':[row]}))
        self.archive=O4bRecoveryArchive(self.root);self.opts=dict(allow_candidate=True,acknowledge_source_issues=True)
    def certify(self,digest):
        proof,p,registry=install(self.root,digest,'GWsynthetic');proof.update(producer_revision='sha256:'+'a'*64,upstream_fit_sha256='1'*64);p.write_text(json.dumps(proof));registry['certificates'][0]['evidence']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(self.root/'provenance/public_data_certificates.json').write_text(json.dumps(registry))
    def test_explicit_collection_and_path_containment(self):
        import shutil
        relative='collections/precessing/synthetic-new-edition'
        shutil.copytree(self.root/COLLECTION,self.root/relative)
        other=O4bRecoveryArchive(self.root,collection=relative)
        self.assertEqual(other.restore_model('GWsynthetic',self.grid,**self.opts),self.full)
        for path in ['../outside','/tmp/outside']:
            with self.assertRaises(ValueError):O4bRecoveryArchive(self.root,collection=path)
    def test_default_access_does_not_promote_candidate(self):
        with self.assertRaisesRegex(ValueError,'candidate'):self.archive.artifact('GWsynthetic')
        with self.assertRaisesRegex(ValueError,'Acknowledge'):self.archive.artifact('GWsynthetic',allow_candidate=True)
    def test_exact_model_restoration_and_asset_tamper(self):
        self.assertEqual(self.archive.restore_model('GWsynthetic',self.grid,**self.opts),self.full)
        (self.grid/'synthetic.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'grid checksum'):self.archive.restore_model('GWsynthetic',self.grid,**self.opts)
    def test_certificate_cannot_be_skipped(self):
        (self.root/'provenance/public_data_certificates.json').unlink()
        with self.assertRaisesRegex(ValueError,'certificate'):self.archive.artifact('GWsynthetic',**self.opts)
    def test_refreshed_certificate_does_not_bypass_restoration_witness(self):
        a=json.loads(self.fit.read_text());a['fit']['density']['means']=[[2.]];self.fit.write_text(json.dumps(a));digest=hashlib.sha256(self.fit.read_bytes()).hexdigest();self.certify(digest)
        m=json.loads(self.manifest.read_text());m['events'][0]['fit']['sha256']=digest;self.manifest.write_text(json.dumps(m));archive=O4bRecoveryArchive(self.root)
        with self.assertRaisesRegex(ValueError,'restoration differs'):archive.restore_model('GWsynthetic',self.grid,**self.opts)
    def test_gmm_coefficients_are_scanned(self):
        import importlib.util
        p=Path(__file__).resolve().parents[1]/'tools/check_public_data.py';s=importlib.util.spec_from_file_location('public_scan',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        self.assertTrue(m.coefficient_product({'density':{'means':[[0]],'covariances':[[[1]]]}}))
        self.assertFalse(m.coefficient_product({'diagnostic_means':[0,1]}))
