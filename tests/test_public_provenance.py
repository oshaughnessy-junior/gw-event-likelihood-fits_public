import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from public_provenance import require_public_fit
from public_fixture_certificate import install

class PublicOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.digest='1'*64
    def test_missing_certificate_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'certificate'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_exact_public_file_certificate_passes(self):
        install(self.root,self.digest,'GWsynthetic');self.assertEqual(require_public_fit(self.root,self.digest,'GWsynthetic')['release_version'],'v1')
    def test_fit_or_event_relabeling_is_not_proof(self):
        install(self.root,self.digest,'GWsynthetic')
        for digest,event in [('2'*64,'GWsynthetic'),(self.digest,'GWother')]:
            with self.assertRaisesRegex(ValueError,'withdrawn'):require_public_fit(self.root,digest,event)
    def test_tampered_proof_fails(self):
        _,p,_=install(self.root,self.digest,'GWsynthetic');p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'checksum'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def mutate(self,key,value,method='public_release_file_checksum_match'):
        proof,p,reg=install(self.root,self.digest,'GWsynthetic',method);proof[key]=value;p.write_text(json.dumps(proof));reg['certificates'][0]['evidence']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(self.root/'provenance/public_data_certificates.json').write_text(json.dumps(reg))
    def test_nonmatching_download_checksum_fails(self):
        self.mutate('observed_file_checksum',dict(algorithm='md5',value='0'*32))
        with self.assertRaisesRegex(ValueError,'differs'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_public_path_alone_is_insufficient(self):
        self.mutate('doi','public_html/release')
        with self.assertRaisesRegex(ValueError,'DOI'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_equivalence_requires_matching_columns_and_priors(self):
        install(self.root,self.digest,'GWsynthetic','public_posterior_and_prior_equivalence');require_public_fit(self.root,self.digest,'GWsynthetic')
        self.mutate('original_used_columns_sha256','0'*64,'public_posterior_and_prior_equivalence')
        with self.assertRaisesRegex(ValueError,'equivalence fails'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_unrelated_public_metadata_does_not_certify_file(self):
        self.mutate('public_file_name','GWsynthetic-unrelated-private-input.h5')
        with self.assertRaisesRegex(ValueError,'filename/checksum'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_release_doi_must_match_metadata(self):
        self.mutate('doi','10.9999/unrelated-release')
        with self.assertRaisesRegex(ValueError,'DOI mismatch'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_release_record_identity_cannot_be_relabeled(self):
        self.mutate('release_id','another-record')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_release_version_cannot_be_relabeled(self):
        self.mutate('release_version','v99')
        with self.assertRaisesRegex(ValueError,'version mismatch'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_unimplemented_public_reproduction_fails_closed(self):
        self.mutate('method','documented_public_reproduction')
        with self.assertRaisesRegex(ValueError,'Unsupported'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def record_pin(self,source='zenodo_record_id',version='record:9999999'):
        proof,p,reg=install(self.root,self.digest,'GWsynthetic');mp=self.root/proof['public_metadata']['path'];m=json.loads(mp.read_text());m['metadata'].pop('version');mp.write_text(json.dumps(m));md=hashlib.sha256(mp.read_bytes()).hexdigest();proof.update(release_version=version,release_version_source=source,public_metadata_sha256=md);proof['public_metadata']['sha256']=md;p.write_text(json.dumps(proof));reg['certificates'][0]['evidence']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(self.root/'provenance/public_data_certificates.json').write_text(json.dumps(reg))
    def test_versionless_metadata_requires_exact_record_pin(self):
        self.record_pin();self.assertEqual(require_public_fit(self.root,self.digest,'GWsynthetic')['release_version'],'record:9999999')
    def test_record_version_cannot_be_latest_or_another_record(self):
        for v in ['latest','record:latest','record:1234']:
            self.record_pin(version=v)
            with self.assertRaisesRegex(ValueError,'record pin'):require_public_fit(self.root,self.digest,'GWsynthetic')
    def test_versionless_metadata_rejects_inferred_filename_version(self):
        self.record_pin(source='public_file_name')
        with self.assertRaisesRegex(ValueError,'record pin'):require_public_fit(self.root,self.digest,'GWsynthetic')


class ProofArtifactBindingTest(unittest.TestCase):
    """Synthetic proof/artifact pairs: a proof must name what the loaded fit records."""
    def setUp(self):
        from public_provenance import require_proof_binds_artifact
        self.bind = require_proof_binds_artifact
        self.artifact = dict(source=dict(reference_sha256='a' * 64, source_sha256='b' * 64), upstream_fit_sha256='c' * 64, fit=dict(reference_sha256='a' * 64))
        self.proof = dict(event='GWsynthetic', reference_sha256='a' * 64, source_file_sha256='b' * 64, upstream_fit_sha256='c' * 64)

    def test_matching_proof_passes(self):
        self.bind(self.proof, self.artifact)

    def test_base_schema_producer_and_model_must_match(self):
        artifact = dict(upstream_model_sha256='d' * 64, fit=dict(source=dict(reference_sha256='a' * 64, fitter_sha256='e' * 64)))
        proof = dict(event='GWsynthetic', reference_sha256='a' * 64, source_model_sha256='d' * 64, producer_revision='e' * 64)
        self.bind(proof, artifact)
        for key in ['source_model_sha256', 'producer_revision']:
            with self.assertRaisesRegex(ValueError, 'differs from the fit artifact'):
                self.bind(dict(proof, **{key: 'f' * 64}), artifact)

    def test_each_recorded_field_must_match(self):
        for key in ['reference_sha256', 'source_file_sha256', 'upstream_fit_sha256']:
            for value in ['f' * 64, None]:
                proof = dict(self.proof); proof[key] = value
                with self.assertRaisesRegex(ValueError, 'differs from the fit artifact'):
                    self.bind(proof, self.artifact)


class MetadataBindingTest(unittest.TestCase):
    def test_group_and_columns_must_match_source_metadata(self):
        from public_provenance import require_metadata_binding
        proof = dict(event='GWsynthetic', posterior_group='C01:X', used_columns=['a', 'b'])
        require_metadata_binding(proof, dict(source=dict(dataset='C01:X'), public_source_proof=dict(used_columns=['b', 'a'])))
        with self.assertRaisesRegex(ValueError, 'posterior_group'):
            require_metadata_binding(proof, dict(source=dict(dataset='C01:Y')))
        with self.assertRaisesRegex(ValueError, 'used_columns'):
            require_metadata_binding(proof, dict(public_source_proof=dict(used_columns=['a'])))

    def test_file_name_must_name_the_event(self):
        root = Path(tempfile.mkdtemp()); install(root, 'd' * 64, 'GWsynthetic', public_file_name='GWother-synthetic-fixture.h5')
        with self.assertRaisesRegex(ValueError, 'does not name the event'):
            require_public_fit(root, 'd' * 64, 'GWsynthetic')
