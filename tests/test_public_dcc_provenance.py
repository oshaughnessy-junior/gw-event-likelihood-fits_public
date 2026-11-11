"""Synthetic DCC witnesses exercise archive/member and model binding failures."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from public_provenance import require_public_fit


class DCCWitnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.event='GW000000_000000';self.fit='1'*64
        self.url='https://dcc.ligo.org/public/0169/P2000223/007/'+self.event+'.tar'
        self.page=self.root/'evidence/page.html';self.page.parent.mkdir()
        self.page.write_text('<title>P2000223-v7</title><a href="'+self.url+'">synthetic fixture</a>')
        self.metadata=dict(schema='dcc-public-archive-witness-v1',event=self.event,
            release_id='P2000223',release_version='v7',
            landing_page_url='https://dcc.ligo.org/LIGO-P2000223-v7/public',
            landing_page=self.ref(self.page),archive_url=self.url,archive_sha256='a'*64,
            archive_member=self.event+'/'+self.event+'.h5',member_sha256='b'*64,
            posterior_group='SYNTHETIC',access='public_https_without_authentication',
            checksum_authority='observed_https_retrieval',retrieved_at='2026-11-10T00:00:00+00:00',retriever_sha256='c'*64)
        self.proof=dict(schema='public-fit-source-proof-v1',status='verified',event=self.event,
            fit_sha256=self.fit,reference_sha256='d'*64,posterior_group='SYNTHETIC',
            producer_revision='e'*64,leaf_revision='f'*64,source_model_sha256='2'*64,
            source_file_sha256='b'*64,public_file_sha256='a'*64,
            method='public_dcc_archive_member_match',release_id='P2000223',release_version='v7',
            public_file_url=self.url,public_file_name=self.event+'.tar',
            archive_member=self.event+'/'+self.event+'.h5',archive_member_sha256='b'*64,
            public_file_checksum=dict(algorithm='sha256',value='a'*64),
            observed_file_checksum=dict(algorithm='sha256',value='a'*64))
        self.artifact=dict(upstream_model_sha256='2'*64,fit=dict(source=dict(reference_sha256='d'*64,fitter_sha256='e'*64,leaf_sha256='f'*64)))
        self.source_metadata=dict(source=dict(dataset='SYNTHETIC'))

    def tearDown(self):self.tmp.cleanup()

    def ref(self,p):return dict(path=str(p.relative_to(self.root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    def write(self):
        p=self.root/'evidence/metadata.json';p.write_text(json.dumps(self.metadata))
        self.proof['public_metadata']=self.ref(p);self.proof['public_metadata_sha256']=self.ref(p)['sha256']
        p=self.root/'evidence/proof.json';p.write_text(json.dumps(self.proof))
        registry=self.root/'provenance/public_data_certificates.json';registry.parent.mkdir(exist_ok=True)
        registry.write_text(json.dumps(dict(schema='public-fit-certificates-v1',certificates=[dict(event=self.event,fit_sha256=self.fit,evidence=self.ref(p))])))

    def load(self):
        self.write()
        return require_public_fit(self.root,self.fit,self.event,artifact=self.artifact,source_metadata=self.source_metadata)

    def test_valid_synthetic_archive_member_and_model(self):
        self.assertEqual(self.load()['archive_member_sha256'],'b'*64)

    def test_host_substitution_rejected(self):
        self.proof['public_file_url']=self.url.replace('dcc.ligo.org','example.org')
        with self.assertRaisesRegex(ValueError,'archive URL'):self.load()

    def test_other_document_version_rejected(self):
        self.proof['release_version']='v5'
        with self.assertRaisesRegex(ValueError,'v7 release'):self.load()

    def test_archive_byte_substitution_rejected(self):
        self.proof['observed_file_checksum']['value']='3'*64
        with self.assertRaisesRegex(ValueError,'archive checksum'):self.load()

    def test_archive_member_substitution_rejected(self):
        self.proof['archive_member_sha256']='3'*64
        with self.assertRaisesRegex(ValueError,'member differs'):self.load()

    def test_unlisted_archive_rejected(self):
        self.page.write_text('<title>P2000223-v7</title>no archive link')
        self.metadata['landing_page']=self.ref(self.page)
        with self.assertRaisesRegex(ValueError,'does not list'):self.load()

    def test_page_corruption_rejected(self):
        self.page.write_text(self.page.read_text()+'changed')
        with self.assertRaisesRegex(ValueError,'page snapshot checksum'):self.load()

    def test_page_traversal_rejected(self):
        self.metadata['landing_page']['path']='../page.html'
        with self.assertRaisesRegex(ValueError,'must be contained'):self.load()

    def test_private_retrieval_label_rejected(self):
        self.metadata['access']='authenticated'
        with self.assertRaisesRegex(ValueError,'release/access'):self.load()

    def test_rebound_model_reference_rejected(self):
        self.artifact['fit']['source']['reference_sha256']='3'*64
        with self.assertRaisesRegex(ValueError,'reference_sha256 differs'):self.load()

    def test_rebound_posterior_group_rejected(self):
        self.source_metadata['source']['dataset']='ANOTHER'
        with self.assertRaisesRegex(ValueError,'posterior_group differs'):self.load()


if __name__=='__main__':unittest.main()
