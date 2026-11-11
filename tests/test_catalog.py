import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from catalog import Catalog
from public_fixture_certificate import install

class CatalogIntegrityTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'releases').mkdir();(self.root/'fits').mkdir()
  f={'event':'GWtest','fit':{'eta_a':0.,'eta_b':0.,'conditional_intercept':[30.,0.],
     'conditional_slope':[0.,0.],'conditional_cov':[[1.,0.],[0.,.01]]}}
  self.fit=self.root/'fits/event.json';self.fit.write_text(json.dumps(f));self.index=self.root/'releases/index.json'
  self.obj={'status':'complete','expected_events':1,'accepted_events':1,'repository_root_relative_to_index':'..',
     'events':[{'event':'GWtest','aliases':['Stest'],'acceptance':'accepted_for_declared_target','paper_records':[],
     'improved':{'file':'fits/event.json','sha256':hashlib.sha256(self.fit.read_bytes()).hexdigest()}}]}
  proof=self.root/'validation.json'
  proof.write_text(json.dumps({'event':'GWtest','status':'pass','fit_sha256':hashlib.sha256(self.fit.read_bytes()).hexdigest(),
    'reference_sha256':'reference','released_fit':[{'prior':p,'status':'pass'}for p in['flat_retained_coordinates','spin_gaussian']],
    'holdout':{'tests':[{'prior':p,'status':'pass'}for p in['flat_retained_coordinates','spin_gaussian']]}}))
  self.obj['events'][0]['improved']['reference_sha256']='reference'
  self.obj['events'][0]['validation']={'file':'validation.json','sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}
  install(self.root,self.obj['events'][0]['improved']['sha256'],'GWtest')
  self.write()
 def write(self):self.index.write_text(json.dumps(self.obj))
 def tearDown(self):self.tmp.cleanup()
 def test_alias_and_joint_evaluation(self):
  cat=Catalog(self.index);self.assertTrue(np.isfinite(cat.log_likelihood('Stest',[30,.2,0])))
  self.assertTrue(np.isneginf(cat.log_likelihood('GWtest',[30,.3,0])))
 def test_hash_mismatch_fails(self):
  self.fit.write_text('{}')
  with self.assertRaisesRegex(ValueError,'checksum'):Catalog(self.index).fit('GWtest')
 def test_forward_api_rechecks_distinct_prior_proof(self):
  row=self.obj['events'][0];row['acceptance']='accepted_for_declared_forward_priors_only'
  self.obj['status']='incomplete_scientific_acceptance';self.obj['accepted_events']=0
  proof=self.root/'forward.json';case={'event':'GWtest','fit_sha256':row['improved']['sha256'],
      'prior':'forward_mass_spin_prior','prior_modifier':'flat_retained_coordinates','status':'pass'}
  proof.write_text(json.dumps({'events':[case,case]}))
  row['forward_validation']={'file':'forward.json','sha256':hashlib.sha256(proof.read_bytes()).hexdigest(),'status':'pass'};self.write()
  with self.assertRaisesRegex(ValueError,'both distinct priors'):
   Catalog(self.index,require_complete=False).forward_posterior('GWtest')
 def test_missing_validation_fails_closed(self):
  del self.obj['events'][0]['validation'];self.write()
  with self.assertRaisesRegex(ValueError,'no bound validation'):Catalog(self.index).fit('GWtest')
 def test_validation_evidence_is_bound_to_fit_bytes(self):
  proof=self.root/'validation.json';proof.write_text(json.dumps({'status':'pass','fit_sha256':'wrong'}))
  self.obj['events'][0]['validation']={'file':'validation.json','sha256':hashlib.sha256(proof.read_bytes()).hexdigest()};self.write()
  with self.assertRaisesRegex(ValueError,'does not accept'):Catalog(self.index).fit('GWtest')
  proof.write_text('{}')
  with self.assertRaisesRegex(ValueError,'Validation checksum'):Catalog(self.index).fit('GWtest')
 def test_no_silent_acceptance_or_event_drop(self):
  self.obj['expected_events']=2;self.write()
  with self.assertRaisesRegex(ValueError,'completeness'):Catalog(self.index)
 def test_unaccepted_records_remain_visible_but_fail_closed(self):
  self.obj['status']='incomplete_scientific_acceptance';self.obj['accepted_events']=0;self.obj['events'][0]['acceptance']='fail';self.write()
  with self.assertRaisesRegex(ValueError,'not uniformly accepted under generic priors'):Catalog(self.index)
  cat=Catalog(self.index,require_complete=False);self.assertEqual(cat.record('Stest')['acceptance'],'fail')
  with self.assertRaisesRegex(ValueError,'not accepted'):cat.log_likelihood('Stest',[30,.2,0])

if __name__=='__main__':unittest.main()
