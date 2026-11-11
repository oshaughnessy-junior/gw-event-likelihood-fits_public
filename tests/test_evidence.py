import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence import generic_proof_passes,forward_cases_pass,FORWARD_PRIORS
class EvidenceTest(unittest.TestCase):
 def test_two_distinct_forward_priors_and_fit_binding(self):
  cases=[{'event':'E','fit_sha256':'fit','prior':p,'prior_modifier':m,'status':'pass'}for p,m in FORWARD_PRIORS.items()]
  self.assertTrue(forward_cases_pass(cases,'E','fit'))
  self.assertFalse(forward_cases_pass([cases[0],cases[0]],'E','fit'))
  self.assertFalse(forward_cases_pass(cases,'other','fit'))
  self.assertFalse(forward_cases_pass(cases,'E','other'))
  cases[1]['prior_modifier']='flat_retained_coordinates'
  self.assertFalse(forward_cases_pass(cases,'E','fit'))
 def test_fullsource_only_is_not_heldout_validation(self):
  proof={'event':'E','fit_sha256':'fit','reference_sha256':'ref','status':'pass',
         'released_fit':[{'prior':p,'status':'pass'}for p in['flat_retained_coordinates','spin_gaussian']]}
  self.assertFalse(generic_proof_passes(proof,'E','fit','ref'))
  proof['holdout']={'tests':proof['released_fit']}
  self.assertTrue(generic_proof_passes(proof,'E','fit','ref'))
  self.assertFalse(generic_proof_passes(proof,'E','fit','other'))
if __name__=='__main__':unittest.main()
