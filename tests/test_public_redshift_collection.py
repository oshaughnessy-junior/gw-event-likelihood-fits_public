"""Public eligibility and explicit native-CDF 4D dispatch, using synthetic models."""
import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import test_collection_registry as registry_fixture
from public_fixture_certificate import install
from native_posterior_redshift import SCHEMA,log_likelihood_x

class PublicRedshiftCollectionTest(unittest.TestCase):
    def setUp(self):
        self.f=registry_fixture.RegistryTest('test_complete_identity_and_pairs');self.f.setUp();self.addCleanup(self.f.doCleanups);f=self.f
        f.coords=['mc_source','eta','chi_eff','z'];f.manifest.update(coordinates=f.coords,likelihood_backend='native_posterior_redshift')
        c=dict(bounded_mass_coordinate='delta',conditional_mass_chart='detector',bounded_center=[0.,0.],bounded_scale=[1.,1.],natural_parameters=[0.,0.,0.,0.,0.],bounded_log_integral_standardized=float(np.log(.9/4*1.98)),fit_bounded_mass_bounds=[0.,.9],joint_conditional_intercept=[35.,-1.],joint_conditional_slopes=[[0.,0.],[0.,0.]],joint_conditional_cov=[[100.,0.],[0.,.25]])
        self.model=dict(schema=SCHEMA,conditional_mass_chart='detector',bounded_mass_coordinate='delta',spin_caps=[.99,.99],cdf_order=64,inverse_iterations=25,components=[c],component_weights=[1.],analysis_support=dict(amax=.99,q_min=1/19,q_max=1.,z_min=.001,z_max=3.),prior_contract=dict(spin_caps=[.99,.99],distance_prior=dict(kind='distance_squared'),spin_quadrature_order=128))
        canonical=hashlib.sha256(json.dumps(self.model,sort_keys=True,separators=(',',':')).encode()).hexdigest();f.fit.update(coordinates=f.coords,fit=self.model,upstream_model_sha256='a'*64,model_canonical_sha256=canonical);fit=f.write('fit.json',f.fit);f.manifest['entries'][0]['fit']=fit;f.manifest['entries'][0]['acceptance']['status']='public_source_verified_candidate'
        upstream='a'*64;original=f.write('original-validation.json',dict(event='event-1',model_sha256=upstream,status='candidate_unresolved_or_failed',metrics={}))
        proof=dict(schema='native-redshift-candidate-validation-v1',status='candidate_unvalidated',event='event-1',analysis_id='analysis-1',fit_sha256=fit['sha256'],coordinates=f.coords,upstream_model_sha256=upstream,model_canonical_sha256=canonical,original_validation=original)
        f.manifest['entries'][0]['validation']=f.write('proof.json',proof);install(f.root,fit['sha256'],'event-1',source_model_sha256=upstream);f.save()
    def test_four_dimensional_public_candidate_dispatch(self):
        c=self.f.load();x=np.array([[30.,.2,.03,.4],[25.,.24,-.1,.2]])
        with self.assertRaisesRegex(ValueError,'not accepted'):c.log_likelihood('event-1',x)
        np.testing.assert_allclose(c.log_likelihood('event-1',x,allow_candidate=True),log_likelihood_x(x,self.model),atol=1e-12)
        self.assertEqual(c.verify()['analyses'],1)
    def test_public_eligibility_has_no_candidate_override(self):
        (self.f.root/'provenance/public_data_certificates.json').unlink()
        with self.assertRaisesRegex(ValueError,'certificate'):self.f.load().log_likelihood('event-1',[30.,.2,0.,.3],allow_candidate=True)
    def test_undeclared_backend_cannot_select_four_coordinates(self):
        self.f.manifest.pop('likelihood_backend');self.f.save()
        with self.assertRaisesRegex(ValueError,'coordinate order'):self.f.load()
    def test_candidate_report_cannot_grant_scientific_acceptance(self):
        self.f.manifest['entries'][0]['acceptance']['status']='accepted_for_declared_target';self.f.save()
        with self.assertRaisesRegex(ValueError,'passing recovery'):self.f.load().fit('event-1')
    def test_embedded_model_cannot_change_behind_original_validation(self):
        f=self.f;f.fit['fit']['components'][0]['joint_conditional_intercept'][0]+=.1
        changed=f.write('fit.json',f.fit);f.manifest['entries'][0]['fit']=changed;install(f.root,changed['sha256'],'event-1',source_model_sha256='a'*64)
        proof=json.loads((f.base/'proof.json').read_text());proof['fit_sha256']=changed['sha256'];f.manifest['entries'][0]['validation']=f.write('proof.json',proof);f.save()
        with self.assertRaisesRegex(ValueError,'embedded model'):f.load().evidence('event-1')
