"""Read-only, hash-verified access to a typed event likelihood release.

Values are log likelihood SHAPES with arbitrary event-wise additive constants.
A complete set of event records is not a claim that every record is accepted.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from bounded_fit import log_shape,sample
from evidence import generic_proof_passes,forward_cases_pass
from public_provenance import require_public_fit

class Catalog:
    def __init__(self,index,require_complete=True,root=None):
        self.index_path=Path(index).resolve()
        self.metadata=json.loads(self.index_path.read_text())
        self.root=Path(root).resolve() if root else (self.index_path.parent/self.metadata.get("repository_root_relative_to_index","..")).resolve()
        if self.metadata.get('status')=='public_sources_pending' and require_complete:
            raise ValueError('No public-certified fits are currently available; see PUBLIC_DATA_ONLY.md')
        records=self.metadata["events"]
        if len(records)!=self.metadata["expected_events"] or len({r["event"] for r in records})!=len(records):
            raise ValueError("Catalog completeness/count mismatch")
        accepted=sum(r["acceptance"]=="accepted_for_declared_target" for r in records)
        if accepted!=self.metadata["accepted_events"] or (self.metadata["status"]=="complete" and accepted!=len(records)):
            raise ValueError("Catalog acceptance/count mismatch")
        if require_complete and self.metadata['status']!='complete':
            raise ValueError('Catalog not uniformly accepted under generic priors: %s/%s events accepted for that scope' %
                             (self.metadata['accepted_events'],self.metadata['expected_events']))
        self.events={r['event']:r for r in self.metadata['events']};self.aliases={}
        for event,row in self.events.items():
            for alias in row['aliases']:
                if alias in self.aliases and self.aliases[alias]!=event:raise ValueError('Ambiguous alias: '+alias)
                self.aliases[alias]=event
        self._fits={};self._artifacts={}
    def record(self,name):return self.events[self.aliases.get(name,name)]
    def fit(self,name,allow_unaccepted=False):
        row=self.record(name)
        if row['acceptance']!='accepted_for_declared_target' and not allow_unaccepted:
            raise ValueError('Event not accepted for generic likelihood access: '+row['event']+
                             ' (scope: '+row['acceptance']+')')
        data=row['improved']
        if data is None:raise ValueError('No replacement artifact: '+row['event'])
        require_public_fit(self.root,data['sha256'],row['event'])
        path=self.root/data['file']
        if row['event'] not in self._fits:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=data['sha256']:raise ValueError('Fit checksum mismatch: '+str(path))
            check=row.get('validation')
            if row['acceptance']=='accepted_for_declared_target':
                if not check:raise ValueError('Accepted model has no bound validation evidence')
                evidence=self.root/check['file']
                if hashlib.sha256(evidence.read_bytes()).hexdigest()!=check['sha256']:raise ValueError('Validation checksum mismatch')
                result=json.loads(evidence.read_text())
                if not generic_proof_passes(result,row['event'],data['sha256'],data.get('reference_sha256')):raise ValueError('Validation does not accept these fit bytes and declared full/heldout scope')
            artifact=json.loads(path.read_text())
            if artifact['event']!=row['event']:raise ValueError('Event identity mismatch')
            self._fits[row['event']]=artifact['fit'];self._artifacts[row['event']]=artifact
        return self._fits[row['event']]
    def log_likelihood(self,name,points,allow_unaccepted=False):
        points=np.asarray(points,float)
        if points.shape[-1]!=3:raise ValueError('Expected (mc_source [Msun], eta, chi_eff)')
        return log_shape(points,self.fit(name,allow_unaccepted))
    def samples(self,name,n=10000,seed=1,allow_unaccepted=False):
        return sample(self.fit(name,allow_unaccepted),n,seed)[0]

    def forward_posterior(self,name,n=10000,seed=1,modifier='flat_retained_coordinates'):
        """Explicitly use only the two declared forward-prior recovery views."""
        row=self.record(name)
        if row.get('acceptance')!='accepted_for_declared_forward_priors_only' or row.get('forward_validation',{}).get('status')!='pass':
            raise ValueError('No accepted forward-prior validation: '+row['event'])
        if modifier not in ['flat_retained_coordinates','spin_gaussian']:
            raise ValueError('Prior is outside the validated forward views')
        check=row['forward_validation'];path=self.root/check['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=check['sha256']:raise ValueError('Forward validation checksum mismatch')
        proof=json.loads(path.read_text());cases=[c for c in proof.get('events',[])if c.get('event')==row['event']]
        if not forward_cases_pass(cases,row['event'],row['improved']['sha256']):
            raise ValueError('Forward evidence does not accept both distinct priors for these event/fit bytes')
        from prior_folded_likelihood import sample_forward
        f=self.fit(name,allow_unaccepted=True)
        resolution=max(262144,self._artifacts[row['event']].get('validation',{}).get('integration_samples_per_seed',262144))
        return sample_forward(f,n,seed,integration_samples=resolution,modifier=modifier)
