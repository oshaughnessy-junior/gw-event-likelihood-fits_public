"""Structural checks for exact-byte, complete declared recovery evidence."""
GENERIC_PRIORS={'flat_retained_coordinates','spin_gaussian'}
FORWARD_PRIORS={'forward_mass_spin_prior':'flat_retained_coordinates',
 'forward_mass_spin_prior_times_spin_gaussian':'spin_gaussian'}

def cases_pass(cases,expected,modifiers=None):
 return (isinstance(cases,list)and len(cases)==len(expected)
  and {c.get('prior')for c in cases}==set(expected)
  and all(c.get('status')=='pass'for c in cases)
  and (modifiers is None or all(c.get('prior_modifier')==modifiers.get(c.get('prior'))for c in cases)))

def generic_proof_passes(proof,event,fit_sha,reference_sha):
 return (bool(reference_sha)and proof.get('event')==event
  and proof.get('fit_sha256')==fit_sha and proof.get('reference_sha256')==reference_sha
  and proof.get('status')=='pass'and cases_pass(proof.get('released_fit'),GENERIC_PRIORS)
  and cases_pass(proof.get('holdout',{}).get('tests'),GENERIC_PRIORS))

def forward_cases_pass(cases,event,fit_sha):
 return (cases_pass(cases,FORWARD_PRIORS,FORWARD_PRIORS)
  and all(c.get('event')==event and c.get('fit_sha256')==fit_sha for c in cases))
