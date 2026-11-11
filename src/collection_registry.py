"""Versioned, provenance-checked collections alongside the legacy 3D Catalog.

An analysis is not an event: one event may have several releases/waveforms.
No implicit latest release, alternate waveform, prior scope, or data correction
is selected. Loading a model verifies bytes; it does not validate the source PE.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from public_provenance import require_public_fit
from pathlib import Path


GENERIC = "accepted_for_declared_target"
SCOPED = "accepted_for_declared_forward_priors_only"
# Native-posterior 4D leaves (mc_source, eta, chi_eff, z): spin-CDF chart, and spin prior as base measure.
NATIVE_REDSHIFT_BACKENDS = ("native_posterior_redshift", "native_spin_base_redshift")


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(base, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Expected a contained relative artifact path")
    result = (base / p).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("Artifact path escapes the collection")
    return result


def _verified(base, ref):
    p = _inside(base, ref["path"])
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("Artifact checksum mismatch: " + str(p))
    return json.loads(raw)


class Collection:
    """Explicit collection identity with independent recovery/source-quality scopes."""

    def __init__(self, root, collection_id):
        self.root = Path(root).resolve()
        self.directory = _inside(self.root / "collections", collection_id)
        self.manifest = _read(self.directory / "manifest.json")
        if self.manifest.get("schema") != "nal-collection-v1":
            raise ValueError("Unsupported collection manifest schema")
        if self.manifest.get("collection_id") != collection_id:
            raise ValueError("Collection identity mismatch")
        self.coordinates = tuple(self.manifest["coordinates"])
        if len(self.coordinates) != len(set(self.coordinates)):
            raise ValueError("Duplicate coordinate names")
        if self.manifest["family"] not in {"quasicircular", "eccentric"}:
            raise ValueError("Unknown physical family")
        self.backend = self.manifest.get("likelihood_backend", "default")
        supported = ("mc_source", "eta", "chi_eff")
        if self.backend in NATIVE_REDSHIFT_BACKENDS:
            if self.manifest["family"] != "quasicircular":
                raise ValueError("Native aligned/redshift backend requires quasicircular family")
            supported += ("z",)
        elif self.backend != "default":
            raise ValueError("Unsupported likelihood backend")
        elif self.manifest["family"] == "eccentric":
            supported += ("eccentricity", "redshift")
        if self.coordinates != supported:
            raise ValueError("Unsupported backend coordinate order: " + repr(self.coordinates))
        data = _verified(self.directory, self.manifest["source_manifest"])
        if data.get("schema") != "nal-sources-v1":
            raise ValueError("Unsupported source manifest schema")
        self.sources = {}
        for source in data["analyses"]:
            self._check_source(source)
            key = source["analysis_id"]
            if key in self.sources:
                raise ValueError("Duplicate source analysis identity")
            self.sources[key] = source
        self.entries = {}
        for row in self.manifest["entries"]:
            key = row["analysis_id"]
            if key in self.entries:
                raise ValueError("Duplicate fit analysis identity")
            source = self.sources[row["source_id"]]
            if source["analysis_id"] != key or source["event"] != row["event"]:
                raise ValueError("Fit/source analysis or event identity mismatch")
            if not isinstance(row["acceptance"].get("scope"), (str, dict, list)):
                raise ValueError("A fit must declare its recovery scope")
            self.entries[key] = row

    @staticmethod
    def _check_source(source):
        required = {"analysis_id", "event", "authority", "release", "analysis_label",
                    "waveform", "files", "priors", "settings", "known_issues",
                    "provenance_gaps"}
        if required - source.keys():
            raise ValueError("Missing source provenance: " + str(sorted(required - source.keys())))
        if source["authority"] not in {"LVK", "NoahManning", "NatalieMalagon"}:
            raise ValueError("Source authority is outside this collection policy")
        if not {"id", "version", "review_status", "evidence"} <= source["release"].keys():
            raise ValueError("Release/review provenance is incomplete")
        if not {"name", "version"} <= source["waveform"].keys():
            raise ValueError("Waveform identity/version must be explicit (null if unknown)")
        if not {"description", "measure", "metadata_source"} <= source["priors"].keys():
            raise ValueError("Prior and measure provenance is incomplete")
        if not source["files"]:
            raise ValueError("No exact source files recorded")
        for item in source["files"]:
            if not {"host", "path", "sha256", "role"} <= item.keys():
                raise ValueError("Source file provenance is incomplete")
            digest = item["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Source file SHA256 is malformed")
        for key in ["known_issues", "provenance_gaps"]:
            if not isinstance(source[key], list):
                raise ValueError(key + " must be a list")
        issue_states = {"not_affected", "affected_fixed", "affected_unfixed", "review_required", "unknown"}
        for issue in source["known_issues"]:
            if not isinstance(issue, dict) or not issue.get("issue_id") or issue.get("status") not in issue_states:
                raise ValueError("Known issues require an identity and explicit applicability status")

    def record(self, event=None, *, analysis_id=None):
        if analysis_id is not None:
            row = self.entries[analysis_id]
            if event is not None and event != row["event"] and event not in row.get("aliases", []):
                raise ValueError("Requested event/analysis mismatch")
            return row
        matches = [r for r in self.entries.values()
                   if event == r["event"] or event in r.get("aliases", [])]
        if len(matches) != 1:
            raise ValueError("Specify analysis_id; event selection is missing or ambiguous")
        return matches[0]

    def source(self, event=None, *, analysis_id=None):
        return self.sources[self.record(event, analysis_id=analysis_id)["source_id"]]

    def artifact(self, event=None, *, analysis_id=None):
        row = self.record(event, analysis_id=analysis_id)
        result = _verified(self.directory, row["fit"])
        fit_source = result.get("fit", {}).get("source", {}) if isinstance(result.get("fit"), dict) else {}
        metadata_source = self.sources[row["source_id"]].get("priors", {}).get("metadata_source")
        metadata = None
        if isinstance(fit_source, dict) and fit_source.get("metadata_sha256") and metadata_source:
            meta = (self.directory / metadata_source).resolve()
            if not meta.is_relative_to(self.directory) or not meta.is_file() or hashlib.sha256(meta.read_bytes()).hexdigest() != fit_source["metadata_sha256"]:
                raise ValueError("Source metadata does not match the fit's metadata_sha256: " + row["event"])
            metadata = json.loads(meta.read_text())
        require_public_fit(self.root, row["fit"]["sha256"], row["event"], artifact=result, source_metadata=metadata)
        if result.get("event") != row["event"]:
            raise ValueError("Artifact event mismatch")
        if result.get("analysis_id") != row["analysis_id"]:
            raise ValueError("Artifact analysis mismatch")
        if "coordinates" in result and tuple(result["coordinates"]) != self.coordinates:
            raise ValueError("Artifact coordinate mismatch")
        return result

    def evidence(self, event=None, *, analysis_id=None):
        row = self.record(event, analysis_id=analysis_id)
        proof = _verified(self.directory, row["validation"])
        if self.backend in NATIVE_REDSHIFT_BACKENDS and proof.get("schema") == "native-redshift-candidate-validation-v1":
            if proof.get("status") != "candidate_unvalidated":
                raise ValueError("Native redshift candidate evidence cannot grant acceptance")
            for key in ["event", "analysis_id"]:
                if proof.get(key) != row[key]:
                    raise ValueError("Validation identity mismatch: " + key)
            if proof.get("fit_sha256") != row["fit"]["sha256"] or tuple(proof.get("coordinates", [])) != self.coordinates:
                raise ValueError("Candidate validation does not bind this fit/chart")
            original = _verified(self.directory, proof["original_validation"])
            if original.get("model_sha256") != proof.get("upstream_model_sha256") or original.get("event") != row["event"]:
                raise ValueError("Upstream validation does not bind this model/event")
            _require_declared_outcome(row, proof, original)
            artifact = self.artifact(analysis_id=row["analysis_id"])
            canonical = hashlib.sha256(json.dumps(artifact["fit"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if artifact.get("upstream_model_sha256") != proof["upstream_model_sha256"] or artifact.get("model_canonical_sha256") != canonical or proof.get("model_canonical_sha256") != canonical:
                raise ValueError("Candidate evidence does not bind the embedded model")
            return proof
        if proof.get("schema") != "nal-validation-v1":
            raise ValueError("Unsupported collection validation schema")
        for key in ["event", "analysis_id"]:
            if proof.get(key) != row[key]:
                raise ValueError("Validation identity mismatch: " + key)
        if proof.get("fit_sha256") != row["fit"]["sha256"]:
            raise ValueError("Validation does not describe these fit bytes")
        if tuple(proof.get("coordinates", [])) != self.coordinates:
            raise ValueError("Validation coordinate mismatch")
        expected_pairs = {"|".join(p) for p in itertools.combinations(self.coordinates, 2)}
        cases = proof.get("cases", [])
        if not cases:
            raise ValueError("Validation has no recovery cases")
        for case in cases:
            if set(case.get("marginals_1d", {})) != set(self.coordinates):
                raise ValueError("Incomplete 1D marginal evidence")
            if set(case.get("marginals_2d", {})) != expected_pairs:
                raise ValueError("Incomplete standard 2D marginal evidence")
            if not case.get("prior") or not case.get("split"):
                raise ValueError("Validation prior/split must be explicit")
        return proof

    def fit(self, event=None, *, analysis_id=None, allow_scoped=False,
            allow_candidate=False, acknowledge_source_issues=False):
        row = self.record(event, analysis_id=analysis_id)
        status = row["acceptance"]["status"]
        if status == SCOPED and not allow_scoped:
            raise ValueError("Prior-scoped fit: explicitly acknowledge the declared scope")
        if status not in {GENERIC, SCOPED} and not allow_candidate:
            raise ValueError("Fit is not accepted for recovery: " + status)
        source = self.sources[row["source_id"]]
        flagged = [i for i in source["known_issues"] if isinstance(i, dict)
                   and i.get("status") in {"affected_unfixed", "review_required", "unknown"}]
        if flagged and not acknowledge_source_issues:
            raise ValueError("Source PE has unresolved issue provenance; acknowledge_source_issues=True is required")
        artifact = self.artifact(analysis_id=row["analysis_id"])
        if status in {GENERIC, SCOPED}:
            proof = self.evidence(analysis_id=row["analysis_id"])
            if proof.get("status") != "pass":
                raise ValueError("Accepted fit lacks passing recovery evidence")
        return artifact.get("fit", artifact)

    def log_likelihood(self, event, points, **policy):
        """Physical coordinates in manifest order; arbitrary event log constant.

        No coordinate Jacobian is introduced at this interface. Implementations
        own the coordinate/measure algebra encoded in their model artifacts.
        """
        import numpy as np
        x = np.asarray(points, dtype=float)
        if x.ndim == 0 or x.shape[-1] != len(self.coordinates):
            raise ValueError("Expected coordinates " + repr(self.coordinates))
        fit = self.fit(event, **policy)
        if self.backend == "native_posterior_redshift":
            from native_posterior_redshift import log_likelihood_x
            return log_likelihood_x(x, fit)
        if self.backend == "native_spin_base_redshift":
            from native_spin_base_redshift import log_likelihood_x
            return log_likelihood_x(x, fit)
        if self.manifest["family"] == "quasicircular":
            from bounded_fit import log_shape
            return log_shape(x, fit)
        from eccentric_likelihood import log_likelihood
        return log_likelihood(x, fit)

    def verify(self):
        """Verify every artifact and report coverage without changing acceptance."""
        counts = {}
        for key, row in self.entries.items():
            self.artifact(analysis_id=key)
            self.evidence(analysis_id=key)
            status = row["acceptance"]["status"]
            counts[status] = counts.get(status, 0) + 1
        requested = self.manifest["requested_events"]
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate requested event identity")
        observed = {r["event"] for r in self.entries.values()}
        missing = self.manifest.get("unresolved", [])
        if any(not isinstance(r.get("reason"), str) or not r["reason"].strip() for r in missing):
            raise ValueError("Unresolved coverage needs an explicit reason")
        unresolved = {r["event"] for r in missing}
        if set(requested) != observed | unresolved:
            raise ValueError("Requested-event coverage differs from fitted plus explicitly unresolved events")
        return {"collection_id": self.manifest["collection_id"],
                "analyses": len(self.entries), "events": len({r["event"] for r in self.entries.values()}),
                "acceptance_counts": counts, "unresolved": self.manifest.get("unresolved", [])}


def _require_declared_outcome(row, proof, original):
    """The manifest entry states each candidate's check outcome; its validation records must agree.

    Editions that state no outcome predate this rule: their recorded replication splits must all pass
    (the rule they were built under). The upstream verdict must match the original validation in either case."""
    passes = original.get("passes_declared_checks")
    if proof.get("upstream_passes_declared_checks", passes) != passes:
        raise ValueError("Candidate validation misstates the upstream verdict: " + row["event"])
    replication = proof.get("replication_splits", [])
    if any(r.get("passes") != (not r.get("failed", [])) for r in replication if "failed" in r):
        raise ValueError("Replication split verdict disagrees with its failed checks: " + row["event"])
    stated = row["acceptance"].get("declared_checks")
    observed = {"published_split_passes": passes is True, "failed_checks": proof.get("upstream_failed_checks", []),
                "replication": [{"split_seed": r.get("split_seed"), "passes": r.get("passes")} for r in replication]}
    if stated is None:
        if any(r.get("passes") is not True for r in replication):
            raise ValueError("Entry states no check outcome but a recorded replication split fails: " + row["event"])
    elif stated != observed:
        raise ValueError("Stated check outcome differs from the validation records: " + row["event"])


def discover(root):
    """Return explicit IDs; never select an unversioned latest collection."""
    base = Path(root).resolve() / "collections"
    return sorted(str(p.parent.relative_to(base)) for p in base.glob("*/*/manifest.json"))
