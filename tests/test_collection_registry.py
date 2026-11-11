"""Scientific provenance regressions; no waveform or fitting work in these tests."""
import copy
import hashlib
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collection_registry import Collection
from public_fixture_certificate import install


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = "quasicircular/test-v1"
        self.base = self.root / "collections" / self.identity
        self.base.mkdir(parents=True)
        self.coords = ["mc_source", "eta", "chi_eff"]
        source = dict(analysis_id="analysis-1", event="event-1", authority="LVK",
                      release=dict(id="GWTC-test", version="v1", review_status="reviewed", evidence=[]),
                      analysis_label="NRSur", waveform=dict(name="NRSur", version=None),
                      files=[dict(host="CIT", path="/source/file.h5", role="posterior", sha256="a" * 64)],
                      priors=dict(description="native", measure="declared", metadata_source="config"),
                      settings={}, known_issues=[], provenance_gaps=["waveform version unknown"])
        self.sources = {"schema": "nal-sources-v1", "analyses": [source]}
        self.fit = {"event": "event-1", "analysis_id": "analysis-1", "fit": {"schema": "fixture"}}
        fit_ref = self.write("fit.json", self.fit)
        self.proof = dict(schema="nal-validation-v1", event="event-1", analysis_id="analysis-1",
                          fit_sha256=fit_ref["sha256"], coordinates=self.coords, status="pass", scope="test",
                          cases=[dict(prior="native", split="heldout_group",
                                      marginals_1d={k: {"js_nats": 0.001} for k in self.coords},
                                      marginals_2d={"|".join(k): {"js_nats": 0.002}
                                                    for k in itertools.combinations(self.coords, 2)})])
        self.manifest = dict(schema="nal-collection-v1", collection_id=self.identity,
                             family="quasicircular", coordinates=self.coords,
                             requested_events=["event-1"], unresolved=[],
                             source_manifest=self.write("sources.json", self.sources),
                             entries=[dict(event="event-1", analysis_id="analysis-1", source_id="analysis-1",
                                           fit=fit_ref, validation=self.write("proof.json", self.proof),
                                           acceptance=dict(status="accepted_for_declared_target", scope="test"))])
        install(self.root,fit_ref["sha256"],"event-1")
        self.save()

    def write(self, name, data):
        p = self.base / name
        p.write_text(json.dumps(data))
        return {"path": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def save(self):
        (self.base / "manifest.json").write_text(json.dumps(self.manifest))

    def load(self):
        return Collection(self.root, self.identity)

    def test_complete_identity_and_pairs(self):
        self.assertEqual(self.load().verify()["events"], 1)

    def test_source_hash_cannot_be_replaced(self):
        (self.base / "sources.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load()

    def test_fit_hash_cannot_be_replaced(self):
        (self.base / "fit.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load().artifact("event-1")

    def test_missing_pair_is_not_complete(self):
        self.proof["cases"][0]["marginals_2d"].pop("mc_source|eta")
        self.manifest["entries"][0]["validation"] = self.write("proof.json", self.proof)
        self.save()
        with self.assertRaisesRegex(ValueError, "2D"):
            self.load().verify()

    def test_validation_binds_fit_bytes(self):
        self.proof["fit_sha256"] = "b" * 64
        self.manifest["entries"][0]["validation"] = self.write("proof.json", self.proof)
        self.save()
        with self.assertRaisesRegex(ValueError, "fit bytes"):
            self.load().evidence("event-1")

    def test_no_silent_event_loss(self):
        self.manifest["requested_events"].append("event-2")
        self.save()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.load().verify()

    def test_explicit_unresolved_event_is_retained(self):
        self.manifest["requested_events"].append("event-2")
        self.manifest["unresolved"] = [{"event": "event-2", "reason": "missing authorized source"}]
        self.save()
        self.assertEqual(self.load().verify()["unresolved"][0]["event"], "event-2")

    def test_ambiguous_event_requires_analysis_identity(self):
        row = copy.deepcopy(self.manifest["entries"][0])
        row.update(analysis_id="analysis-2", source_id="analysis-2")
        src = copy.deepcopy(self.sources["analyses"][0])
        src["analysis_id"] = "analysis-2"
        self.sources["analyses"].append(src)
        self.manifest["source_manifest"] = self.write("sources.json", self.sources)
        self.manifest["entries"].append(row)
        self.save()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.load().record("event-1")

    def test_scope_is_explicit(self):
        self.manifest["entries"][0]["acceptance"]["status"] = "accepted_for_declared_forward_priors_only"
        self.save()
        with self.assertRaisesRegex(ValueError, "Prior-scoped"):
            self.load().fit("event-1")
        self.assertEqual(self.load().fit("event-1", allow_scoped=True)["schema"], "fixture")

    def test_source_quality_is_separate_from_recovery(self):
        self.sources["analyses"][0]["known_issues"] = [{"issue_id": "window", "status": "review_required"}]
        self.manifest["source_manifest"] = self.write("sources.json", self.sources)
        self.save()
        with self.assertRaisesRegex(ValueError, "Source PE"):
            self.load().fit("event-1")
        self.assertEqual(self.load().fit("event-1", acknowledge_source_issues=True)["schema"], "fixture")

    def test_artifact_paths_are_contained(self):
        self.manifest["entries"][0]["fit"]["path"] = "../outside.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "contained"):
            self.load().artifact("event-1")

    def test_unresolved_reason_is_required(self):
        self.manifest["requested_events"].append("event-2")
        self.manifest["unresolved"] = [{"event": "event-2", "reason": ""}]
        self.save()
        with self.assertRaisesRegex(ValueError, "explicit reason"):
            self.load().verify()

    def test_issue_applicability_cannot_be_implicit(self):
        self.sources["analyses"][0]["known_issues"] = ["windowing"]
        self.manifest["source_manifest"] = self.write("sources.json", self.sources)
        self.save()
        with self.assertRaisesRegex(ValueError, "applicability"):
            self.load()

    def test_coordinate_permutation_cannot_change_dispatch(self):
        self.manifest["coordinates"] = ["eta", "mc_source", "chi_eff"]
        self.save()
        with self.assertRaisesRegex(ValueError, "coordinate order"):
            self.load()

    def test_artifact_analysis_identity_cannot_be_implicit(self):
        self.fit.pop("analysis_id")
        self.manifest["entries"][0]["fit"] = self.write("fit.json", self.fit)
        install(self.root,self.manifest["entries"][0]["fit"]["sha256"],"event-1")
        self.save()
        with self.assertRaisesRegex(ValueError, "analysis mismatch"):
            self.load().artifact("event-1")

    def test_verified_aliases_preserve_explicit_analysis_selection(self):
        self.manifest["entries"][0]["aliases"] = ["alias-1"]
        self.save()
        self.assertEqual(self.load().record("alias-1")["event"], "event-1")
        self.assertEqual(self.load().record("alias-1", analysis_id="analysis-1")["event"], "event-1")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.load().record("another-event", analysis_id="analysis-1")


if __name__ == "__main__":
    unittest.main()
