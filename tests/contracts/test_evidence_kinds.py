import copy
import pytest
from videntra.contracts.evidence import validate_evidence, COMMON
from videntra.cli import validate_manifests
from test_contracts import evidence


@pytest.mark.parametrize("kind,payload", [
    ("FrameObservation", dict(frame_id="f", original_decoded_frame_index=0, timestamp_us=0,
                              dimensions={"width": 64, "height": 48}, fingerprint="a"*64)),
    ("OCRTrack", dict(observation_ids=["obs"], first_observed_us=0, last_observed_us=0,
                      association_method="singleton", association_scores=[], holes=[],
                      persistence_kind="instant", boundary_uncertainty_us=0)),
    ("ASRSegment", dict(start_us=0, end_us=100, text_original="hi", text_nfc="hi", language="en",
                        chunk_id="c", audio_sample_start=0, audio_sample_end=10, timestamp_method="test",
                        word_records=[{"text":"hi","start_us":None,"end_us":None}], speaker_cluster=None)),
    ("EntityTrack", dict(track_id="t", entity_type="person", frame_boxes=[], attribute_observations=[],
                         track_links=[], identity_confidence=None)),
    ("EventHypothesis", dict(event_id="e", predicate="give", roles={"subject":"t"}, start_us=0, end_us=100,
                             support_refs=["f"], assessment_state="unknown", method="test",
                             boundary_uncertainty={})),
    ("RelationAssessment", dict(query_predicate_id="q", event_or_track_refs=["e"], state="unknown",
                                features={}, reason_codes=["missing"], support_refs=[], contradict_refs=[])),
    ("SpeakerBinding", dict(speaker_cluster="c", actor_track_id="t", start_us=0, end_us=100,
                             audio_visual_model_revision="test", support_frame_refs=["f"], sync_error_us=0,
                             state="supported")),
    ("CandidateMoment", dict(candidate_id="c", seed_branch_ids=["visual"], start_us=0, end_us=100,
                              proposal_method="test", seed_evidence_ids=["obs"], original_query_rank=None)),
])
def test_archival_kinds_and_missing_payload(kind, payload):
    record = {key:value for key,value in evidence().items() if key in COMMON}
    record.update(kind=kind, **payload)
    validate_evidence(record, known_refs={"f","t","e","obs"})
    broken = copy.deepcopy(record)
    del broken[next(iter(payload))]
    with pytest.raises(ValueError):
        validate_evidence(broken, known_refs={"f","t","e","obs"})


def test_singleton_cannot_claim_persistence():
    record = {key:value for key,value in evidence().items() if key in COMMON}
    record.update(kind="OCRTrack", observation_ids=["obs"], first_observed_us=0, last_observed_us=100,
                  association_method="test", association_scores=[], holes=[], persistence_kind="tracked_interval",
                  boundary_uncertainty_us=0)
    with pytest.raises(ValueError):
        validate_evidence(record, known_refs={"obs"})

