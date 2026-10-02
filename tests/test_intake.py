"""Intake tests with fake LLM outputs: no API calls."""
from pathlib import Path

from src.intake import (
    FieldEvidence,
    NotesExtraction,
    PhotoObservation,
    build_draft,
    create_intake_draft,
    next_animal_id,
)
from src.schemas import AdoptionStatus, TriState


def extraction(**overrides) -> NotesExtraction:
    """A notes extraction where everything is unknown, unless overridden."""
    values = dict(
        name="Test", species="dog", sex="unknown", estimated_age_months=None, size="unknown",
        vaccination_status="unknown", sterilized="unknown", medical_notes=None, behavior_notes=None,
        good_with_dogs="unknown", good_with_cats="unknown", good_with_children="unknown", evidence=[],
    )
    values.update(overrides)
    return NotesExtraction(**values)


def photo(**overrides) -> PhotoObservation:
    values = dict(species="dog", coat_colors=["black"], coat_pattern="solid",
                  size_impression="unclear", photo_issue=None)
    values.update(overrides)
    return PhotoObservation(**values)


def test_photo_alone_never_sets_temperament_or_compatibility():
    draft = build_draft("RP-0100", "Name: Test.", photo(size_impression="large"), extraction())
    p = draft.profile
    assert p.good_with_children == p.good_with_dogs == p.good_with_cats == TriState.UNKNOWN
    assert p.behavior_notes is None and p.estimated_age_months is None
    assert p.appearance == "black, solid"
    assert p.evidence["appearance"] == "Photo"


def test_compatibility_without_quote_is_reset_to_unknown():
    draft = build_draft("RP-0100", "Seems sweet.", None, extraction(good_with_children="yes"))
    assert draft.profile.good_with_children == TriState.UNKNOWN
    assert any("good_with_children" in flag for flag in draft.review_flags)


def test_quote_not_in_notes_is_ignored():
    fake_quote = FieldEvidence(field="good_with_cats", quote="Loves living with cats")
    draft = build_draft("RP-0100", "Friendly dog.", None,
                        extraction(good_with_cats="yes", evidence=[fake_quote]))
    assert draft.profile.good_with_cats == TriState.UNKNOWN
    assert any("quote not found" in flag for flag in draft.review_flags)


def test_supported_compatibility_is_kept_with_evidence():
    notes = "Plays gently with the foster family's 4-year-old twins every day."
    quote = FieldEvidence(field="good_with_children", quote="Plays gently with the foster family's 4-year-old twins")
    draft = build_draft("RP-0100", notes, None, extraction(good_with_children="yes", evidence=[quote]))
    assert draft.profile.good_with_children == TriState.YES
    assert "twins" in draft.profile.evidence["good_with_children"]


def test_species_conflict_is_flagged():
    draft = build_draft("RP-0100", "A cat.", photo(species="dog"), extraction(species="cat"))
    assert draft.profile.species.value == "cat"  # notes win
    assert any("Species conflict" in flag for flag in draft.review_flags)


def test_size_from_photo_is_flagged_for_review():
    draft = build_draft("RP-0100", "Name: Test.", photo(size_impression="large"), extraction())
    assert draft.profile.size.value == "large"
    assert any("verify" in flag for flag in draft.review_flags)


def test_new_intake_is_not_adoptable_yet():
    draft = build_draft("RP-0100", "Name: Test.", None, extraction())
    assert draft.profile.status == AdoptionStatus.ON_HOLD


def test_image_is_only_sent_to_the_photo_call():
    calls = []

    def fake_generate(system, user, output_type, image_path=None):
        calls.append((output_type, image_path))
        return photo() if output_type is PhotoObservation else extraction()

    create_intake_draft("RP-0100", "Name: Test.", Path("dog.jpg"), generate=fake_generate)
    assert calls == [(PhotoObservation, Path("dog.jpg")), (NotesExtraction, None)]


def test_next_animal_id():
    assert next_animal_id(["RP-0001", "RP-0045", "RP-0007"]) == "RP-0046"
    assert next_animal_id([]) == "RP-0001"
