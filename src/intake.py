"""Intake: photo + volunteer notes -> draft AnimalProfile for a volunteer to review.

Safety design: the photo and the notes are processed by two separate LLM calls with
separate output schemas. The photo schema has no fields for temperament, health, age,
or compatibility, so those can never come from appearance.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.llm import generate_structured
from src.schemas import AdoptionStatus, AnimalProfile

TriStateValue = Literal["yes", "no", "unknown"]

COMPATIBILITY_FIELDS = ["good_with_dogs", "good_with_cats", "good_with_children"]


# ---------------------------------------------------------------------------
# What the LLM is allowed to return
# ---------------------------------------------------------------------------

class PhotoObservation(BaseModel):
    """Only what can be seen. By design there are no temperament, health, age, or breed fields."""
    species: Literal["dog", "cat", "other", "unclear"]
    coat_colors: list[str]
    coat_pattern: str | None
    size_impression: Literal["small", "medium", "large", "unclear"]
    photo_issue: str | None  # e.g. "animal partly hidden", "blurry"


class FieldEvidence(BaseModel):
    field: str
    quote: str  # copied word for word from the volunteer notes


class NotesExtraction(BaseModel):
    name: str | None
    species: Literal["dog", "cat", "other"] | None
    sex: Literal["male", "female", "unknown"]
    estimated_age_months: int | None
    size: Literal["small", "medium", "large", "unknown"]
    vaccination_status: Literal["up_to_date", "partial", "none", "unknown"]
    sterilized: TriStateValue
    medical_notes: str | None
    behavior_notes: str | None
    good_with_dogs: TriStateValue
    good_with_cats: TriStateValue
    good_with_children: TriStateValue
    evidence: list[FieldEvidence]


PHOTO_SYSTEM_PROMPT = """You describe an animal shelter intake photo. Report ONLY what is directly visible.

- species: "unclear" if you cannot tell.
- coat_colors and coat_pattern (e.g. "tabby", "solid", "bicolor", "spotted").
- size_impression: only if something in the photo gives a sense of scale; otherwise "unclear".
- photo_issue: note problems like blur, the animal being partly hidden, or several animals.

Never describe mood, temperament, health, age, or breed. A photo cannot show those reliably."""

NOTES_SYSTEM_PROMPT = """You turn a shelter volunteer's intake notes into structured fields.

Rules:
- Use ONLY what the notes state. Anything not stated is "unknown" (or null).
- species: null unless the notes say what kind of animal it is (words like dog, puppy, terrier, cat, kitten, rabbit).
- good_with_dogs / good_with_cats / good_with_children: "yes" or "no" ONLY when the notes describe an actual interaction with that kind of animal or person. General impressions like "sweet", "friendly", or "gentle" are NOT evidence of compatibility: use "unknown".
- Never infer anything from breed, appearance, or species.
- estimated_age_months: convert stated ages to months ("about 2 years" -> 24, "10 weeks" -> 2). null if no age is stated.
- vaccination_status: "up_to_date" only if the notes say vaccines are current or complete; "partial" if some are given and others are due.
- medical_notes / behavior_notes: short summaries of what the notes say, or null.
- evidence: for EVERY field you set to something other than unknown/null (except name and species), add an entry with a quote copied word for word from the notes."""


# ---------------------------------------------------------------------------
# Combining photo + notes into a draft
# ---------------------------------------------------------------------------

@dataclass
class IntakeDraft:
    profile: AnimalProfile
    review_flags: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def build_draft(
    animal_id: str,
    notes: str,
    photo: PhotoObservation | None,
    extracted: NotesExtraction,
) -> IntakeDraft:
    """Merge photo observations and note extraction into a validated draft profile."""
    flags: list[str] = []
    evidence: dict[str, str] = {}
    notes_normalized = normalize_text(notes)

    # Keep only evidence quotes that really appear in the notes
    quotes: dict[str, str] = {}
    for item in extracted.evidence:
        if normalize_text(item.quote) in notes_normalized:
            quotes[item.field] = item.quote
        else:
            flags.append(f"Ignored evidence for {item.field}: quote not found in notes")

    values = extracted.model_dump(exclude={"evidence"})

    # Grounding guard: compatibility and medical/vaccination fields need a real quote
    for name in COMPATIBILITY_FIELDS + ["vaccination_status", "sterilized"]:
        if values[name] != "unknown" and name not in quotes:
            flags.append(f"{name} was '{values[name]}' without a quote from the notes; reset to unknown")
            values[name] = "unknown"
    for name, quote in quotes.items():
        if name in AnimalProfile.model_fields:
            evidence[name] = f'Volunteer notes: "{quote}"'

    # Species: notes first, then photo
    species = values["species"]
    if photo and photo.species != "unclear":
        if species is None:
            species = photo.species
            evidence["species"] = "Photo"
        elif species != photo.species:
            flags.append(f"Species conflict: notes say {species}, photo looks like {photo.species}")
    if species is None:
        species = "other"
        flags.append("Species not stated in notes or visible in photo; set to 'other'")

    # Size: notes first, then a visible size impression (needs checking)
    size = values["size"]
    if size == "unknown" and photo and photo.size_impression != "unclear":
        size = photo.size_impression
        evidence["size"] = "Photo (size impression)"
        flags.append(f"Size '{size}' comes from the photo only; please verify")

    appearance = None
    if photo and photo.coat_colors:
        appearance = " and ".join(photo.coat_colors)
        if photo.coat_pattern:
            appearance += f", {photo.coat_pattern}"
        evidence["appearance"] = "Photo"
    if photo and photo.photo_issue:
        flags.append(f"Photo issue: {photo.photo_issue}")

    profile = AnimalProfile(
        animal_id=animal_id,
        name=values["name"] or "Unnamed",
        species=species,
        sex=values["sex"],
        estimated_age_months=values["estimated_age_months"],
        size=size,
        vaccination_status=values["vaccination_status"],
        sterilized=values["sterilized"],
        medical_notes=values["medical_notes"],
        behavior_notes=values["behavior_notes"],
        good_with_dogs=values["good_with_dogs"],
        good_with_cats=values["good_with_cats"],
        good_with_children=values["good_with_children"],
        appearance=appearance,
        location="Intake",
        status=AdoptionStatus.ON_HOLD,  # not adoptable until a volunteer approves it
        evidence=evidence,
    )
    if not values["name"]:
        flags.append("No name in notes")
    return IntakeDraft(profile=profile, review_flags=flags)


def create_intake_draft(
    animal_id: str,
    notes: str,
    photo_path: Path | None = None,
    generate=generate_structured,
) -> IntakeDraft:
    """Run the two LLM calls (photo, notes) and merge them into a draft."""
    photo = None
    if photo_path is not None:
        photo = generate(PHOTO_SYSTEM_PROMPT, "Describe this intake photo.", PhotoObservation,
                         image_path=photo_path)
    extracted = generate(NOTES_SYSTEM_PROMPT, f"Volunteer notes:\n{notes}", NotesExtraction)
    return build_draft(animal_id, notes, photo, extracted)


def next_animal_id(existing_ids: list[str]) -> str:
    numbers = [int(animal_id.split("-")[1]) for animal_id in existing_ids]
    return f"RP-{max(numbers, default=0) + 1:04d}"
