"""Data models for RescuePaws AI."""
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Species(str, Enum):
    DOG = "dog"
    CAT = "cat"
    OTHER = "other"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class Size(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    UNKNOWN = "unknown"


class TriState(str, Enum):
    """Yes / No / Unknown. Unknown is NOT the same as No, and NOT the same as Yes."""
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class VaccinationStatus(str, Enum):
    UP_TO_DATE = "up_to_date"
    PARTIAL = "partial"
    NONE = "none"
    UNKNOWN = "unknown"


class AdoptionStatus(str, Enum):
    AVAILABLE = "available"
    ON_HOLD = "on_hold"
    MEDICAL_CARE = "medical_care"
    ADOPTED = "adopted"


class AnimalProfile(BaseModel):
    # extra="forbid": a typo like "good_with_kid" raises an error instead of being silently ignored
    model_config = ConfigDict(extra="forbid")

    animal_id: str = Field(pattern=r"^RP-\d{4}$", description="e.g. RP-0001")
    name: str = Field(min_length=1)
    species: Species
    sex: Sex = Sex.UNKNOWN
    estimated_age_months: int | None = Field(default=None, ge=0, le=300)
    size: Size = Size.UNKNOWN

    vaccination_status: VaccinationStatus = VaccinationStatus.UNKNOWN
    sterilized: TriState = TriState.UNKNOWN
    medical_notes: str | None = None      # None means "not recorded"
    behavior_notes: str | None = None

    good_with_dogs: TriState = TriState.UNKNOWN
    good_with_cats: TriState = TriState.UNKNOWN
    good_with_children: TriState = TriState.UNKNOWN

    appearance: str | None = None         # observable only, e.g. "orange tabby"
    location: str | None = None
    status: AdoptionStatus = AdoptionStatus.AVAILABLE

    # field name -> where this information came from
    evidence: dict[str, str] = Field(default_factory=dict)
    @field_validator("evidence")
    @classmethod
    def evidence_keys_must_be_fields(cls, value: dict[str, str]) -> dict[str, str]:
        unknown_keys = set(value) - set(cls.model_fields)
        if unknown_keys:
            raise ValueError(f"evidence refers to unknown fields: {sorted(unknown_keys)}")
        return value
