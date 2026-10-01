"""Semantic adoption matching: adopter request -> ranked candidate animals."""
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from src.schemas import AdoptionStatus, AnimalProfile, Size, Species, TriState

MODEL_NAME = "all-MiniLM-L6-v2"

# (household attribute, animal attribute, label used in explanations)
COMPATIBILITY_CHECKS = [
    ("has_children", "good_with_children", "children"),
    ("has_dogs", "good_with_dogs", "dogs"),
    ("has_cats", "good_with_cats", "cats"),
]


@dataclass
class HouseholdNeeds:
    """Structured facts about the adopter's household. Used as hard constraints."""
    species: Species | None = None
    size: Size | None = None
    has_children: bool = False
    has_dogs: bool = False
    has_cats: bool = False


@dataclass
class MatchResult:
    animal: AnimalProfile
    score: float
    reasons: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)


def age_group(months: int | None) -> str:
    if months is None:
        return "age not recorded"
    if months < 12:
        return "young (under 1 year)"
    if months < 84:
        return f"adult (about {months // 12} years)"
    return f"senior (about {months // 12} years)"


def animal_to_text(animal: AnimalProfile) -> str:
    """Describe an animal for embedding, using only recorded information.

    Compatibility (dogs/cats/children) is deliberately left out: it is handled by
    structured checks, because embeddings cannot reliably tell "yes" from "unknown".
    """
    size = "size not recorded" if animal.size == Size.UNKNOWN else f"{animal.size.value} size"
    behavior = animal.behavior_notes or "Behavior not recorded."
    return f"{animal.species.value}, {size}, {age_group(animal.estimated_age_months)}. {behavior}"


class AnimalMatcher:
    def __init__(self, animals: list[AnimalProfile], model_name: str = MODEL_NAME):
        self.animals = animals
        self.model = SentenceTransformer(model_name)
        texts = [animal_to_text(animal) for animal in animals]
        # normalize_embeddings=True makes every vector length 1, so dot product == cosine similarity
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def similarities(self, request: str) -> np.ndarray:
        """Cosine similarity between the request and every animal, in dataset order."""
        query = self.model.encode([request], normalize_embeddings=True)[0]
        return self.embeddings @ query

    def search(self, request: str, top_k: int = 5) -> list[tuple[AnimalProfile, float]]:
        """Pure semantic search, no constraints. Useful for comparison and debugging."""
        scores = self.similarities(request)
        best = np.argsort(-scores)[:top_k]
        return [(self.animals[i], float(scores[i])) for i in best]

    def match(
        self, request: str, needs: HouseholdNeeds | None = None, top_k: int = 5
    ) -> list[MatchResult]:
        """Semantic ranking + structured safety constraints, with reasons and unknowns."""
        needs = needs or HouseholdNeeds()
        scores = self.similarities(request)

        results = []
        for animal, score in zip(self.animals, scores):
            if animal.status != AdoptionStatus.AVAILABLE:
                continue
            if needs.species is not None and animal.species != needs.species:
                continue
            if needs.size is not None and animal.size != needs.size:
                continue

            result = MatchResult(animal=animal, score=float(score))
            excluded = False
            for need_attr, animal_attr, label in COMPATIBILITY_CHECKS:
                if not getattr(needs, need_attr):
                    continue
                value = getattr(animal, animal_attr)
                if value == TriState.NO:
                    excluded = True
                    break
                if value == TriState.YES:
                    source = animal.evidence.get(animal_attr, "source not recorded")
                    result.reasons.append(f"Recorded as good with {label}: {source}")
                else:
                    result.unknowns.append(f"Compatibility with {label} has not been tested")
            if excluded:
                continue

            result.reasons.append(f"Description similarity to request: {score:.2f}")
            results.append(result)

        # Fully confirmed matches first, then matches with unknowns; each group by similarity
        results.sort(key=lambda r: (len(r.unknowns) > 0, -r.score))
        return results[:top_k]
