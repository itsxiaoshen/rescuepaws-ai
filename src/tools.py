"""Shelter tools the agent can call. Each tool wraps existing, tested code."""
from src.animal_search import filter_animals, get_animal_profile
from src.data_loader import load_animals
from src.matching import AnimalMatcher, HouseholdNeeds
from src.rag import PolicyRetriever, answer_policy_question, load_policy_chunks
from src.schemas import AdoptionStatus, Size, Species

# Tool definitions in the format the OpenAI API expects.
# The descriptions matter: the LLM reads them to decide which tool fits a request.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_animal_profile",
            "description": "Get the full shelter record for one animal by its ID (e.g. RP-0009).",
            "parameters": {
                "type": "object",
                "properties": {
                    "animal_id": {"type": "string", "description": "Animal ID, e.g. RP-0009"},
                },
                "required": ["animal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_animals",
            "description": (
                "Find animals by name or exact attributes. Use this to look up an animal "
                "by name (names are not unique) or to list animals matching fixed criteria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Animal name, case-insensitive"},
                    "species": {"type": "string", "enum": ["dog", "cat", "other"]},
                    "size": {"type": "string", "enum": ["small", "medium", "large"]},
                    "include_unavailable": {
                        "type": "boolean",
                        "description": "Also include adopted, on-hold, and in-medical-care animals",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_animals",
            "description": (
                "Recommend available animals for an adopter based on a free-text description of "
                "what they want plus facts about their household. Returns reasons and unknowns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "What the adopter is looking for, in their words"},
                    "species": {"type": "string", "enum": ["dog", "cat", "other"]},
                    "size": {"type": "string", "enum": ["small", "medium", "large"]},
                    "has_children": {"type": "boolean", "description": "Children live in the home"},
                    "has_dogs": {"type": "boolean", "description": "Another dog lives in the home"},
                    "has_cats": {"type": "boolean", "description": "A cat lives in the home"},
                },
                "required": ["request"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_policy_question",
            "description": (
                "Answer a question about shelter policy (fees, adoption process, eligibility, "
                "returns, transport, vaccination policy, opening hours) from official documents. "
                "Policies are general, not per animal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A general policy question. Don't include animal names or IDs; include the "
                            "facts that matter instead, e.g. 'adoption fee for a 3-year-old dog'."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    },
]


def animal_summary(animal) -> dict:
    return {
        "animal_id": animal.animal_id,
        "name": animal.name,
        "species": animal.species.value,
        "size": animal.size.value,
        "estimated_age_months": animal.estimated_age_months,
        "status": animal.status.value,
    }


class ShelterTools:
    """Holds the shared data and models, and runs tools by name."""

    def __init__(self, animals=None, matcher=None, retriever=None, generate=None):
        self.animals = animals if animals is not None else load_animals()
        self.matcher = matcher or AnimalMatcher(self.animals)
        self.retriever = retriever or PolicyRetriever(load_policy_chunks())
        self.generate = generate  # optional fake LLM for policy answers in tests

    def get_animal_profile(self, animal_id: str) -> dict:
        animal = get_animal_profile(animal_id.strip().upper(), self.animals)
        if animal is None:
            return {"error": f"No animal with ID {animal_id} in shelter records."}
        profile = animal.model_dump(mode="json")
        profile["record_conflicts"] = self.matcher.conflicts.get(animal.animal_id, [])
        return profile

    def search_animals(self, name=None, species=None, size=None, include_unavailable=False) -> dict:
        results = filter_animals(
            self.animals,
            species=Species(species) if species else None,
            size=Size(size) if size else None,
            status=None if include_unavailable else AdoptionStatus.AVAILABLE,
        )
        if name:
            results = [a for a in results if a.name.lower() == name.strip().lower()]
        return {"count": len(results), "animals": [animal_summary(a) for a in results[:20]]}

    def match_animals(self, request, species=None, size=None,
                      has_children=False, has_dogs=False, has_cats=False) -> dict:
        needs = HouseholdNeeds(
            species=Species(species) if species else None,
            size=Size(size) if size else None,
            has_children=has_children,
            has_dogs=has_dogs,
            has_cats=has_cats,
        )
        results = self.matcher.match(request, needs, top_k=5)
        return {
            "matches": [
                {**animal_summary(r.animal), "reasons": r.reasons, "unknowns": r.unknowns}
                for r in results
            ]
        }

    def answer_policy_question(self, question: str) -> dict:
        kwargs = {"generate": self.generate} if self.generate else {}
        result = answer_policy_question(question, self.retriever, **kwargs)
        return {
            "found_in_policy": result.answerable,
            "answer": result.answer,
            "sources": [f"{c.source} > {c.section}" for c in result.citations],
        }

    def execute(self, name: str, arguments: dict) -> dict:
        """Run a tool by name. Errors are returned to the LLM instead of crashing the agent."""
        tool = {
            "get_animal_profile": self.get_animal_profile,
            "search_animals": self.search_animals,
            "match_animals": self.match_animals,
            "answer_policy_question": self.answer_policy_question,
        }.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool(**arguments)
        except (TypeError, ValueError) as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
