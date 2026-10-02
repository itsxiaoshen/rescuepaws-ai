"""LLM-as-judge: is every factual claim in an answer supported by the evidence?"""
from pydantic import BaseModel

from src.llm import generate_structured


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    explanation: str


class GroundednessJudgment(BaseModel):
    claims: list[ClaimCheck]

    @property
    def grounded(self) -> bool:
        return all(c.supported for c in self.claims)

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


JUDGE_SYSTEM_PROMPT = """You check whether an answer is fully supported by the policy excerpts provided.

Split the answer into its individual factual claims, read in the context of the question. A bare "Yes" or "No" is a claim about the question (e.g. "No" to "Can I pay in cash?" means "cash is not accepted"). For each claim, decide whether the excerpts directly state it or clearly imply it.
- Applying a stated rule to the question is supported (e.g. "weekday visits by appointment only" supports "No, not without an appointment on Tuesday").
- A claim that adds details, numbers, conditions, or options that are not in the excerpts is NOT supported, even if it sounds reasonable.
- A claim that changes a number, time period, or condition from the excerpts is NOT supported.
- A claim that something is excluded or not allowed is NOT supported if the excerpts simply don't mention it.
- Friendly phrases with no factual content (e.g. "Happy to help!") are not claims; skip them."""


def judge_groundedness(question: str, answer: str, evidence: str, generate=generate_structured) -> GroundednessJudgment:
    user = f"Policy excerpts:\n{evidence}\n\nQuestion: {question}\n\nAnswer to check:\n{answer}"
    return generate(JUDGE_SYSTEM_PROMPT, user, GroundednessJudgment)
