# Evaluation Results

## Matching (Phase 2)

### Run 1 — 2026-10-01: hybrid matching v1
- Model: all-MiniLM-L6-v2 (local), top_k = 5
- Dataset: 45 animals (32 available), 17 hand-labeled cases (16 with acceptable
  candidates, 1 no-match case)

| Metric                          | Baseline (semantic only) | Hybrid |
|---------------------------------|--------------------------|--------|
| Hit@5                           | 1.00                     | 1.00   |
| MRR@5                           | 0.57                     | 0.94   |
| Precision@5                     | 0.33                     | 0.44   |
| No-match cases passed           | 0/1                      | 1/1    |
| Cases with constraint violation | 16/17                    | 0/17   |
| Cases with unsafe animal        | 1                        | 2      |

**Observations**
- Structured constraints removed all constraint violations (16 -> 0).
- Hit@5 is saturated on this small dataset, so MRR and Precision@5 are more informative.
- Precision@5 is capped below 1.0 for cases with fewer than 5 acceptable animals.
- Known issue: Otis (RP-0019) is returned for families with young children
  (M01, M15). Matching ignores the toddler incident in behavior_notes. Fix planned for Phase 6.
- M02: Otis ranked #1 for a running companion. The small embedding model
  doesn't capture energy level well.

## Policy Retrieval (Phase 3)

### Run 1: section-based chunking v1
- Model: all-MiniLM-L6-v2 (local), top_k = 3
- Corpus: 7 synthetic policy documents, 34 chunks (one per section)
- Questions: 24 answerable, 6 unanswerable (phrased differently from the documents)

| Metric | Value |
|--------|-------|
| Hit@1  | 0.71 (17/24) |
| Hit@3  | 0.92 (22/24) |
| MRR@3  | 0.80 |

**Observations**
- Misses: R06 ("can no longer keep him" vs "returns") and R07 ("give to neighbor"
  vs "rehoming"). Vocabulary mismatch; candidates for hybrid BM25 + embedding search.
- Top-1 similarity cannot detect unanswerable questions. Answerable min = 0.34,
  unanswerable max = 0.65; the distributions overlap. U02 (pet insurance) scored
  0.65 because it is on-topic with "What the fee covers", which never mentions
  insurance. Decision: abstention is handled by the LLM checking the evidence,
  not by a score threshold.

## Grounded Answers (Phase 3)

- Model: gpt-5.4-mini, top_k = 3, same 30 questions as retrieval eval
- Grounding guard: answers without a valid citation (a chunk that was actually
  retrieved) are converted to abstentions in code.

| Metric                            | Run 1 (prompt v1) | Run 2 (prompt v2) |
|-----------------------------------|-------------------|-------------------|
| Answerable questions answered     | 19/24             | 22/24             |
| Citation correct (when answered)  | 19/19             | 22/22             |
| Unanswerable questions abstained  | 6/6               | 6/6               |
| Hallucinated answers              | 0                 | 0                 |

**What changed:** v1 over-abstained on questions that only needed a stated rule
applied to the user's situation (e.g. "Can I pay in cash?" vs "We do not accept cash").
v2 adds one rule with an example. Abstention on unanswerable questions did not regress.

**Observations**
- Remaining misses (R06, R07) are retrieval failures. The LLM abstained instead of
  guessing, so the system fails safe. The next improvement should target retrieval.
- Results vary slightly between runs (e.g. R23), so a single run is not a stable estimate.
- Model comparison during setup: gpt-4.1-mini answered "No, pet insurance is not
  included" from a fee schedule that never mentions insurance (an ungrounded claim);
  gpt-5.4-mini abstained.
## Agent (Phase 4)

- Model: gpt-5.4-mini; manual agent loop (no framework), max 6 steps per turn
- Tools: get_animal_profile, search_animals, match_animals, answer_policy_question
- 15 cases / 17 turns: single-tool routing, argument extraction, duplicate names,
  missing animal, medical question, small talk, unknown fields, conflicting record,
  and one 3-turn flow (match -> inspect -> policy)

| Metric                     | Result |
|----------------------------|--------|
| Tool selection accuracy    | 17/17  |
| Argument extraction        | 4/4    |
| Turns passing all checks   | 17/17  |
| Cases completed end-to-end | 15/15  |

**Iterations during development**
- The agent asked the policy tool "What is the adoption fee for RP-0001 Mochi?", which
  retrieval can't answer (policies are per category, not per animal). Fixed by
  describing in the tool's parameter how to phrase the question ("adoption fee for
  a 3-year-old dog"). The fee was then answered correctly ($250).
- Evaluator bug: the reply "I couldn’t find..." used a curly apostrophe and failed
  a keyword check. Fixed by normalizing quotes before comparing.

**Limitations**
- Small, self-written case set; keyword checks are coarse; single run (LLM output varies).
- Known issue still open: match results list Otis (RP-0019) as good with children
  without mentioning the toddler incident. The agent surfaces the conflict only when
  the user asks about Otis directly.
## Multimodal Intake (Phase 5)

- Model: gpt-5.4-mini (vision for the photo, text-only for the notes)
- Design: two separate LLM calls. The photo schema has no temperament, health, age,
  breed, or compatibility fields; the notes call never sees the photo.
- Code guard: compatibility, vaccination, and sterilization values need a quote that
  actually appears in the notes, or they are reset to unknown and flagged for review.
- 6 human-verified cases (3 with photos), 47 checked fields

| Metric                            | Run 1 | Run 2 (prompt fix) | Run 3 |
|-----------------------------------|-------|--------------------|-------|
| Field accuracy                    | 46/47 | 47/47              | 47/47 |
| Unsupported compatibility claims  | 0     | 0                  | 0     |

**What changed:** Run 1 set species to "other" for notes that never named the species,
which overrode the correct "dog" from the photo. Added a rule: species is null unless
the notes say what kind of animal it is.

**Safety traps that passed:** a sleeping dog described as "seems sweet" (compatibility
stayed unknown); a dog with a grey muzzle and no stated age (age stayed null);
"very friendly with everyone" (compatibility stayed unknown).
## Reliability (Phase 6, Part A)

### Fix: conflicting records (known issue since Phase 2)
Offline record audit (`src/record_audit.py`) writes `data/record_conflicts.json`.
Matching treats a "yes" compatibility value with a recorded conflict as unknown,
with a "conflicting records, ask staff" note.

Audit design iterations (45 animals; Otis RP-0019 is the only true conflict):

| Version | Change                                                  | Result                        |
|---------|---------------------------------------------------------|-------------------------------|
| v1      | One open question: "find contradictions"                | 7 flagged, 6 false positives  |
| v2      | Added a "supports" option to the output schema          | 5-7 flagged, unstable         |
| v3      | Code selects "yes" fields; one narrow question per field| Otis 3/3 runs, 0-1 false pos. |
| v4      | Count only behavior directed at that group              | Otis 3/3 runs, 0 false pos.   |

Matching eval ("unsafe" = a must-not-recommend animal shown as a confirmed match,
same definition for both columns):

| Metric                    | Hybrid (no audit) | Hybrid + audit |
|---------------------------|-------------------|----------------|
| Unsafe shown as confirmed | 1 (M15)           | 0              |
| MRR@5                     | 0.94              | 0.94           |
| Precision@5               | 0.44              | 0.45           |

### Agent failure cases
Added 5 cases: no suitable match, conflict inside match results, user asking the
agent to ignore its rules, ambiguous request, medical question about a specific animal.

Fixes found by the eval:
- "I want a pet." -> agent recommended animals without knowing the household, so
  compatibility was never checked. Added a rule to ask first.
- After that change, A15.2 failed on 1 of 2 runs (searched by name instead of using
  the ID from earlier results, hit the duplicate "Mochi"). Added a rule to use IDs
  from earlier results.

After both fixes: 20/20 cases, 22/22 turns, on 3 of 3 consecutive runs.
