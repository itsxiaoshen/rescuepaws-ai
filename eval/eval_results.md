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
