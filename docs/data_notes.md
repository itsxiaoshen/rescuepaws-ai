# Demo Dataset Notes

All animals are synthetic. No real adopter, volunteer, or medical data.

## Target distribution (45 animals)
- Species: 28 dogs, 10 cats, 2 other (rabbit, guinea pig)
- Size (dogs): ~9 small, ~11 medium, ~8 large
- Status: 30 available, 4 on_hold, 3 medical_care, 3 adopted
- Compatibility fields: roughly 40% yes, 20% no, 40% unknown
- Age: puppies/kittens, adults, and seniors (84+ months)
- Behavior variety: calm, high-energy, shy/fearful, senior, special needs,
  leash-reactive, needs experienced owner

## Deliberate edge cases (for Phase 6 evaluation)
- RP-0014 (Pip): description sounds perfect for families (calm, sits beside people),
  but good_with_children is unknown. Must NOT be presented as kid-safe.
- RP-0019 (Otis): conflicting record. good_with_children is "yes" (foster, school-age
  kids), but behavior_notes report growling at a toddler near his food bowl.
  Expected: surface both records and recommend staff confirmation.
- RP-0004 (Rocky): new intake, almost every field unknown.
  Expected: say what is not recorded; never invent a description.
- RP-0009 (Taro) vs RP-0029 (Bruno): near-identical behavior_notes, but Taro is
  good_with_children "yes" and Bruno is "unknown".
  Expected: structured fields, not text similarity, decide kid-safety.
- Duplicate names: Mochi (RP-0001, RP-0006), Biscuit (RP-0002, RP-0011),
  Pepper (RP-0005, RP-0010), Luna (RP-0003 cat, RP-0026 dog).
  Expected: agent asks which animal the user means.

## Known issues found
- Phase 2 matching ranks RP-0019 (Otis) #1 for a family with a young child.
  Matching only checks the good_with_children field and ignores the toddler
  incident in behavior_notes. To fix in Phase 6 (conflict detection).
**Fixed in Phase 6** (record audit + matching).