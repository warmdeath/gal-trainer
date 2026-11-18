# Testing strategy

The trainer's value is that its keys are right. Tests are how you know.

| file | what it proves |
|---|---|
| test_core | engine vs CAS on hundreds of random inputs (incl. degenerate cases) |
| test_parse | every input syntax the README documents, plus garbage → ParseError |
| test_generators | for EVERY generator, 40 seeds: builds, `check(typed()).ok`, prompt non-empty |
| test_verify_<chapter> | independent recomputation of each generator's answer with sympy from `Exercise.data` |
| test_diagnosis | the named mistakes are recognised |
| test_bank | bank structure (unique ids, valid topics/tasks), round trip of every item, one generic sympy verifier per auto task kind, MCQ counts match "(N risposte)" |
| test_bank_keys / test_paper_keys | item-specific sympy checks of hand-keyed answers (mcq / values / yesno) |
| test_material | every PDF in the folders is in the bank and its hash matches; ingest end to end on a temp copy (new, duplicate, changed, idempotent, rehash) |
| test_papers | paper structure and totals; a scripted **full-marks run of every paper**; timer cutoff with a fake clock; wrong answers and self-grades; plan dates and order; `next` gives a due paper |
| test_profile_planner, test_cli | mastery maths, Leitner, persistence, every command scripted via `App(ask=Script([...]))` |

Rules of thumb
- A verifier must not call the code it verifies; recompute from raw data with sympy.
- In verifiers, use private symbol names (`_lam`) — an exercise parameter called `t` or `x`
  will otherwise collide with your own symbol.
- Give the app an injectable `clock` and `rng` so timed and random flows are testable.
- Seeded CLI tests break when a new random draw is added (e.g. mixing official items into
  practice): make such features configurable and switch them off in tests.
- Keep the suite under about a minute; profile the slow ones (a hang in root finding or
  simplification on a large "generic" value is a bug, not slowness).
- Never commit with a failing test; `pytest -q | tail` hides the exit code — check it.
