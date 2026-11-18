# CLAUDE.md — gal-trainer

## Goal
Terminal trainer (Python, rich + prompt_toolkit) for Politecnico di Milano
"Geometria e Algebra Lineare", for a student catching up before the January exam.
`Note_GAL26.pdf` is the source of truth: topic order, notation and Italian terminology follow it
(M_K(m,n), rg(A), Sol(A|b), MEG, Span, ker, Im, M^B_C(f) [superscript = domain basis],
p_A(t) = det(tI − A), ma/mg, ⟨·,·⟩, W⊥, π_W, ...). SPEC.md holds the original request,
REVIEW.md the design decisions. The course's official practice material (Fogli di esercizi,
Quiz, handwritten Esercitazioni; PDFs in `Esercizi-*/`, `Quiz-*/`, `Esercitazioni-*/`) is
transcribed in the practice bank `gal/bank/data/*.toml`, and new PDFs come in via `gal ingest`.
Past exam papers (`Esami_anni_precedenti-*/`, `Esami_A.A._*/`, 37 papers P*) are bank sources
with `kind = "exam"`, sat with `paper <id>` and scheduled by `plan`; the exam rules come from
`Informazioni_esame_GAL.pdf` (gal/papers.py).

**Correctness first.** Exact arithmetic everywhere (`fractions.Fraction`, `Surd` = q·√n).
Floats only in `gal/ai/nn.py` (numerical training demo). Every generator is verified with sympy.

## Architecture
```
gal/
  core/            exact engine, no UI
    matrix.py      Matrix over Q (immutable rows of Fractions), vector helpers
    gauss.py       MEG / Gauss-Jordan with step logs (notes' notation), rank, det (MEG),
                   det_steps (Laplace), inverse_steps, solve (Rouché-Capelli), kernel, Col basis
    poly.py        Poly over Q, charpoly = det(tI − A) via Laplace, rational roots, factored()
    surd.py        q·√n numbers (norms, distances, cosines)
    geom.py        Cartesian ⇄ parametric (rank method of §2.5), positions in R³,
                   Gram-Schmidt, projections, W⊥, cross product
    param.py       PMatrix: matrices with one parameter (entries = Poly), det, minors,
                   rank_profile (generic rank + rational drop values via gcd of minors;
                   irrational drops detected with Sturm and refused)
    fmt.py         text rendering of matrices/systems/vectors (also the input syntax), reflow
    parse.py       parser of student answers (expressions, vectors, matrices, param. sets, eqs)
  exercise.py      Answer checkers (+ likely-error diagnosis), Part, Exercise, mk()
  topics/          one module per chapter: ch0..ch5, conics (extra), ai (bonus)
    base.py        Topic (lesson, worked examples, generators) and Gen
    rand.py        "nice numbers": unimodular matrices, rank-r matrices, invertible ...
    __init__.py    TOPICS registry in course order, BY_ID, get()
  bank/            the official course material
    __init__.py    Item/Source/Bank, load_bank (one TOML per source), build(item) -> Exercise,
                   past_exam_exercises (chained 'Dal tema d'esame' exercises for mock exams)
    tasks.py       TASKS registry: task kind -> builder(data, text) -> (extra prompt, parts, steps)
    ingest.py      `gal ingest`: SHA-256 manifest, new/duplicate/changed PDFs, text extraction,
                   split into items, topic guess, skeleton TOML
    data/*.toml    F01..F08 sheets, Q01..Q04 quizzes, ES01..ES07 esercitazioni (links only),
                   P<yymm><A|I1|I2><V1..> exam papers (Parte A + B merged for older papers)
  papers.py        exam rules (RULES, written_score), PaperResult (points per Parte), study plan:
                   session_dates, paper_order, make_plan, due_session
  profile.py       JSON profile: mastery, Leitner boxes, exam history, per-item bank progress, plan,
                   past_exams (real papers graded by hand, weight PAST_WEIGHT=2 via
                   record(weight=w): alpha_w = 1-(1-alpha)^w), readiness(), parse_grade
  planner.py       `next` recommendation
  exam.py          mock-exam blueprint, timer, /30 grading
  cli.py           App (all I/O via App.ask/App.out → scriptable), commands, main()
  ai/nn.py         float matrix ops + 2-4-1 network, backprop, XOR training
  ai/chapter.py    interactive bonus chapter
tests/             helpers.py (sympy conversions), verify.py (run(topic, key, verifier)),
                   test_core / test_parse / test_generators (round trip for ALL generators) /
                   test_verify_ch01, ch2, ch34, ch5 (sympy) / test_ai / test_cli / test_diagnosis /
                   test_profile_planner /
                   test_bank (structure, round trip, generic sympy verifier per task kind) /
                   test_bank_keys (item-specific sympy checks of hand-keyed answers) /
                   test_material (every PDF ingested + hashes match; ingest pipeline end to end) /
                   test_papers (paper structure, full-marks run of EVERY paper, timer, plan) /
                   test_paper_keys (sympy re-derivation of hand-keyed paper answers)
```

### Conventions for generators
- Signature `ex_xxx(rng, data=None) -> Exercise`; with `data` it builds a fixed instance
  (worked examples from the notes). Never touch `rng` when `data` is given.
- Put the raw inputs in `mk(..., **data)` so tests can recompute the answer with sympy.
- Every Answer has `typed()` (a string the parser accepts); test_generators checks
  `check(typed()).ok` for 40 seeds of every generator.
- Add known wrong values with explanations as `mistakes=[(value, why), ...]`.
- Register the generator in the Topic's `gens`; mark `exam=True` if it fits mock exams.
- Add a sympy verifier in the matching tests/test_verify_*.py.

## How to run
```bash
pip install -e '.[dev]'
gal                 # or: python -m gal ; single command: gal practice det
pytest              # full suite (~2900 tests, under a minute)
pytest tests/test_generators.py -k determinant
pytest tests/test_bank.py tests/test_bank_keys.py tests/test_material.py   # just the bank
gal ingest          # after downloading new course PDFs (needs pdftotext or `pip install '.[ingest]'`)
```
Profile location: `$GAL_TRAINER_HOME` (default `~/.gal_trainer/`), or `--profile path`.
Bank / material location overrides: `$GAL_BANK_DIR`, `$GAL_MATERIAL_DIR` (default: repo root).

## Adding course material
`tests/test_material.py` fails while a PDF in the material folders is not in the bank:
run `gal ingest`. It writes `gal/bank/data/<ID>.toml` with `status = "skeleton"` and every item
as `task = "open"` (self-assessed, text auto-extracted, so formulas may be garbled). To
transcribe a source properly:

1. Read the PDF (pdftotext; for handwriting, render pages to PNG and look at them). Fix each
   item's `text` (the notes' notation, Italian as in the PDF; extra instructions in English in
   parentheses, e.g. "(Type 'not defined' ...)"). Keep `id` = `<SRC>-E<n><letter>` /
   `-D<n>` / `-C<n>`, `ref`, `page`, and a valid `topic` id (gal/topics/__init__.py).
2. Choose a `task` and `data` for each item. Computational items get an auto task; proofs and
   examples stay `open` with `data.solution` = a model answer. One-line examples per kind
   (all numbers exact: ints or "3/2" strings; a parameter is a string such as "λ" or "2k+1"):

   | task | data |
   |---|---|
   | open | `solution = """model answer"""` |
   | yesno / values / mcq | `answer = false, why = "..."` / `answers = [["det(A^-1)", "1/2"], ...], why` / `options = [...], correct = ["a","c"], why = {a = "..."}` (+ `multi`) |
   | solve, nsol, homog | `A = [[..]], b = [..]` (`names`) / `Ab = [[..]]` / `A` |
   | rank, det, inverse, kerim, charpoly, eigen | `A = [[..]], name = "P"` (`eigen`: `eigenspaces`, `ask_P`; `charpoly`: `spectrum = false`) |
   | matexpr | `matrices = {A = [[..]], ...}, exprs = ["A*B", "B*(A+C)^T", ...]` |
   | indep, basis_span, complete_basis, in_span | `vectors = [[..]]` (`hide`, `coords_note`, `ask_generators`); `in_span`: `+ v` |
   | basis_eqs, coords | `A = [[..]]` (equations of the subspace) / `basis = [[..]], v = [..]` |
   | cartesian, affine_forms, position | `point, dirs` / `points` or `point, dirs` (`check`) / `S1 = {point, dirs}, S2 = {...}` |
   | grassmann | `U = { A = [[..]] }` or `{ span = [[..]] }`, `W = ...` |
   | repmat, interp_matrix, change_basis | `A, B, C` or `images` (`label`, `hide`) / `basis, images, ask = ["E_E","B_B",...]` / `B, C` (`vB`, `vC`) |
   | roots, charpoly_target, eigen_target | `p = "x^3 - 3x^2 + 4"` (`var`); `eigen_target`: `+ eigenspaces = {"2" = [[..]]}, P, J` |
   | metric, gram, gram_schmidt, orthobasis | `v, w` (`S` = Gram matrix) / `G` / `vectors` (`S`) / `basis = [{dir = [..], scale = "1/sqrt(3)"}], v` |
   | param_rank, param_invertible, param_indep, param_system, param_diag | `param = "λ", A = [[1, "λ"], ...]` (`ask_det`; `param_indep`: `vectors`; `param_system`: `+ b`; `param_diag`: `candidates = [..]`, the values to inspect) |
   | perp, proj, affine_meet | `span` or `A` (`S` Gram, `name`, `orthogonal`, `only_basis`) / `+ v`, `ask = ["W","perp","dist"]` / `S1, S2` (`name1/2`) |

   Generic options: `keep = ["basis of", ...]` asks only the parts whose label starts with a
   prefix (the generic verifiers still check all of them); grassmann `parts = ["U", "W",
   "dim_sum", "dim_int", "direct", "int_basis", "sum_basis"]` and `uw = ["V", "W"]` (the
   paper's names); matexpr `labels` (hide the formula, e.g. `X` for `M^-1*N`) and `^-1`;
   values accept exact roots (`"sqrt(2)"`); mcq `why` may be one string, `accept = ["a"]`
   for an option that is also true as worded; `setup = "..."` on the first sub-item of an
   exercise is inherited by its siblings (shown once in paper mode).

   Builders live in gal/bank/tasks.py: read the one you use for every option. `build(item)`
   must not raise; a TaskError (e.g. a parameter whose rank drops at irrational values)
   means the item must be `open` or keyed by hand (`values`/`yesno`/`mcq`).
3. Past-exam exercises: set `exam = "dd/mm/yyyy"` on every sub-item; they are then chained
   into one multi-part exercise in mock exams.
4. Exam papers: `[source] kind = "exam"`, `paper = { format = "appello"|"itinere", minutes,
   points, lecturer, version, half = 1|2, points_estimated }`, the solutions PDF in `files`
   with `role = "solutions"`. Every item needs `section = "I"` (test) or `"II"` (exercises)
   and `points`; this lecturer's papers must total 32 (appello) / 16 (itinere). Keys: from the
   official solutions (the `@`-marked option; for itinere papers check by computing — their
   solutions reorder the options), else compute them and add a check in
   tests/test_paper_keys.py. Delete the ingest skeleton(s) of the files you merge.
5. Handwritten solutions (esercitazioni): a source with `kind = "esercitazione"` and no items,
   plus `[[solution]] items = ["F03-E2b", ...], pages = [2]` entries. Every exercise then shows
   "Solved in class (handwritten): ..., p. 2 → <pdf path>".
6. Set `status = "transcribed"`, run `pytest tests/test_bank.py tests/test_material.py`.
   Auto tasks are recomputed with sympy by the generic verifier of their kind in
   tests/test_bank.py (add one there for a NEW task kind, and register the builder with
   `@task("name")`). For hand-keyed answers (yesno/mcq/values) add an item-specific sympy check
   in tests/test_bank_keys.py whenever the answer is computable. MCQ: the number of correct
   options must match "(N risposte)" in the text, if present.
7. A revised PDF with the same name is reported as *changed*: diff it against the source,
   update the items, then `gal ingest --rehash` stores the new hash.
