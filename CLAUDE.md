# CLAUDE.md — gal-trainer

## Goal
Terminal trainer (Python, rich + prompt_toolkit) for Politecnico di Milano
"Geometria e Algebra Lineare", for a student catching up before the January exam.
`Note_GAL26.pdf` is the source of truth: topic order, notation and Italian terminology follow it
(M_K(m,n), rg(A), Sol(A|b), MEG, Span, ker, Im, M^B_C(f) [superscript = domain basis],
p_A(t) = det(tI − A), ma/mg, ⟨·,·⟩, W⊥, π_W, ...). SPEC.md holds the original request,
REVIEW.md the design decisions.

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
    fmt.py         text rendering of matrices/systems/vectors (also the input syntax)
    parse.py       parser of student answers (expressions, vectors, matrices, param. sets, eqs)
  exercise.py      Answer checkers (+ likely-error diagnosis), Part, Exercise, mk()
  topics/          one module per chapter: ch0..ch5, conics (extra), ai (bonus)
    base.py        Topic (lesson, worked examples, generators) and Gen
    rand.py        "nice numbers": unimodular matrices, rank-r matrices, invertible ...
    __init__.py    TOPICS registry in course order, BY_ID, get()
  profile.py       JSON profile: mastery, Leitner boxes, exam history
  planner.py       `next` recommendation
  exam.py          mock-exam blueprint, timer, /30 grading
  cli.py           App (all I/O via App.ask/App.out → scriptable), commands, main()
  ai/nn.py         float matrix ops + 2-4-1 network, backprop, XOR training
  ai/chapter.py    interactive bonus chapter
tests/             helpers.py (sympy conversions), verify.py (run(topic, key, verifier)),
                   test_core / test_parse / test_generators (round trip for ALL generators) /
                   test_verify_ch01, ch2, ch34, ch5 (sympy) / test_ai / test_cli / test_diagnosis /
                   test_profile_planner
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
pytest              # full suite (~1250 tests, ~25 s)
pytest tests/test_generators.py -k determinant
```
Profile location: `$GAL_TRAINER_HOME` (default `~/.gal_trainer/`), or `--profile path`.
