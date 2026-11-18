# Architecture reference

Contents: Layout · Engine · Answers and exercises · Topics and generators · Bank ·
Ingest · Profile and planner · CLI · Linear-algebra notes (reference implementation)

## Layout

```
<course>-trainer/
  pyproject.toml        console script `<short>` → <pkg>.cli:main ; extras: dev, ingest
  SPEC.md CLAUDE.md REVIEW.md README.md
  notes/digest/*.md     typed digest of the (handwritten) notes, with page refs
  <pkg>/
    core/               engine, no UI: exact/symbolic maths, parse.py (student input), fmt.py
    exercise.py         Answer types, Check, Part, Exercise
    topics/             base.py (Topic, Gen), one module per chapter, rand.py (nice numbers),
                        __init__.py (TOPICS in course order, BY_ID)
    bank/               __init__.py (Item/Source/Bank, load_bank, build), tasks.py (TASKS),
                        ingest.py, data/*.toml
    papers.py           exam RULES, PaperResult, study plan
    profile.py planner.py exam.py cli.py
  tests/                helpers.py + one test file per area (see testing.md)
  <material folders exactly as downloaded, e.g. Esercizi-20261117/, Esami_.../>
```

Material folders stay at the repo root with their download names; ingest scans them by
pattern. Never rename the student's files: hashes and paths in the bank refer to them.

## Engine

- Pure functions, no I/O, every function returns its **steps** as text in the course's
  notation (`R2 → R2 − 2R1`, "per il teorema di de l'Hôpital ...") so solutions can be shown.
- Exactness: rationals via `fractions.Fraction`; irrational results via a small exact type
  (e.g. q·√n) or, for analysis, sympy expressions. Floats only where the topic itself is
  numerical, never in a key.
- Keep "nice number" generators separate (`topics/rand.py`): build answers first
  (integer roots, unimodular matrices, perfect squares), then the question.

## Answers and exercises

```python
@dataclass
class Check:            # result of checking one student answer
    ok: bool
    score: float = 0.0  # partial credit in [0, 1]
    msg: str = ""       # diagnosis shown to the student
    @staticmethod
    def right(msg=""): ...
    @staticmethod
    def wrong(msg="", score=0.0): ...

class Answer:
    fmt_hint: str                    # shown on '?': how to type it
    self_assessed = False            # True for OpenAns (proofs): y/p/n after the model answer
    def __init__(self, ..., mistakes=()): ...   # [(wrong_value, explanation), ...]
    def check(self, text) -> Check   # raise ParseError if unreadable (never counted wrong)
    def show(self) -> str            # the correct answer, pretty
    def typed(self) -> str           # a string the parser accepts and check() marks right

@dataclass
class Part:  label: str; answer: Answer
@dataclass
class Exercise:
    topic: str; kind: str; prompt: str; parts: list[Part]; solution: list[str]; data: dict
```

Answer types used by the reference project (adapt to the course): scalar/fraction, exact
real, vector, matrix (or "not defined"), set of values, polynomial, yes/no, single and
multiple choice (partial credit (right − wrong)/#right), span/basis (any basis of the same
space), affine set (any point + any directions), equations (any equivalent system),
"diagonaliser" (any valid P), vectors up to scaling, open (self-assessed: the student types
y/p/n after seeing the model answer).

Diagnosis: each Answer may carry `mistakes` — specific wrong values with an explanation —
and generic heuristics (sign error, transposed, off by a factor). Say "likely error",
because the guess can be wrong.

## Topics and generators

```python
@dataclass
class Gen:  make: Callable[[Random], Exercise]; weight: float = 1; exam: bool = False
@dataclass
class Topic:
    id: str; ref: str; title: str; title_local: str; lesson: list[tuple[str, str]]
    examples: list[Callable[[], Exercise]]; gens: list[Gen]; prereqs: list[str]; extra=False
```
Generator signature `ex_xxx(rng, data=None) -> Exercise`; with `data` it rebuilds a fixed
instance (the notes' worked examples) without touching `rng`. Put raw inputs in
`Exercise.data` so tests can recompute the answer independently.

## Bank

One TOML per source in `bank/data/`:
```toml
[source]
id = "F02"            # F = sheet, Q = quiz, ES = class solutions, P = exam paper
title = "Foglio di esercizi 2"
kind = "sheet"        # sheet | quiz | esercitazione | exam
files = [{ path = "Esercizi-20261117/Foglio_esercizi_02.pdf", sha256 = "...", pages = 3 }]
status = "transcribed"            # or "skeleton" (written by ingest)
# exam papers only:
paper = { format = "appello", minutes = 105, points = 32, lecturer = "...", version = "V1", half = 2, points_estimated = true }

[[item]]
id = "F02-E1a"        # <SRC>-E<n><letter> | -D<n> (domanda) | -C<n>
ref = "Esercizio 1 (a)"
topic = "systems"     # a Topic id
page = 1
task = "solve"        # kind in tasks.TASKS; "open" = self-assessed
setup = "..."         # shared statement, first sub-item only; siblings inherit it
text = "..."          # the question, course wording; English hints in parentheses
data = { A = [[2, 1], [2, -5]], b = [3, 5] }   # numbers as ints or "3/2"; params as "λ"
section = "II"        # exam papers: I = test, II = exercises
points = 2.2          # exam papers

[[solution]]          # in esercitazione sources: handwritten solution pages
items = ["F03-E2b"]
pages = [2]
```
`tasks.py`: `@task("name") def t_name(d, text) -> (extra_prompt, [(label, Answer)], steps)`.
Generic options worth implementing from day one: `keep = ["label prefix", ...]` (ask only
some parts of a rich task), `labels` (hide formulas), naming of objects as in the paper
(`uw = ["V", "W"]`), `accept` for MCQ alternatives, `why` (string or per-option dict).
`build(item, with_setup=True) -> Exercise`; a builder raising `TaskError` means the item must
be `open` or keyed by hand (`values` / `yesno` / `mcq`).

## Ingest

`ingest(root, bank_dir, rehash=False)`: walk folder patterns (`Esercizi*`, `Esercitazioni*`,
`Quiz*`, `Esami*`, `Temi*`, `material/`), SHA-256 each PDF.
- hash known → duplicate (a re-downloaded zip): ignore;
- same file name, new hash → *changed*: report, accept only with `--rehash` after review;
- new → `kind_of` (by file name and parent folder, e.g. `soluzion` in the *file* name),
  propose an id (`P<yymm><A|I1|I2><V..><PA|PB>` for papers), extract text (pdftotext,
  fallback pypdf), split at `Esercizio N` / `Domanda N` / `N.` lists, guess topics by
  keywords, write a skeleton TOML (items `open`, exam items with section + estimated points).
Fill missing hashes automatically. A test fails while any PDF is not in the bank, so new
material can't be forgotten. Show a startup notice when new PDFs are found.

## Profile and planner

Profile JSON in `$<APP>_HOME` (default `~/.<app>/`), atomic write (tempfile + replace),
corrupt file → backup + fresh profile. Per topic: mastery, attempts, correct, Leitner box,
due date, last. Per bank item: attempts, best, last. Past papers: label, date, grader,
points/max, grade, per-exercise topics and points, paper_id. Plan: list of
`{date, paper, done, result}`.
- `record(topic, score, weight=1)`: alpha = max(0.25, 1/attempts); weight w ⇒
  alpha_w = 1 − (1 − alpha)^w (w identical results, one attempt). Real papers use w = 2.
- Leitner intervals 0, 1, 2, 4, 7, 14 days; fully right moves up, < 0.5 resets to box 0.
- Mastered: mastery ≥ 0.75, ≥ 3 attempts, box ≥ 1.

## CLI

`App(profile, console, ask, rng, clock)`; `command(line) -> bool`. Commands: next, topics,
learn, practice, review, exam, stats, examdate, sheets/sheet/quiz/official/item, proofs,
grade, papers, paper, plan, examinfo, past, ingest, help, quit. prompt_toolkit session
with history and a bottom-toolbar countdown during timed runs. Inputs `?` (format), `skip`,
`q`. Mix ~30% unsolved official items into practice. Self-graded parts: "write it on
paper, Enter", then model answer, then y/p/n (also 0.7, 70%, 3/4).

## Linear-algebra notes (reference implementation)

Exact `Fraction` matrices; Gauss/Gauss-Jordan with step logs; rank, det (MEG and Laplace),
inverse, kernel, column space; characteristic polynomial det(tI − A) by Laplace on
polynomial entries; rational roots; parametric matrices (rank drops = rational roots of the
gcd of minors, irrational drops detected with a Sturm sequence and refused); Gram-Schmidt,
projections, W⊥; exact surds for norms. Answer types: span, affine set, equations,
diagonaliser. Bank tasks: solve, rank, det, inverse, kerim, repmat, change_basis, eigen,
param_* (λ discussions), grassmann, perp, proj, affine_meet, gram, gram_schmidt, metric.
