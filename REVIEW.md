# REVIEW — architecture, decisions, simplifications, next ideas

## What was built

| | |
|---|---|
| Topics | 26: 24 that follow `Note_GAL26.pdf`, plus *conics* (extra) and the *AI* bonus chapter |
| Exercise generators | 87, all random with nice numbers, each verified against sympy |
| Worked examples | 37, mostly the numerical examples of the notes, recomputed by the engine |
| Official material | 8 Fogli + 4 Quiz = 385 items (341 auto-checked, 44 self-assessed); 7 handwritten Esercitazioni linked to 69 items |
| Tests | ~1900 (`pytest`, ~25 s) |
| Code | ~8.5k lines in `gal/`, ~2.5k lines in `tests/`, ~4.4k lines of bank data |

## Architecture (summary — details in CLAUDE.md)

1. **Exact core (`gal/core`)**
   - `Matrix` over Q, built on `fractions.Fraction`.
   - Gaussian elimination and Gauss-Jordan that *log their steps* in the notes' notation
     (`R1 ↔ R3`, `R2 → R2 - 2R1`).
   - Characteristic polynomial computed as `det(tI − A)` by Laplace expansion on polynomial
     entries. Exact rational root finding gives `ma`.
   - `Surd` (q·√n) for norms, distances and cosines, so those answers stay exact
     (`12*sqrt(11)/11`).
   - A dedicated parser for what the student types: implicit multiplication, `sqrt`, √,
     parametric sets, equation systems.
2. **Exercises (`gal/exercise.py`)**: an `Exercise` has a prompt, one or more `Part`s and a
   full solution. Every `Answer` type checks *mathematical equivalence*, not string equality:
   - `SpanAns`: any basis of the subspace.
   - `AffineAns`: any particular solution plus any basis of the directions.
   - `EquationsAns`: any equivalent Cartesian system.
   - `DiagonalizerAns`: any invertible P with P⁻¹AP diagonal.
   - `ScaledVectorsAns`: Gram-Schmidt vectors up to rescaling.
   - Plus scalar, exact-real, vector, matrix, polynomial, set, yes/no and multiple-choice types.
3. **Topics (`gal/topics`)**: one module per chapter. Each `Topic` has a lesson (intuition →
   definitions with Italian terms → method), worked examples, generators and prerequisites.
   The registry order is the order of the notes.
4. **Learning loop**: `profile.py` (mastery + Leitner boxes, atomic JSON saves, corrupt-file
   backup), `planner.py` (`next`), `exam.py` (blueprint, timer, grade /30), `cli.py` (rich
   panels, prompt_toolkit input with history and a countdown toolbar in exams).
5. **Testability**: all I/O goes through `App.ask` / `App.out`, so tests drive the whole UI
   with scripted answers (`tests/test_cli.py`).
6. **Practice bank (`gal/bank`)**: the official material as *data*, not code. Each source
   (sheet, quiz, esercitazione) is one TOML file; each item names a *task kind* plus the
   raw data (matrices, vectors, parameter expressions). `tasks.py` turns `(task, data)` into
   the same `Part`/`Answer` objects the generators use. Official exercises get the same
   equivalence checking, diagnosis and step-by-step solutions for free. `ingest.py` is the
   front door for future PDFs. The analogy is a library: `ingest` is the acquisitions desk,
   which stamps every new book with a fingerprint (SHA-256) so a second copy isn't shelved
   twice. A skeleton TOML is the catalogue card, readable at once. Transcribing an item
   into a task is the full cataloguing that makes it searchable (auto-checked).

## How correctness is ensured

- `tests/test_core.py`: rank, rref, det, inverse, kernel, column space, `solve` and charpoly
  are compared with sympy on hundreds of random matrices, including rank-deficient ones.
  The numerical examples of the notes are asserted explicitly: det −33 and −72, the inverse
  on p.23, (t−1)²(t−2), the solution (−28,−12,7), the plane 7x − y − 3z = 6, and the
  Cartesian equations of §2.5.
- `tests/test_generators.py`: for **every** generator and 40 seeds, the model answer typed
  back through the parser must be accepted (catches bugs in the formatter, the parser and the
  checker), and generation must be deterministic.
- `tests/test_verify_*.py`: for every generator, the expected answer is recomputed
  **independently with sympy** from the raw data stored in the exercise. Examples:
  - `gauss_jordan_solve` for systems;
  - `is_diagonalizable` for diagonalisation, including the k-parameter case scanned over k;
  - `GramSchmidt`;
  - projections via W(WᵀW)⁻¹Wᵀ;
  - distances via calculus minimisation with `sympy.solve`;
  - conic types via the value at the centre.
- `tests/test_diagnosis.py`: typical mistakes are recognised: BA instead of AB, entry-wise
  product, transposed inverse, missing Laplace signs, det(A−tI), Fourier coefficients with a
  non-orthogonal basis.
- AI chapter: backprop is checked against numerical gradients (error < 1e-6), training
  converges on 5 seeds, and the exact ReLU XOR network is evaluated with `Matrix`.

## Coverage of Note_GAL26.pdf

| Notes | Trainer topic | Exercises |
|---|---|---|
| §0.1 vettori geometrici | `vectors` | combinations, [AB] |
| §0.2 rette e piani | `lines_planes` | line through 2 points, param → Cartesian, plane through 3 points |
| §1.1 matrici, operazioni | `matrices` | aA+bB, AB, when AB is defined, (A+B)², (AB)ᵀ |
| §1.2–1.2.2 sistemi, a scala, MEG, Gauss-Jordan | `systems` | solve, echelon?, free variables |
| §1.2.3 rango, Rouché-Capelli, struttura, Cramer | `rank` | rank, parameter in b, parameter in A |
| §1.2.4 inversa | `inverse` | A⁻¹ via (A\|I), invertible? |
| §1.3 determinante, Laplace, operazioni, Binet | `determinant` | Laplace, row ops, Binet/det(λA) |
| §2.1–2.2 spazi e sottospazi | `vector_spaces` | is W a subspace? (R³, polynomials, matrices) |
| §2.3 combinazioni, generatori, indipendenza | `span_independence` | coefficients, b ∈ Span?, independence, dependence relation, generators, polynomials |
| §2.4 basi, dimensione, coordinate, rango (2.4.1) | `bases` | dim & basis of a span, coordinates, completing a basis, Col(A) |
| §2.5 equazioni cartesiane/parametriche | `cart_param` | both directions (rank method of the notes) |
| §2.6 intersezione, somma, Grassmann | `sum_intersection` | all dimensions + direct sum, basis of U∩W |
| §2.7 sottospazi affini, posizioni reciproche, fasci | `affine` | line–line (+ intersection point), line–plane, plane–plane, pencil of planes |
| §3.1–3.3 applicazioni lineari, ker, Im, iniettività/suriettività, nullità+rango | `linear_maps` | linear?, ker/Im of L_A, nullity-rank, maps on R[t]≤2 |
| §3.4 interpolazione, matrice rappresentativa, cambio di base, simili | `rep_matrix` | M^E_E, f on a basis, M^B_C(f), M^E_{B'}(id) and coordinates, P⁻¹AP |
| §4.1–4.2 autovalori, polinomio caratteristico | `eigen` | p_A, spectrum over R, eigenspace, is v an eigenvector? |
| §4.3–4.5 molteplicità, diagonalizzazione, matrice compagna | `diagonalization` | diagonalisable?, ma/mg, find P, parameter k, C(p) |
| §5.1 forme bilineari, Gram, congruenza | `bilinear` | Gram matrix, vᵀAw, PᵀAP |
| §5.2 prodotti scalari, norma, distanza, angolo | `scalar_products` | positive definite? (completing squares, as in the notes), norm (also with a Gram matrix), distance, cos θ |
| §5.3–5.3.1 ortogonalità, basi ortonormali, Fourier | `orthogonality` | Fourier coefficients, normalising, w⊥ |
| §5.3.2 Gram-Schmidt | `gram_schmidt` | orthogonal basis, orthonormal vector |
| §5.3.3–5.3.4 complemento ortogonale, proiezioni, matrici | `projections` | W⊥, π_W(v), M(π_W), d(v, W) |
| §5.4–5.5 prodotto vettoriale, distanze | `cross_distances` | v×w, area, point–plane, point–line, skew lines |
| §5.6 matrici ortogonali, isometrie | `isometries` | orthogonal?, rotation or reflection?, reflection across a line |

Lesson-only (no generator):

- §1.3.1 geometric meaning of det;
- §1.3.3 teorema degli orlati;
- §3.1.1 Hom(V,W) as a vector space;
- the isomorphism with Kⁿ (§3.4);
- §5.6.2 rotations ρ and affine transformations;
- Cauchy-Schwarz and the triangle inequality.

These are mostly theory. The definition, statement and an example are in the lesson.

## Decisions and simplifications (with reasons)

1. **The notes override the requested topic list.** The order and grouping follow the notes:
   - lines and planes appear at §0.2, in §2.7 (affine positions) and in §5.5 (distances),
     instead of a single "lines and planes" topic near the end;
   - kernel/image and linear maps are one topic, because §3.1–3.3 treat them together.
2. **Conics are not in the notes.** The notes stop at §5.6 and announce quadratic forms and
   quadrics for a later chapter. The request lists "conics basics", so there is a short
   *extra* topic: invariants of the 3×3 matrix, type, centre. It is flagged `extra`, listed
   last, excluded from `next` and from mock exams, and its lesson says to follow the official
   notes once published.
3. **Language.** The lessons are in English with the Italian term next to the English one, as
   requested; theorem names stay in Italian (Rouché-Capelli, nullità più rango, ...). The UI
   is in English. Italian yes/no answers are accepted (`sì`, `no`).
4. **Representative matrix notation.** In plain text it is written M^B_C(f): superscript =
   basis of the domain, subscript = basis of the codomain (the notes' typesetting). The
   lesson states this explicitly.
5. **Fraction-free elimination in the displayed steps.** When dividing would create fractions
   from integers, the steps use `Ri → a·Ri + k·Rj` (a ≠ 0): move 2 followed by move 3. This is
   always legal and much easier to follow by hand. The lesson explains it and notes that it
   multiplies det by a. `det()` itself uses the plain MEG. Pivots equal to ±1 are preferred,
   which is what students do on paper.
6. **One attempt per part.** On a wrong answer, the likely error and the full solution appear
   immediately, as requested. Unreadable input is never penalised.
7. **Likely-error diagnosis.** It works on two levels:
   - generic heuristics per answer type: wrong sign, reciprocal, k-times, transposed, a single
     wrong entry/row/column, dependent vectors, a vector not in W, wrong number of
     parameters, square instead of root;
   - per-generator "known wrong answers", computed by *simulating* the typical mistake (BA,
     entry-wise product, Laplace without signs, det(A−tI), naive Fourier with a
     non-orthogonal basis, P A P⁻¹, P⁻¹AP for a bilinear form, forgetting ReLU or the bias, ...).

   It can't reconstruct arbitrary errors. In those cases the full solution is the fallback.
8. **Fields.** Exercises are over Q/R only. Diagonalisation over C and the field with two
   elements appear in the lessons, not in the exercises, because exact complex arithmetic
   would add a lot of code for very little exam value.
9. **Nice numbers.** These generators give small integers by construction:
   - unimodular change of basis matrices, which have integer inverses;
   - A = P D P⁻¹ for diagonalisation;
   - rank-r products L·R;
   - Householder products for rational orthogonal matrices;
   - Pythagorean triples for O(2).

   Generators that can still produce ugly fractions (projections, Gram-Schmidt, least squares)
   resample until the denominators are small.
10. **Mastery and scheduling.** I chose simple, explainable rules over SM-2:
    - mastery is a running mean of the scores, then a moving average with α = 0.25;
    - review intervals follow Leitner boxes: 0, 1, 2, 4, 7, 14 days;
    - "mastered" means mastery ≥ 75%, at least 3 attempts, and a last answer that wasn't a
      failure (otherwise a just-failed topic would never come back).
11. **Mock exams.** The 6-slot blueprint is my *approximation* of a GAL written exam (systems
    with parameters, subspaces, linear maps, diagonalisation, Euclidean, geometry), not an
    official template. There is no feedback until hand-in; then you get a grade /30, a
    per-exercise table and solutions on request. Exam results also update mastery. Real
    exams also contain theory questions and proofs, which are not simulated.
12. **The AI chapter uses floats on purpose.** Training is numerical, and it is the only
    place with floats. To keep the link with the course exact, the chapter also runs:
    - an exact hand-built ReLU network for XOR, computed with `Matrix`;
    - exact exercises (a layer forward pass, collapsing two linear layers into W2W1, least
      squares as an orthogonal projection) verified with sympy.
13. **Default exam date** is 2027-01-18 (a guess for the January session). Change it with
    `examdate`.
14. **PDF extraction.** The notes were read through `pdftotext`; formulas come out garbled
    in places. Every numerical example reused from the notes was recomputed by the engine
    and is asserted in the tests, so a transcription error would show up as a failure.

### Practice bank (official material)

15. **Transcribed by hand, verified by machine.** pdftotext garbles matrices, so every
    item was rewritten from the PDF pages. Correctness comes from the tests:
    - every auto task is recomputed from its raw data by an independent sympy verifier per
      task kind (`tests/test_bank.py`);
    - every hand-keyed answer that is computable (yes/no, multiple choice, values) has an
      item-specific sympy check (`tests/test_bank_keys.py`);
    - MCQ keys must match the "(N risposte corrette)" count printed in the quiz.
16. **Data, not code, for new material.** Adding a sheet means writing TOML, never Python,
    unless a genuinely new kind of question appears. New PDFs are usable the moment they are
    ingested (self-assessed), and become auto-checked item by item. `test_material.py` fails
    while a PDF in the folders is unknown, so new material can't be forgotten silently.
17. **Fingerprints over file names.** The course website zips folders with a date suffix
    (`Esercizi-20261117`), so a re-download duplicates every file. The manifest keys on
    SHA-256:
    - same content means a duplicate, ignored;
    - same name with different content means a revision, reported and accepted only with
      `--rehash`, after a human has checked it.
18. **"Al variare di λ" is exact.** For a matrix with a parameter, rank drops happen exactly
    at the common roots of the r×r minors, so the trainer takes the gcd of those polynomials
    and finds its rational roots. If the gcd also has *irrational* real roots (checked with a
    Sturm sequence), the builder refuses the item rather than give an incomplete answer, and
    the item must be keyed by hand. The *critical values* the student must list are the
    values where the outcome differs from the generic one. The trainer does not ask for every
    root of det, because a root can leave the outcome unchanged, and the sheets don't ask for
    it either.
19. **Self-assessment for proofs.** Proofs, counterexamples and "give an example" questions
    have many correct answers that can't be checked mechanically. The student writes theirs,
    sees a model answer, and grades it y/p/n. The grade feeds mastery like any other score.
20. **Handwritten esercitazioni are linked, not parsed.** OCR of tablet handwriting is
    unreliable, so each page was read once and mapped to the sheet items it solves. The
    trainer points to the PDF and page. Their numerical results were cross-checked against
    the engine, and all matched.
21. **Past exams in mock exams.** Exercises marked "Dal tema d'esame" (F06-E4, F08-E1) are
    chained into one multi-part exercise, as on the real paper. One replaces the last mock-exam
    slot with probability 1/2, and its score is credited to each official sub-item.
    Self-assessed sub-questions are left out of the exam.
22. **Typos in the material, and how they were read** (each is stated in the item text):
    - F08-E2 prints `v = (1, −1, 1 − 1)`; it is read as (1, −1, 1, −1).
    - Q02-D15's options call the same line both *s* and *l*; they are the same line.
    - F02-E4b is inconsistent for *every* λ: the key says so, rather than "fixing" the
      exercise.
    - F07-E13 b) is ambiguous ("Xv ∈ Span(v)" vs "v eigenvector"); the read that includes O is
      used, as the model answer explains.

## Known limitations

- No checking of intermediate steps: only final answers are graded. A wrong echelon form
  with a correct rank is accepted.
- Generic diagnosis can be wrong about *why* an answer is wrong. It says "likely error"
  for that reason.
- Generated exam exercises are shorter than real exam exercises (which chain many steps), so
  the default 10 minutes per exercise is generous; use `exam n minutes` to tighten it.
- Official items are only as good as their transcription. A misread entry would be
  "verified" against the wrong data. Mitigations: page numbers on every item, the PDF path
  in the solution, and cross-checks against the handwritten esercitazioni where they exist.
- `gal ingest` text extraction is rough (formulas garbled, headers occasionally missed). The
  skeleton items are readable but need a human (or Claude Code) pass before auto-checking.
- Self-assessment is only as honest as the student.
- Terminals without good Unicode fonts may show boxes for ⎡ ⎤ ⟨ ⟩ (see the README for
  Windows).

## Next ideas

1. Theory drill: true/false and "state the theorem" cards for every definition and theorem
   of the notes (orlati, Steinitz, Grassmann, nullità+rango, dominazione, Cauchy-Schwarz),
   with spaced repetition.
2. Step-level feedback: let the student type the echelon form or the eigenvalues before the
   final answer, and check each step (the engine already has the step logs).
3. Progressive hints: reveal the solution one step at a time instead of all at once.
4. Spaced repetition per generator (not just per topic), and SM-2-style ease factors.
5. Quadratic forms and quadrics as soon as the notes publish that chapter (signature,
   Sylvester, canonical forms). This would replace the extra conics topic.
6. Exercises over C (complex eigenvalues, unitary diagonalisation) using an exact
   Gaussian-rational type.
7. Full past papers: when the course publishes *temi d'esame* (folders `Esami*`/`Temi*` are
   already scanned by `gal ingest`), transcribe them as sources with `exam = "date"` on every
   item and add an `exam past <date>` command that replays a real paper under its real
   time limit.
8. Assisted transcription: a `gal ingest --draft` that proposes `task`/`data` for simple
   items (systems, determinants) from the extracted text, for a human to confirm.
9. Fully Italian UI option, and an export of the mistakes log to review with a tutor.
