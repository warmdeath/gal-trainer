---
name: course-trainer
description: "Build a complete terminal study trainer (Python, rich + prompt_toolkit) for a university course from its own materials — lecture notes (typed or handwritten), exercise sheets, past exams with or without solutions, and the exam-rules sheet — with lessons, unlimited auto-checked exercises, mistake diagnosis, the official exercises and every past paper transcribed and auto-graded, timed exam simulations, spaced repetition and a study plan up to the exam date. Use this skill whenever someone wants to turn course material into a practice/revision tool, exam simulator or study app, wants to 'catch up' on a course before an exam with exercises that get checked, or asks to do for another course what was done for a previous trainer (e.g. 'make the same thing for Analisi 1 / Fisica / Probabilità'), even if they don't say 'trainer'. Includes a dedicated playbook for Analisi Matematica 1 (symbolic answers: limits, derivatives, integrals, series, ODEs)."
---

# Course trainer

You are going to build, for one university course, a study tool that a student will use
daily until the exam. It is modelled on a working project (a trainer for Politecnico di
Milano "Geometria e Algebra Lineare") and this skill distils what made that project work
and what went wrong along the way, so you can get there faster and avoid the same holes.

Think of the result as a gym with three rooms: **lessons** (learn a topic from the
course's own notes), the **practice floor** (unlimited generated exercises + every
official exercise, each checked and explained), and the **match room** (real past papers
under exam conditions, scheduled by a plan that ends on exam day). A profile follows the
student between rooms and decides what to do `next`.

## Non-negotiables (and why)

- **The course material is the source of truth.** Topic order, notation and terminology
  (in the course's language, e.g. Italian terms) follow the notes. A student who learns a
  different notation from the trainer loses points on the real exam.
- **Correctness first.** Every answer key the trainer holds — generated or transcribed —
  must be verified by an independent computation in the test suite. A trainer that
  confidently marks a right answer wrong does more harm than no trainer.
- **Accept every correct form.** Check mathematical equivalence, never string equality:
  any basis of the same subspace, any equivalent equation, `2/sqrt(2)` = `sqrt(2)`,
  `C·e^x` with any letter for the constant. Unparseable input is never counted as wrong.
- **Explain mistakes.** On a wrong answer, name the likely error when you can
  ("you differentiated the inner function only", "you forgot the +C") and always show the
  full worked solution in the course's notation.
- **Exam conditions are real conditions.** Past papers run with the real duration and
  scoring, no feedback until hand-in, proofs written on paper and self-graded at the end.

## Working style with the user

Ask questions whenever a choice is genuinely theirs (scope, exam date, which papers count,
where the code lives) — batch them, give a recommended option, and explain trade-offs.
Prefer finding a way to reach the user's full goal over quietly reducing scope; if
something is very large (e.g. 60 exam PDFs), say so, propose an order, and keep going.
Commit after every milestone (one paper, one chapter) so nothing is lost if the session
restarts, and keep the docs (CLAUDE.md, README, REVIEW) current as you go. Analogies help
when explaining design choices to the student.

## Phases

Work through these in order; each ends with green tests and a commit. Read the reference
file named in each phase when you get there — not before.

### Phase 0 — Kick-off (ask, then set up)
1. Ask: exam date and session; which materials exist (notes, sheets, solutions, past
   exams, rules sheet); language of the UI (default: English UI, course-language terms);
   whether old papers by other lecturers should be included; Windows/WSL needs.
2. Create the repo skeleton (see `references/architecture.md` → *Layout*), `pyproject.toml`
   (rich, prompt_toolkit; dev: pytest, sympy; optional: pypdf), and three docs:
   `SPEC.md` (the user's request verbatim), `CLAUDE.md` (goal, architecture, conventions,
   how to run tests — keep it updated, it is the next agent's map), `REVIEW.md` (decisions).

### Phase 1 — Read and digest the material
Read **everything** before designing. For handwritten material follow
`references/materials.md` (render pages to images, read them, transcribe into a typed
digest `notes/digest/*.md` with page references, cross-check every worked example with a
CAS). The digest becomes the backbone for lessons and the topic list; note the exact
notation and the definitions' wording.

### Phase 2 — Engine (exact or symbolic) + answer checkers
Build the computation core with no UI. Linear algebra could be done with exact
`Fraction` arithmetic; analysis needs a CAS at runtime (see the course playbook). Then the
answer types (`Answer.check(text) → Check(ok, score, msg)`, `typed()`, `show()`,
`fmt_hint`) and the input parser. Details and interfaces: `references/architecture.md`.

### Phase 3 — Topics: lessons, worked examples, generators
One module per chapter, a `Topic` per section in course order, with prerequisites.
Lessons: intuition first, then the definitions/theorems in the course's words, then the
method. Worked examples are the notes' own examples, recomputed by the engine.
Generators produce unlimited exercises with **nice numbers** (design the answer first,
then build the question backwards). Every generator gets a sympy verifier and a
round-trip test (`check(typed()).ok` over many seeds).

### Phase 4 — The bank: official exercises + ingest
Transcribe every exercise sheet / quiz into data files (one TOML per source), each item
with a `task` kind that rebuilds the auto-checked parts from raw data; proofs stay `open`
(self-graded against a model answer). Link handwritten class solutions page by page.
Add `ingest`: hash every PDF (SHA-256), detect new / duplicate / revised files, write a
skeleton source for new ones so they are usable immediately. Formats and the task
registry: `references/architecture.md` → *Bank*.

### Phase 5 — Past papers, exam rules, study plan
Read the exam-rules sheet (duration, parts, points, pass mark, oral, itinere formula) into
a `RULES` table. Each past paper is a bank source with `kind = "exam"`, items carrying
`section` and `points`. `paper <id>` sits it under real conditions; `plan` schedules the
papers until the exam date; `next` gives a due paper priority. Keys: official solutions
when they exist, otherwise computed by you and re-verified in a dedicated test file.
Everything about this phase, including the plan algorithm and the traps: 
`references/papers-and-plan.md`.

### Phase 6 — Profile, `next`, CLI, docs
Profile (JSON, atomic save): mastery per topic (running mean then moving average),
Leitner boxes for spaced repetition, per-item progress, past papers (weight 2× in
mastery), the plan. `next` priority: due paper → due weak topic → overdue review → next
topic in course order (after shaky prerequisites) → unsolved official material → mock
exam. All I/O through `App.ask` / `App.out` so tests can script the whole UI.
README with Windows/WSL setup and every command; REVIEW.md with decisions, limits, next
ideas.

## Course playbooks

- **Analisi Matematica 1** → `references/analisi1.md` (read it in Phase 0: it changes the
  engine choice, the answer types and the exercise kinds).
- Linear algebra / geometry: the reference implementation's choices are summarised inside
  `references/architecture.md` (exact `Fraction` engine, subspace/affine answer types).
- Another course: write a short playbook of the same shape before Phase 2 (answer kinds,
  engine, typical mistakes, exam format) and confirm it with the user.

## Testing (what "done" means)

See `references/testing.md`. In short: independent verifier per generator and per bank
task kind; item-specific checks for hand-keyed answers; a full-marks scripted run of
**every** paper; ingest end-to-end on a temp copy; CLI flows scripted through `App.ask`.
The whole suite should run in about a minute — slow tests get skipped and then rot.

## Lessons learned the hard way (read before Phase 4/5)

- Solution PDFs may **reorder options** (correct one first) — derive keys from the paper's
  own option order, never copy letters blindly. When every key of a version is "A", check
  twice, and say so in the notes for the student.
- Some questions have **two defensible answers** as worded: accept both (`accept`) and
  say which is intended, rather than penalise a correct reading.
- Typos in the material (R³ vs R⁴, a variable renamed mid-question): keep the original,
  state the reading in the item text, list them in REVIEW.md.
- **Unprinted points**: split the exercise points evenly and flag `points_estimated`.
- A label must not leak the method (`X` rather than `M^-1*N`).
- Enumerating divisors by trial up to n hangs on large "generic" values — use √n.
- In verifiers, never reuse a symbol name that the exercise uses as a parameter (`t`).
- Ingest must not classify a whole folder as "solutions" because the *folder* name
  contains the word — look at the file name and its parent.
- Long transcriptions: commit after each paper; render images at low resolution
  (≈60 dpi) to save context, higher only for dense formulas.
