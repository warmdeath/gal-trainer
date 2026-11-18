# Reading the course material (typed and handwritten)

## Inventory first
List every file with page counts (`pdfinfo`) and classify it: notes, exercise sheet, quiz,
class solutions (esercitazioni), exam paper, exam solutions, exam-rules sheet. Unzip each
archive into its own new folder; keep the original names (they carry dates and versions:
`2026-02 Appello - V1.pdf`, `Parte A/B`, `versione A/B`, `Soluzioni/`).

## Typed PDFs
`pdftotext -layout` keeps columns; it garbles matrices, fractions and column vectors. Use
the text for prose, the page image for every formula. Multiple-choice keys in official
solutions are often marked typographically (e.g. a crossed box rendered as `C@`): extract
them with `scripts/mc_keys.py`, then confirm by computing (see "keys" below).

## Handwritten notes and solutions
1. Render pages: `python scripts/render_pages.py notes.pdf out_dir --dpi 60` (one PNG per
   page) and `--contact 6` for contact sheets of 6 thumbnails to map a long file quickly.
   60 dpi is enough to read most handwriting and saves context; go to 90–110 dpi only for
   dense formulas.
2. Map before transcribing: one line per page in `notes/digest/INDEX.md`
   (`p.12 — limiti notevoli, esempi 3-5`). This tells you the topic order and lets you
   link pages later (the reference project linked each class-solution page to the sheet
   items it solves, and showed "Solved in class: …, p. 4 → <pdf>").
3. Transcribe into `notes/digest/<chapter>.md`: definitions and theorems verbatim (they
   are what the oral exam asks), notation exactly as written, worked examples with their
   page. Mark anything illegible `[??]` and ask the user instead of guessing.
4. Cross-check every worked example and every numeric result with the CAS. Handwriting
   slips are common (a sign, an exponent); when the notes contain an error, keep the
   correct value in the trainer and record the discrepancy in REVIEW.md.
5. Do not OCR-and-trust. OCR of tablet handwriting is unreliable; reading the images
   yourself is slower but correct.

## Keys and solutions
- Official solutions: use them, but derive each multiple-choice key against the *paper's*
  option order — some solution files list the correct option first.
- No solutions (e.g. older lecturers): solve every question yourself; for each computable
  one add a sympy check to `tests/test_paper_keys.py` (ranks, limits, integrals,
  convergence, coordinates …).
- Typos and ambiguities: keep the original text, state your reading in the item, accept
  both answers when both are defensible, list them in REVIEW.md.

## Exam-rules sheet
Extract into `papers.RULES`: duration, parts (test / exercises / theory), points per part,
pass mark for the oral, partial exams (itinere) and how they combine, allowed material.
Show it with an `examinfo` command and use it for timing and grading papers.
