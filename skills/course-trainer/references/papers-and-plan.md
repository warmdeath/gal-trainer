# Past papers, exam simulation and the study plan

## Papers are bank sources
A paper is `kind = "exam"` with `paper = {format, minutes, points, lecturer, version, half,
points_estimated}`; items add `section` ("I" test, "II" exercises, "III" theory if the
exam has one) and `points`. Everything built for sheets (checkers, verifiers, per-item
progress, `item <id>`) then works for papers too.

- One source per *sitting*: merge "Parte A" and "Parte B" files of the same date; keep
  versions V1/V2 as separate papers (they reorder options and change numbers).
- The solutions PDF goes into the paper's `files` with `role = "solutions"` (shared by V1
  and V2 is fine). Delete ingest skeletons of the files you merged.
- Points: use the printed points; when not printed, split the exercise total evenly over
  the sub-questions and set `points_estimated = true`. Test the totals for the current
  lecturer's papers (e.g. 32 for an appello, 16 for an itinere).
- Sub-questions share a statement: put it in `setup` on the first sub-item; paper mode
  prints it once, practice mode shows it with every sub-item.
- Questions mixing computation and explanation: auto-check the computation, put the
  explanation in an `open` sibling or in the model answer.

## `paper <id>` (exam conditions)
Intro panel (rules, number of questions, total points, notes such as estimated points or
older lecturer) → Enter starts the clock (toolbar countdown) → each question shows its
points; answers are stored silently; `skip` leaves blank, `q` hands in; answers arriving
after time are refused → self-assessed parts were "written on paper" during the run and
are graded at the end against the model answer (y / p / n / 0.7 / 3/4) → report: per
question points, Parte I / II totals, total on the paper's scale (/32, /16), verdict
(admitted to the oral?), time used. Record it like a real past paper (weight 2× in topic
mastery, `paper_id`), update per-item progress, mark the plan session done.
Offer `paper <id> untimed` for study mode.

## Study plan
```
session_dates(start, exam):
    2 sessions/week (e.g. Wed + Sat) until exam − 14 days,
    then every other day, the last one 2 days before the exam (rest + theory)
paper_order(bank):
    first-half papers (itinere 1) first — this lecturer's, then older ones, newest first;
    then the full papers: older lecturers' newest first, then this lecturer's own
    appelli in date order (the most recent = final dress rehearsal)
make_plan: first third of the slots = itinere papers; fill the rest with full papers,
    always keeping all of this lecturer's appelli; drop the oldest older papers if there
    are more papers than slots (they stay available via `papers`).
```
`plan` shows the calendar (✓ result / TODAY / overdue); missed sessions stay due (don't
silently move them); `plan rebuild` re-spreads the remaining papers from today and keeps
done ones. `next` and the startup screen announce the due paper before anything else. If
the student already sat a paper for real, keep it in the plan as a re-take: progress on a
known paper is motivating and measurable.

## Hand-entered results
`past add` (label, date, grader self/agent, minutes, per exercise: topics + got/max, or
total only) for papers done outside the trainer; `grade <ids> <y|p|n>` for proofs done on
paper; `proofs [sheet]` lists open items. A total-only record counts for readiness but not
for topic mastery (don't guess where points were lost). Exam readiness = weighted mean of
the last 5 mock/real papers (real papers 2×).
