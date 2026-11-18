# gal-trainer

A terminal trainer for **Geometria e Algebra Lineare** (Politecnico di Milano, course 082747),
built on the official course notes `Note_GAL26.pdf` (Davide Frapporti, 2026-27). It's meant for
a student who missed lectures and needs to catch up before the January exam.

- **26 topics in the order of the notes**, from §0.1 vettori geometrici to §5.6 isometrie, plus
  an extra *conics* topic and a bonus chapter on AI. Every topic has a short lesson: the
  intuition first, then the definitions with the Italian terms (*rango*, *nucleo*,
  *autospazio*, ...). Worked examples are taken from the notes.
- **Unlimited random exercises with nice numbers** (87 generators). You type the answer and
  get feedback right away. On a mistake the trainer names the likely error ("you computed BA
  instead of AB", "you used Fourier coefficients with a non-orthogonal basis", "that's
  det(A − tI), the notes use det(tI − A)", ...) and shows the full step-by-step solution in
  the notation of the notes.
- **Exact arithmetic**: fractions, plus exact square roots such as `12*sqrt(11)/11`. No floats.
- **The course's official practice material, built in**: all 8 *Fogli di esercizi* and 4
  *Quiz* (385 items, 341 auto-checked, the rest self-assessed against a model answer). The
  handwritten *Esercitazioni* are linked page by page to the exercises they solve.
- **37 real past exam papers (2021–2026), sat under exam conditions**: all of this year's
  lecturer's papers (with the official solutions) and the previous lecturer's, transcribed
  question by question (690 items, 589 auto-checked). A timed `paper` runs exactly like the
  written exam (1 h 45, 10 questions + exercises, graded /32), and a **study plan** schedules
  them from now to your exam date.
- **Ready for new material**: drop newly downloaded PDFs next to the notes and run
  `gal ingest`. They are recognised, split into exercises and usable right away.
- **Verified against sympy**: every generator, solver and official answer key is checked in
  the test suite (about 2900 tests).
- **Progress profile**: mastery per topic, spaced repetition of weak topics, and a `next`
  command that picks what to do. Timed **mock exams** mix the topics and are graded out of 30.
- **Bonus — why this matters for AI**: a tiny neural network built from scratch with the
  matrix operations of the course, explained step by step and trained live on XOR.

## Quick start (Linux / macOS)

```bash
git clone <this repo> gal-trainer && cd gal-trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # installs rich and prompt_toolkit, creates the `gal` command
gal                         # or: python -m gal
```

Python 3.10 or newer is required.

## Windows setup

### Option A — WSL (recommended)

WSL gives you a real Linux terminal: Unicode matrices render well and the prompt has history
and arrow keys.

1. Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install
   ```
   Reboot when asked. Ubuntu is installed by default. Open "Ubuntu" from the Start menu and
   choose a Linux username and password.
2. In the Ubuntu terminal:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
   git clone <this repo> ~/gal-trainer && cd ~/gal-trainer
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   gal
   ```
3. Use **Windows Terminal** (preinstalled on Windows 11, free in the Microsoft Store on
   Windows 10). Its default font (Cascadia Mono) shows the brackets ⎡ ⎤ and symbols such as
   ⟨ ⟩ √ ∈ correctly.
4. Next time, open Ubuntu and run:
   ```bash
   cd ~/gal-trainer && source .venv/bin/activate && gal
   ```

Tip: you can reach the WSL files from Windows Explorer at `\\wsl$\Ubuntu\home\<you>`.

### Option B — native Windows

1. Install Python 3.10+ from python.org and tick **"Add python.exe to PATH"**.
2. In **Windows Terminal** (PowerShell):
   ```powershell
   cd $HOME; git clone <this repo> gal-trainer; cd gal-trainer
   py -m venv .venv
   .venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   pip install -e .
   gal
   ```
   If some symbols show up as boxes, choose the "Cascadia Mono" font in the terminal settings.
   The old `cmd.exe` console works but looks worse.

## How to use it

```
gal> next                 # the trainer decides: today's planned paper, a lesson, practice, a review
gal> topics               # all topics, your mastery, when each review is due
gal> learn rank           # lesson + worked examples from the notes, then practice
gal> practice det         # unlimited exercises ('q' to stop); also: p det, or just: det
gal> review               # the topics whose spaced-repetition review is due today
gal> exam                 # 6 exercises, 60 minutes, graded /30; 'exam 3 25' = 3 exercises, 25 min
gal> stats                # progress, weakest topics, official exercises solved, exam history
gal> sheets               # the official Fogli / Quiz / Esercitazioni and your progress
gal> sheet 3              # work through Foglio di esercizi 3 in order (skips what you got right)
gal> quiz                 # 10 random quiz questions (theory drill); 'quiz 2' = Quiz 2 in order
gal> official eigen       # the official exercises of one topic; 'item F02-E4a' = one exercise
gal> plan                 # your study plan: which past paper on which day, until the exam
gal> paper                # sit the paper that is due, in exam conditions (timer, /32)
gal> papers               # all 37 past papers; 'paper P2602AV1' sits a specific one
gal> examinfo             # the rules of the written and oral exam
gal> proofs 7             # the proofs / open questions of Foglio 7, with your grades
gal> grade F07-E13 F07-E14 y F05-E2a p   # grade proofs you did on paper (y / p / n, 0.7, 3/4)
gal> past add             # record a real past exam you did on paper and graded
gal> past                 # your past exams and your "exam readiness" /30
gal> examdate 2027-01-18  # set your exam date
gal> ai                   # bonus chapter: a tiny neural network
```

You can also run a single command straight from the shell: `gal next`, `gal practice gram`,
`gal --seed 7 exam 2 20`.

### Typing answers

| answer type        | example                                              |
|--------------------|------------------------------------------------------|
| number             | `-3/2`, `0.5`, `2*(1-3)`                              |
| exact real         | `12/sqrt(11)`, `2sqrt(3)/3`, `√2`                     |
| vector             | `(1, -2, 3/2)` or `1 -2 3/2`                          |
| matrix             | `1 2; 3 4` (rows separated by `;`)                    |
| basis / vectors    | `(1,0,1); (0,1,1)`, or `none` for {0}                 |
| solution set       | `none`, `(1,2,0)`, `(1,0,0) + t(1,1,0) + s(0,0,1)`    |
| equations          | `x - 2z - 3w = 0; y + 2z + 2w = 0` (any equivalent system) |
| polynomial         | `t^3 - 4t^2 + 5t - 2` or `(t-1)^2(t-2)`               |
| yes / no, choice   | `y`, `no`, `sì`; `a`, `b`, ...                        |

Type `?` for the expected format, `skip` to see the solution, `q` to stop.
Input that can't be parsed is never counted as a mistake.

Any correct form is accepted: a different basis of the same subspace, another point or
direction of a line, an equivalent system of Cartesian equations, any matrix P with
P⁻¹AP diagonal, or rescaled Gram-Schmidt vectors.

### Progress, spaced repetition and `next`

Your progress is saved in `~/.gal_trainer/profile.json`. Set `GAL_TRAINER_HOME` or use
`--profile` to keep it somewhere else.

- **Mastery** per topic: for the first exercises it's the mean of your scores, after that a
  moving average. A topic counts as *mastered* with mastery ≥ 75%, at least 3 exercises, and
  a last answer that wasn't a failure.
- **Spaced repetition** (Leitner boxes): a fully correct answer moves the topic up a box, so
  the next review comes later (0, 1, 2, 4, 7, 14 days). A wrong answer sends it back to box 0,
  due again today.
- **`next`** picks, in this order:
  0. the past paper your study plan has due (see below);
  1. weak topics that are due;
  2. mastered topics whose review is overdue;
  3. the next topic of the course, after any shaky prerequisite;
  4. once everything is mastered: the official sheets/quizzes you haven't solved yet,
     then a mock exam.

`practice` also mixes in, about 3 times in 10, an official exercise of that topic you
haven't solved yet. Mock exams sometimes include a real past-exam exercise from the sheets
(*"Dal tema d'esame ..."*), with all its parts chained as on the real paper.

### Proofs and real past exams (graded by you)

Proofs can't be checked automatically, so you grade them:
- in the moment, while working through a sheet (`y` / `p` / `n` after the model answer), or
- afterwards, in one go. Write them on paper, then enter the grades, either one per exercise
  or the same grade for a whole series:

  ```
  grade F07-E13 F07-E14 y F05-E2a p Q01-C1 3/4
  ```

  An id without a letter (`F01-E2`) means all its sub-items. An id without a grade
  (`grade F07-E13`) first shows the exercise and the model answer, then asks for the grade.
  Grades can be `y` (1), `p` (½), `n` (0), `0.7`, `70%` or points like `3/4`.

**Real past exams** (*temi d'esame*) show the real exam format, so they weigh **2×** an
ordinary result. Do the paper on paper, grade it yourself (or have another agent grade it),
then run `past add`. It asks for:
- the paper's label, for example `tema 07/02/2024`;
- the date you did it, who graded it, and the minutes you used;
- for each exercise, its topics (`systems, rank`) and your points (`5/8`).

Each exercise then updates the mastery of its topics with weight 2. A weak exercise sends
its topics back to review today, so `next` picks them up. If you only have the total, use
`past add tema giugno 2025 27/33`: it counts in the stats but leaves topic mastery
unchanged. `stats` and `past` show an **exam readiness** grade: the weighted mean of your
last 5 mock and past exams, with past papers counting 2×. `past rm 3` deletes record 3.

## Past exam papers and the study plan

The exam (from *Informazioni sull'esame*, `Informazioni_esame_GAL.pdf`): a **written exam**
of 1 h 45 min with 10 multiple-choice questions (1 point each) and open exercises (22 points),
32 points in total; **16 points** admit you to the **oral** (definitions, examples,
counterexamples, theorems and proofs on the whole course). The optional *prova in itinere*
(60 min, 5 questions + 1 exercise, 16 points, chapters 1–3) counts at the first winter
appello as max(PC, ⅔(PI + PC)). Type `examinfo` for the summary.

`papers` lists the 37 transcribed papers from `Esami_anni_precedenti-<date>/` and
`Esami_A.A._202627-<date>/`:
- **this year's lecturer** (2025-26 and the itinere of 30/10/2026, 13 papers): keys
  checked against the official *Soluzioni*;
- **the previous lecturer** (2021–2025, 24 sittings, Parte A + Parte B merged into one
  paper): no official solutions, so every key was worked out and the computable ones are
  re-verified with sympy in `tests/test_paper_keys.py`.

`paper <id>` sits one under exam conditions. The timer runs in the toolbar and there is no
feedback until you hand in. Write proofs and explanations on paper and press Enter. At the
end you grade them against the model answer (as strictly as the examiner: clarity counts).
The result shows Parte I and Parte II, the total /32 (itinere papers /16), and whether it
would admit you to the oral. It is recorded as a real paper (weight 2× in topic mastery)
and counts in your exam readiness. `paper P2602AV1 untimed` removes the clock.

**The study plan** (`plan`) spreads the papers over the days left until your exam date
(`examdate`):
- **twice a week** (Wednesday and Saturday), then **every other day in the last two weeks**,
  with the last paper two days before the exam;
- **60-minute itinere papers first** (chapters 1–3, while you catch up on the rest);
- then the previous lecturer's most recent appelli;
- then this lecturer's own appelli, the most recent one last as the dress rehearsal.

`next` and the start-up screen tell you when a paper is due. A missed day stays due until
you sit it, and `plan rebuild` re-spreads the remaining papers from today (e.g. after you
change the exam date). Your own October 2026 itinere is in the plan as practice: re-taking
it shows how far you have come.

## Official course material

The PDFs downloaded from the course website live next to `Note_GAL26.pdf`:

| folder                     | what                                   | in the trainer |
|----------------------------|----------------------------------------|----------------|
| `Esercizi-<date>/`         | *Fogli di esercizi* 1–8                | `sheet 1` … `sheet 8` (F01–F08) |
| `Quiz-<date>/`             | *Quiz* 1–4 (multiple choice + *compiti*) | `quiz`, `quiz 1` … (Q01–Q04) |
| `Esercitazioni-<date>/`    | handwritten class solutions            | linked from each exercise (ES01–ES07) |
| `Esami_anni_precedenti-<date>/`, `Esami_A.A._<year>-<date>/` | past exam papers (+ *Soluzioni*) | `papers`, `paper <id>`, `plan` (P…) |

Every exercise was transcribed by hand into `gal/bank/data/*.toml`. Computational ones
(systems, ranks, bases, Grassmann, matrices of maps, eigenvalues, "al variare di λ"
discussions, ...) are checked exactly, like the generated exercises, with the same mistake
diagnosis. Proofs and "give an example" questions are **self-assessed**: you write your
answer (on paper or in the terminal), the trainer shows the model answer, and you grade
yourself `y` (right), `p` (partly) or `n`. When an exercise was solved in class, the
solution shows where: *"Solved in class (handwritten): Esercitazione 2, p. 4 →
Esercitazioni-20261117/2026-10-07 Esercitazione 2.pdf"*.

### Adding new material (new sheets, quizzes, past exams)

1. Download the folder from the course website and unzip it next to `Note_GAL26.pdf`, for
   example `Esercizi-20261201/`. Folders named `Esercizi*`, `Esercitazioni*`, `Quiz*`,
   `Esami*`, `Temi*` or `material/` are scanned. Re-downloading a whole folder is fine:
   files you already have are recognised by their content (SHA-256) and ignored.
2. Run

   ```bash
   gal ingest
   ```

   The trainer also tells you at startup when it sees PDFs it doesn't know yet.
   For each new PDF it extracts the text, splits it at *Esercizio N* / *Domanda N* /
   *Compito N*, guesses the topic of each item, and writes a new source file
   `gal/bank/data/F09.toml` (sheets `F..`, quizzes `Q..`, esercitazioni `ES..`, others `X..`).
   The new exercises are usable immediately, as self-assessed items (`sheet 9`).
3. To make them auto-checked, give each item a `task` and its `data`: see "Adding course
   material" in `CLAUDE.md` (or ask Claude Code to "transcribe F09"). Run `pytest`
   afterwards: every answer is verified with sympy.

Text extraction needs either `pdftotext` (Ubuntu/WSL: `sudo apt install poppler-utils`) or
the `pypdf` package (`pip install -e '.[ingest]'`). If a PDF you had already ingested is
replaced by a revised version with the same name, `gal ingest` reports it as *changed*:
check the differences, then accept it with `gal ingest --rehash`.

## Development

```bash
pip install -e '.[dev]'    # pytest + sympy
pytest                     # ~2900 tests, under a minute
```

See `CLAUDE.md` for the architecture and `REVIEW.md` for design decisions, simplifications and
ideas for the future.
