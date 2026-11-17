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
- **Verified against sympy**: every generator and solver is checked in the test suite
  (about 1250 tests).
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
gal> next                 # the trainer decides: a lesson, more practice, a review, or an exam
gal> topics               # all topics, your mastery, when each review is due
gal> learn rank           # lesson + worked examples from the notes, then practice
gal> practice det         # unlimited exercises ('q' to stop); also: p det, or just: det
gal> review               # the topics whose spaced-repetition review is due today
gal> exam                 # 6 exercises, 60 minutes, graded /30; 'exam 3 25' = 3 exercises, 25 min
gal> stats                # progress, weakest topics, exam history, days to the exam
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
  1. weak topics that are due;
  2. mastered topics whose review is overdue;
  3. the next topic of the course, after any shaky prerequisite;
  4. a mock exam once everything is mastered.

## Development

```bash
pip install -e '.[dev]'    # pytest + sympy
pytest                     # ~1250 tests, about 25 s
```

See `CLAUDE.md` for the architecture and `REVIEW.md` for design decisions, simplifications and
ideas for the future.
