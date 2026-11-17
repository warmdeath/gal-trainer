"""Terminal UI (rich for output, prompt_toolkit for input).

All I/O goes through App.ask / App.out so the whole UI is testable with scripted input.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import date
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .core.parse import ParseError
from .exam import Exam, build_exam
from .exercise import ChoiceAns, Exercise
from .planner import advise
from .profile import MASTERED, Profile, home_dir
from .topics import BY_ID, TOPICS, Topic

QUIT = {"q", "quit", "exit", "menu", "back", ":q"}
SKIP = {"skip", "s?", "pass", "salta"}
HINT = {"?", "hint", "help", "format"}

BANNER = """\
GAL trainer — Geometria e Algebra Lineare (Politecnico di Milano)
Course notes: Note_GAL26.pdf. Exact arithmetic: type fractions like -3/2, roots like sqrt(2).
Type 'help' for the commands, 'next' to let the trainer choose."""

HELP = """\
Commands
  next                 the trainer picks what to study (spaced repetition + course order)
  topics               list all topics with your mastery
  learn <topic>        lesson + worked examples, then practice
  practice <topic>     unlimited random exercises ('q' to stop)       (alias: p)
  review               practise the topics that are due today
  exam [n] [minutes]   timed mock exam with n exercises (default 6, 10 min each)
  stats                progress report, exam history, days to the exam
  examdate YYYY-MM-DD  set the date of your exam
  ai                   bonus chapter: why this matters for AI (a tiny neural network)
  reset                delete your progress
  quit
<topic> can be its number in 'topics', its id (e.g. det, rank) or a prefix of the id.

While answering
  ?        show the expected answer format       skip   show the solution (counts as wrong)
  q        stop practising
Answer formats: number 3/4, root 2*sqrt(3)/3, vector (1, -2, 0), matrix 1 2; 3 4,
several vectors (1,0,1); (0,1,1), parametric (1,0,0) + t(1,1,0) + s(0,0,1), 'none',
equations x - 2z = 0; y + w = 1, polynomial t^3 - 4t^2 + 5t - 2, yes/no, letter a/b/c."""

ALIASES = {"det": "determinant", "inv": "inverse", "gs": "gram_schmidt", "diag": "diagonalization",
           "proj": "projections", "grassmann": "sum_intersection", "maps": "linear_maps",
           "eig": "eigen", "vec": "vectors", "sys": "systems", "rep": "rep_matrix"}


def resolve_topic(arg: str) -> Topic | None:
    a = arg.strip().lower()
    if not a:
        return None
    if a.isdigit() and 1 <= int(a) <= len(TOPICS):
        return TOPICS[int(a) - 1]
    if a in BY_ID:
        return BY_ID[a]
    if a in ALIASES:
        return BY_ID[ALIASES[a]]
    matches = [t for t in TOPICS if t.id.startswith(a)] or [t for t in TOPICS if a in t.id or a in t.title.lower()
                                                           or a in t.title_it.lower()]
    return matches[0] if len(matches) >= 1 else None


def reflow(text: str) -> str:
    """Undo the hard wrapping of lesson paragraphs so the terminal can wrap them itself.
    Kept on their own line: blank lines, bullets/markers, lines indented by 3+ spaces
    (formulas, aligned tables). Joined to the previous line: plain continuation lines and
    bullet continuations indented by exactly 2 spaces."""
    out: list[str] = []
    prev_joinable = False
    in_bullet = False
    for line in text.split("\n"):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        marker = stripped[:1] in "•-–" or stripped[:2].rstrip(")").isdigit() and stripped[1:2] in ").:" \
            or stripped.startswith(("Step", "Attenzione", "Check", "Output", "Input", "Passo", "Controllo"))
        if not stripped:
            out.append("")
            prev_joinable = in_bullet = False
            continue
        # inside a bullet only 2-space continuations join; outside, plain lines join
        cont = prev_joinable and not marker and (indent == 2 if in_bullet else indent == 0)
        if cont:
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(line)
            in_bullet = marker and stripped[:1] in "•-–"
        prev_joinable = indent <= 2 and not stripped.endswith(":")
    return "\n".join(out)


def bar(x: float, width: int = 10) -> str:
    k = round(max(0.0, min(1.0, x)) * width)
    return "█" * k + "·" * (width - k)


class App:
    def __init__(self, profile: Profile, console: Console | None = None,
                 ask: Callable[[str], str] | None = None, rng: random.Random | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.prof = profile
        self.con = console or Console(highlight=False)
        self._ask = ask
        self.rng = rng or random.Random()
        self.clock = clock
        self._session = None
        self.toolbar: Callable[[], str] | None = None

    # ------------------------------------------------------------ I/O
    def out(self, s="", style: str | None = None) -> None:
        if isinstance(s, str):
            self.con.print(Text(s, style=style or ""))
        else:
            self.con.print(s)

    def ask(self, label: str) -> str:
        if self._ask is not None:
            return self._ask(label)
        if self._session is None and sys.stdin.isatty():
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            home_dir().mkdir(parents=True, exist_ok=True)
            self._session = PromptSession(history=FileHistory(str(home_dir() / "history.txt")))
        if self._session is None:
            try:
                return input(label)
            except EOFError:
                return "quit"
        try:
            tb = self.toolbar
            return self._session.prompt(label, bottom_toolbar=tb, refresh_interval=1.0 if tb else 0)
        except (EOFError, KeyboardInterrupt):
            return "quit"

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        self.out(Panel(Text(BANNER), title="GAL trainer", border_style="cyan"))
        days = self.prof.days_to_exam()
        if days is not None and days >= 0:
            self.out(f"{days} days to your exam ({self.prof.exam_date}).  Change it with: examdate YYYY-MM-DD",
                     "dim")
        while True:
            line = self.ask("\ngal> ").strip()
            if not line:
                continue
            if not self.command(line):
                break
        self.out("Arrivederci! Progress saved.", "cyan")

    def command(self, line: str) -> bool:
        """Execute one command; False means quit."""
        cmd, _, arg = line.partition(" ")
        cmd = cmd.lower()
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "h", "?"):
            self.out(HELP)
        elif cmd in ("topics", "list", "ls"):
            self.show_topics()
        elif cmd in ("learn", "lesson", "l"):
            t = self._topic_arg(arg)
            if t:
                self.learn(t)
        elif cmd in ("practice", "p", "train"):
            t = self._topic_arg(arg)
            if t:
                self.practice(t)
        elif cmd == "next":
            self.next()
        elif cmd == "review":
            self.review()
        elif cmd == "exam":
            nums = [int(x) for x in arg.split() if x.isdigit()]
            n = nums[0] if nums else 6
            minutes = nums[1] if len(nums) > 1 else None
            self.exam(max(1, min(n, 12)), minutes)
        elif cmd in ("stats", "progress"):
            self.show_stats()
        elif cmd == "examdate":
            try:
                date.fromisoformat(arg.strip())
                self.prof.exam_date = arg.strip()
                self.prof.save()
                self.out(f"Exam date set to {arg.strip()}.")
            except ValueError:
                self.out("Use the format YYYY-MM-DD, e.g. examdate 2027-01-18", "red")
        elif cmd == "ai":
            from .ai.chapter import run_chapter
            run_chapter(self)
        elif cmd == "reset":
            if self.ask("Delete ALL progress? type 'yes' to confirm: ").strip().lower() == "yes":
                path = self.prof.path
                self.prof.__init__(path=path)
                self.prof.save()
                self.out("Progress reset.")
        else:
            t = resolve_topic(line)
            if t:
                self.practice(t)
            else:
                self.out(f"Unknown command '{cmd}'. Type 'help'.", "red")
        return True

    def _topic_arg(self, arg: str) -> Topic | None:
        if not arg.strip():
            self.out("Which topic? e.g. 'practice det' — see 'topics'.", "yellow")
            return None
        t = resolve_topic(arg)
        if t is None:
            self.out(f"No topic matches '{arg}'. See 'topics'.", "red")
        return t

    # ------------------------------------------------------------ views
    def show_topics(self) -> None:
        wide = self.con.width >= 110
        tab = Table(title="Topics (order of Note_GAL26.pdf)", show_lines=False, pad_edge=False)
        tab.add_column("#", justify="right")
        tab.add_column("id", no_wrap=True)
        tab.add_column("topic")
        tab.add_column("mastery", no_wrap=True)
        tab.add_column("done", justify="right")
        tab.add_column("review", no_wrap=True)
        today = date.today()
        chapter = None
        for i, t in enumerate(TOPICS, 1):
            if t.chapter != chapter:
                chapter = t.chapter
                tab.add_row("", "", Text(t.chapter, style="bold cyan"), "", "", "")
            st = self.prof.stats(t.id)
            due = "" if not st.started else ("now" if st.is_due(today) else st.due or "")
            mark = " ✓" if st.mastered else ""
            title = Text(f"{t.ref} {t.title_it}", style="bold")
            if wide:
                title.append(f" — {t.title}", style="dim")
            tab.add_row(str(i), t.id, title, f"{bar(st.mastery, 8)} {st.mastery:4.0%}{mark}", str(st.attempts), due)
        self.out(tab)

    def show_stats(self) -> None:
        started = [t for t in TOPICS if self.prof.stats(t.id).started]
        mastered = [t for t in TOPICS if self.prof.stats(t.id).mastered]
        core = [t for t in TOPICS if not t.extra]
        days = self.prof.days_to_exam()
        lines = [f"Topics started: {len(started)}/{len(TOPICS)},  mastered: {len(mastered)}/{len(core)} core topics."]
        if days is not None:
            lines.append(f"Exam date {self.prof.exam_date}: {days} day(s) left.")
            left = [t for t in core if not self.prof.stats(t.id).mastered]
            if days > 0 and left:
                lines.append(f"Pace: about {len(left) / max(days, 1):.1f} topic(s) per day to master the rest.")
        total = sum(self.prof.stats(t.id).attempts for t in TOPICS)
        good = sum(self.prof.stats(t.id).correct for t in TOPICS)
        if total:
            lines.append(f"Exercises: {total}, fully correct: {good} ({good / total:.0%}).")
        weak = sorted((t for t in started), key=lambda t: self.prof.stats(t.id).mastery)[:5]
        if weak:
            lines.append("Weakest: " + ", ".join(f"{t.id} ({self.prof.stats(t.id).mastery:.0%})" for t in weak))
        self.out(Panel(Text("\n".join(lines)), title="Progress", border_style="green"))
        if self.prof.exams:
            tab = Table(title="Mock exams")
            for c in ("date", "exercises", "grade /30", "minutes used"):
                tab.add_column(c)
            for e in self.prof.exams[-10:]:
                tab.add_row(e.get("date", ""), str(e.get("n", "")), str(e.get("grade", "")), str(e.get("minutes_used", "")))
            self.out(tab)

    # ------------------------------------------------------------ learning
    def learn(self, t: Topic) -> None:
        self.out(Rule(f"{t.ref}  {t.title_it}  —  {t.title}"))
        if t.extra:
            self.out("(Extra topic: not in the official notes yet.)", "yellow")
        for k, (head, body) in enumerate(t.lesson):
            self.out(Panel(Text(reflow(body)), title=head, border_style="blue"))
            if k < len(t.lesson) - 1 and self._pause():
                return
        for k, make in enumerate(t.examples, 1):
            if self._pause("Enter: worked example, q: stop "):
                return
            ex = make()
            self.out(Panel(Text(ex.prompt + self._choices(ex)), title=f"Worked example {k}", border_style="magenta"))
            self.out(Panel(Text("\n".join(ex.solution) + "\n\nAnswer: " + ex.answer_text()),
                           title="Solution", border_style="green"))
        self.prof.mark_lesson(t.id)
        self.prof.save()
        if self.ask("Start practising this topic? [Y/n] ").strip().lower() not in {"n", "no"} | QUIT:
            self.practice(t)

    def _pause(self, label="Enter: continue, q: stop ") -> bool:
        return self.ask(label).strip().lower() in QUIT

    @staticmethod
    def _choices(ex: Exercise) -> str:
        # choice options are already rendered in prompts; nothing to add
        return ""

    def next(self) -> None:
        adv = advise(self.prof)
        days = self.prof.days_to_exam()
        tail = f"  ({days} days to the exam)" if days is not None and days >= 0 else ""
        self.out(Panel(Text(adv.reason + tail), title=f"Next: {adv.action}" +
                       (f" — {adv.topic.title}" if adv.topic else ""), border_style="cyan"))
        if self.ask("Go? [Y/n] ").strip().lower() in {"n", "no"} | QUIT:
            return
        if adv.action == "exam":
            self.exam(6, None)
        elif adv.action == "learn":
            self.learn(adv.topic)
        else:
            self.practice(adv.topic)

    def review(self) -> None:
        today = date.today()
        due = [t for t in TOPICS if self.prof.stats(t.id).is_due(today)]
        if not due:
            self.out("Nothing is due today. Try 'next'.")
            return
        due.sort(key=lambda t: self.prof.stats(t.id).mastery)
        self.out("Due today: " + ", ".join(t.id for t in due))
        for t in due:
            ex = t.random_exercise(self.rng)
            res = self.solve_exercise(ex, t)
            if res is None:
                return

    # ------------------------------------------------------------ exercises
    def practice(self, t: Topic) -> None:
        st = self.prof.stats(t.id)
        if not st.lesson_seen and not st.started:
            self.out(f"Tip: 'learn {t.id}' shows the lesson first.", "dim")
        self.out(Rule(f"Practice: {t.title_it} ({t.ref}) — 'q' to stop"))
        n = 0
        while True:
            ex = t.random_exercise(self.rng)
            n += 1
            res = self.solve_exercise(ex, t, number=n)
            if res is None:
                break
            st = self.prof.stats(t.id)
            self.out(f"Mastery {bar(st.mastery)} {st.mastery:.0%}" + ("  ✓ mastered!" if st.mastered else ""), "cyan")
            if self.ask("Enter: next exercise, q: stop ").strip().lower() in QUIT:
                break

    def solve_exercise(self, ex: Exercise, t: Topic, number: int | None = None) -> float | None:
        """Ask every part; returns the score in [0,1], or None if the student quit."""
        title = f"Exercise {number}" if number else "Exercise"
        self.out(Panel(Text(ex.prompt), title=f"{title} · {t.id} · {ex.kind}", border_style="magenta"))
        scores, wrong_msgs = [], []
        for part in ex.parts:
            label = (part.label + ": ") if len(ex.parts) > 1 or part.label != "Answer" else "> "
            while True:
                txt = self.ask(label).strip()
                low = txt.lower()
                if low in QUIT:
                    return None
                if low in HINT:
                    self.out("Format: " + part.answer.fmt_hint, "yellow")
                    continue
                if low in SKIP:
                    scores.append(0.0)
                    wrong_msgs.append(f"{part.label}: skipped — answer {part.answer.show()}")
                    break
                try:
                    res = part.answer.check(txt)
                except ParseError as e:
                    self.out(f"I can't read that: {e}. Format: {part.answer.fmt_hint}  (type ? for help)", "yellow")
                    continue
                if res.ok:
                    self.out("✓ correct" + (f" — {res.msg}" if res.msg else ""), "green")
                    scores.append(1.0)
                else:
                    self.out("✗ not correct." + (f" Likely error: {res.msg}" if res.msg else ""), "red")
                    scores.append(res.score)
                    wrong_msgs.append(f"{part.label}: {res.msg}" if res.msg else part.label)
                break
        score = sum(scores) / len(scores)
        if score < 0.999:
            body = "\n".join(ex.solution) + "\n\nCorrect answer:\n" + ex.answer_text()
            self.out(Panel(Text(body), title="Step-by-step solution", border_style="yellow"))
        self.prof.record(t.id, score, ex.kind)
        self.prof.save()
        return score

    # ------------------------------------------------------------ exam
    def exam(self, n: int = 6, minutes: int | None = None) -> None:
        studied = [t.id for t in TOPICS if self.prof.stats(t.id).started]
        pool = None
        if len(studied) >= 4 and len(studied) < len(TOPICS) - 2:
            if self.ask("Only topics you have started? [Y/n] ").strip().lower() not in ("n", "no"):
                pool = studied
        ex = build_exam(self.rng, self.prof, n, minutes, pool)
        self.out(Panel(Text(f"{n} exercises, {ex.minutes} minutes. No feedback until the end.\n"
                            "Type 'skip' to leave an answer blank, 'q' to hand in early."),
                       title="Mock exam", border_style="red"))
        if self.ask("Press Enter to start the clock ").strip().lower() in QUIT:
            return
        ex.start()
        t0 = self.clock()
        self.toolbar = lambda: f" Time left: {_mmss(ex.remaining())} "
        handed_in = False
        try:
            for k, item in enumerate(ex.items, 1):
                if ex.expired() or handed_in:
                    break
                e = item.exercise
                self.out(Panel(Text(e.prompt), title=f"Exam {k}/{n} · {_mmss(ex.remaining())} left",
                               border_style="red"))
                for part in e.parts:
                    label = (part.label + ": ") if len(e.parts) > 1 or part.label != "Answer" else "> "
                    while True:
                        if ex.expired():
                            self.out("⏰ Time is up!", "red")
                            break
                        txt = self.ask(label).strip()
                        if txt.lower() in QUIT:
                            handed_in = True
                            break
                        if txt.lower() in SKIP or not txt:
                            item.answers.append(None)
                            item.scores.append(0.0)
                            item.msgs.append("blank")
                            break
                        if txt.lower() in HINT:
                            self.out("Format: " + part.answer.fmt_hint, "yellow")
                            continue
                        try:
                            res = part.answer.check(txt)
                        except ParseError as err:
                            self.out(f"I can't read that: {err}", "yellow")
                            continue
                        if ex.expired():
                            self.out("⏰ Time is up — this answer arrived too late.", "red")
                            break
                        item.answers.append(txt)
                        item.scores.append(res.score if not res.ok else 1.0)
                        item.msgs.append(res.msg)
                        break
                    if handed_in or ex.expired():
                        break
        finally:
            self.toolbar = None
        used = round((self.clock() - t0) / 60, 1)
        self._exam_report(ex, used)

    def _exam_report(self, ex: Exam, used: float) -> None:
        tab = Table(title=f"Mock exam result: {ex.grade30()}/30")
        for c in ("#", "topic", "score", "notes"):
            tab.add_column(c)
        for k, item in enumerate(ex.items, 1):
            n = len(item.exercise.parts)
            sc = sum(item.scores)
            note = "; ".join(m for m in item.msgs if m) if item.msgs else "not reached"
            tab.add_row(str(k), BY_ID[item.exercise.topic].title, f"{sc:.1f}/{n}", note[:80])
        self.out(tab)
        verdict = ("passed (≥ 18/30)" if ex.grade30() >= 18 else "not yet passing (< 18/30)")
        self.out(f"Time used: {used} min of {ex.minutes}. Verdict: {verdict}.", "bold")
        for item in ex.items:
            self.prof.record(item.exercise.topic, item.score, item.exercise.kind, source="exam")
        self.prof.add_exam({"date": date.today().isoformat(), "n": len(ex.items), "grade": ex.grade30(),
                            "minutes_used": used,
                            "topics": [i.exercise.topic for i in ex.items],
                            "scores": [round(i.score, 3) for i in ex.items]})
        self.prof.save()
        while True:
            a = self.ask("Review a solution: number (Enter to finish) ").strip()
            if not a or a.lower() in QUIT:
                break
            if a.isdigit() and 1 <= int(a) <= len(ex.items):
                e = ex.items[int(a) - 1].exercise
                given = ex.items[int(a) - 1].answers
                self.out(Panel(Text(e.prompt), title=f"Exercise {a}"))
                self.out(Panel(Text("\n".join(e.solution) + "\n\nCorrect answer:\n" + e.answer_text() +
                                    "\n\nYour answers: " + ", ".join(x or "(blank)" for x in given)),
                               title="Solution", border_style="green"))


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gal", description="GAL trainer (Politecnico di Milano)")
    ap.add_argument("command", nargs="*", help="optional command to run directly, e.g. 'next' or 'practice det'")
    ap.add_argument("--seed", type=int, default=None, help="random seed (reproducible exercises)")
    ap.add_argument("--profile", default=None, help="path of the profile JSON")
    ns = ap.parse_args(argv)
    from pathlib import Path
    prof = Profile.load(Path(ns.profile) if ns.profile else None)
    app = App(prof, rng=random.Random(ns.seed))
    try:
        if ns.command:
            app.command(" ".join(ns.command))
        else:
            app.run()
    except KeyboardInterrupt:
        pass
    finally:
        prof.save()
    return 0
