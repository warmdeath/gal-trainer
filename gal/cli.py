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

from .core.fmt import reflow  # noqa: F401  (re-exported for tests/chapters)
from .core.parse import ParseError
from .exam import Exam, build_exam
from .exercise import ChoiceAns, Exercise
from .planner import advise
from .profile import MASTERED, Profile, home_dir
from .topics import BY_ID, TOPICS, Topic

QUIT = {"q", "quit", "exit", "menu", "back", ":q"}
SKIP = {"skip", "s?", "pass", "salta"}
HINT = {"?", "hint", "help", "format"}
OFFICIAL_SHARE = 0.3   # in 'practice', share of exercises taken from the official sheets (if unsolved ones exist)

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

Official course material (Fogli, Quiz, Esercitazioni)
  sheets               list the official sheets/quizzes with your progress   (alias: material)
  sheet <n>            work through Foglio di esercizi n, in order (skips what you already got right)
  quiz [n]             theory drill: 10 random quiz questions, or Quiz n in order
  official <topic>     the official exercises of one topic
  item <id>            one official exercise, e.g. item F02-E4a
  ingest [--rehash]    add newly downloaded PDFs (put the folders next to Note_GAL26.pdf)

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
        self.official_share = OFFICIAL_SHARE

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
        self.material_notice()
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
        elif cmd in ("sheets", "material", "bank", "fogli"):
            self.show_sources()
        elif cmd in ("sheet", "foglio", "source"):
            self.work_source(arg, default_prefix="F")
        elif cmd == "quiz":
            if arg.strip():
                self.work_source(arg, default_prefix="Q")
            else:
                self.quiz_drill()
        elif cmd == "official":
            t = self._topic_arg(arg)
            if t:
                self.official(t)
        elif cmd == "item":
            self.do_item_id(arg.strip())
        elif cmd == "ingest":
            self.run_ingest(rehash="--rehash" in arg)
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
        try:
            bank = self._bank()
        except Exception:
            bank = None
        if bank and bank.items:
            done = sum(1 for i in bank.items if self.prof.item_done(i))
            tried = sum(1 for i in bank.items if i in self.prof.bank)
            per = []
            for src in sorted(bank.sources.values(), key=lambda s: s.id):
                if src.items and src.kind in ("sheet", "quiz"):
                    per.append(f"{src.id} {sum(1 for i in src.items if self.prof.item_done(i.id))}/{len(src.items)}")
            lines.append(f"Official exercises: {done}/{len(bank.items)} solved ({tried} attempted).  " + "  ".join(per))
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
        elif adv.action == "official":
            self.work_source(adv.source or "")
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
            ex = None
            if self.official_share and self.rng.random() < self.official_share:
                ex = self._official_exercise(t)
            if ex is None:
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
            if getattr(part.answer, "self_assessed", False):
                if self.ask("Work it out on paper (or type your answer), then press Enter to see the model answer ") \
                        .strip().lower() in QUIT:
                    return None
                self.out(Panel(Text(part.answer.show() + ("\n\n" + "\n".join(ex.solution) if ex.solution else "")),
                               title="Model answer", border_style="green"))
                label = "Did you get it right? y = yes, p = partly, n = no: "
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
        self_assessed = all(getattr(p.answer, "self_assessed", False) for p in ex.parts)
        if score < 0.999 and not self_assessed:
            body = "\n".join(ex.solution) + "\n\nCorrect answer:\n" + ex.answer_text()
            self.out(Panel(Text(body), title="Step-by-step solution", border_style="yellow"))
        self.prof.record(t.id, score, ex.kind, source="bank" if "item_id" in ex.data else "practice")
        if "item_id" in ex.data:
            self.prof.record_item(ex.data["item_id"], score)
        self.prof.save()
        return score

    # ------------------------------------------------------------ official material (bank)
    def _bank(self):
        from .bank import get_bank
        return get_bank()

    def material_notice(self) -> None:
        """Tell the student when new PDFs have been dropped in but not ingested yet."""
        try:
            from .bank.ingest import pending
            new = pending()
        except Exception:  # never block the trainer because of the material folders
            return
        if new:
            names = ", ".join(p.name for p in new[:4]) + (" ..." if len(new) > 4 else "")
            self.out(Panel(Text(f"New course material found ({len(new)} PDF): {names}\n"
                                "Type 'ingest' to add it to the practice bank."),
                           title="New material", border_style="yellow"))

    def run_ingest(self, rehash: bool = False) -> None:
        from .bank import reload_bank
        from .bank.ingest import ingest
        try:
            rep = ingest(rehash=rehash)
        except Exception as e:
            self.out(f"Ingest failed: {e}", "red")
            return
        self.out(Panel(Text(rep.text()), title="Ingest", border_style="cyan"))
        reload_bank()

    def show_sources(self) -> None:
        bank = self._bank()
        tab = Table(title="Official course material")
        for c in ("id", "title", "items", "auto-checked", "done ✓", "status"):
            tab.add_column(c)
        for src in sorted(bank.sources.values(), key=lambda s: ({"sheet": 0, "quiz": 1, "esercitazione": 2}.get(s.kind, 3), s.id)):
            if src.kind == "esercitazione":
                linked = sum(len(s["items"]) for s in src.solutions)
                tab.add_row(src.id, src.title, "—", "—", "—", f"handwritten, {linked} solutions linked")
                continue
            done = sum(1 for i in src.items if self.prof.item_done(i.id))
            tab.add_row(src.id, src.title, str(len(src.items)), str(sum(i.auto for i in src.items)),
                        f"{done}/{len(src.items)}", src.status)
        self.out(tab)
        self.out("Work through one with: sheet 2  ·  quiz 3  ·  quiz (random theory drill)  ·  official <topic>", "dim")

    def _official_exercise(self, t: Topic):
        """A not-yet-solved auto-checked official item of topic t, or None."""
        from .bank import TaskError, build
        items = [i for i in self._bank().by_topic(t.id, auto_only=True) if not self.prof.item_done(i.id)]
        self.rng.shuffle(items)
        for it in items:
            try:
                return build(it)
            except TaskError:
                continue
        return None

    def _do_items(self, items, title: str) -> None:
        from .bank import TaskError, build
        if not items:
            self.out("Nothing to do here.", "yellow")
            return
        self.out(Rule(f"{title} — {len(items)} exercise(s), 'q' to stop"))
        for k, it in enumerate(items, 1):
            try:
                ex = build(it)
            except TaskError as e:
                self.out(f"{it.id}: cannot be auto-checked ({e}); see the PDF.", "yellow")
                continue
            topic = BY_ID.get(it.topic) or TOPICS[0]
            if self.solve_exercise(ex, topic, number=k) is None:
                return
            if k < len(items) and self.ask("Enter: next, q: stop ").strip().lower() in QUIT:
                return
        self.out("Done!", "green")

    def work_source(self, arg: str, default_prefix: str = "F") -> None:
        bank = self._bank()
        ref = arg.strip()
        src = bank.resolve_source(ref) or (bank.resolve_source(default_prefix + ref) if ref.isdigit() else None)
        if src is None:
            self.out(f"No source '{ref}'. See 'sheets'.", "red")
            return
        if src.kind == "esercitazione":
            lines = [f"{src.title}: handwritten solutions — open {src.files[0]['path'] if src.files else ''}"]
            for s in src.solutions:
                lines.append(f"  p. {', '.join(map(str, s['pages']))}: {', '.join(s['items'])}")
            self.out(Panel(Text("\n".join(lines)), title=src.id))
            return
        todo = [i for i in src.items if not self.prof.item_done(i.id)]
        if not todo:
            if self.ask(f"You already solved all of {src.title}. Redo everything? [y/N] ").strip().lower() not in ("y", "yes", "s", "si", "sì"):
                return
            todo = list(src.items)
        self._do_items(todo, src.title)

    def quiz_drill(self, n: int = 10) -> None:
        items = [i for i in self._bank().of_kind("quiz") if i.task == "mcq"]
        fresh = [i for i in items if not self.prof.item_done(i.id)]
        self.rng.shuffle(fresh)
        pick = fresh[:n]
        if len(pick) < n:
            rest = [i for i in items if i not in pick]
            self.rng.shuffle(rest)
            pick += rest[:n - len(pick)]
        self._do_items(pick, "Quiz drill (official multiple-choice questions)")

    def official(self, t: Topic) -> None:
        items = self._bank().by_topic(t.id)
        todo = [i for i in items if not self.prof.item_done(i.id)] or items
        self._do_items(todo, f"Official exercises: {t.title_it}")

    def do_item_id(self, iid: str) -> None:
        bank = self._bank()
        it = bank.items.get(iid) or bank.items.get(iid.upper())
        if it is None:
            self.out(f"No item '{iid}'.", "red")
            return
        self._do_items([it], it.title())

    def _past_exam_exercises(self, pool):
        """Past-exam exercises of the official sheets, as chained multi-part exercises."""
        from .bank import past_exam_exercises
        return past_exam_exercises(self._bank(), pool)

    # ------------------------------------------------------------ exam
    def exam(self, n: int = 6, minutes: int | None = None) -> None:
        studied = [t.id for t in TOPICS if self.prof.stats(t.id).started]
        pool = None
        if len(studied) >= 4 and len(studied) < len(TOPICS) - 2:
            if self.ask("Only topics you have started? [Y/n] ").strip().lower() not in ("n", "no"):
                pool = studied
        ex = build_exam(self.rng, self.prof, n, minutes, pool, past=self._past_exam_exercises(pool))
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
            owners = item.exercise.data.get("part_items")
            if owners:   # a past-exam exercise: credit each official sub-item it chains
                per: dict[str, list[float]] = {}
                for k, iid in enumerate(owners):
                    per.setdefault(iid, []).append(item.scores[k] if k < len(item.scores) else 0.0)
                for iid, sc in per.items():
                    self.prof.record_item(iid, sum(sc) / len(sc))
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
