"""`next`: what should the student practise now?

Priority:
 1. spaced review: a started topic that is due and still weak (lowest mastery first);
 2. a mastered topic whose review is overdue by >= OVERDUE_DAYS days;
 3. the first topic in course order that is not mastered — but if one of its prerequisites
    is shaky, go back to the weakest such prerequisite first;
 4. a due review of a mastered topic;
 5. everything mastered: the official sheets/quizzes not yet solved, then a timed mock exam.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .profile import MASTERED, PREREQ_OK, Profile
from .topics import BY_ID, TOPICS, Topic

OVERDUE_DAYS = 2


@dataclass
class Advice:
    action: str            # "learn" | "practice" | "review" | "official" | "exam"
    topic: Topic | None
    reason: str
    source: str | None = None   # for "official": the sheet/quiz to work through


def _weak_prereq(t: Topic, prof: Profile, seen=None) -> Topic | None:
    """Deepest prerequisite (in course order) with mastery below PREREQ_OK."""
    seen = seen or set()
    for pid in t.prereqs:
        if pid in seen:
            continue
        seen.add(pid)
        p = BY_ID[pid]
        deeper = _weak_prereq(p, prof, seen)
        if deeper:
            return deeper
        if prof.stats(pid).mastery < PREREQ_OK:
            return p
    return None


def advise(prof: Profile, today: date | None = None, include_extra: bool = False) -> Advice:
    today = today or date.today()
    core = [t for t in TOPICS if include_extra or not t.extra]
    due_weak = [t for t in core if prof.stats(t.id).is_due(today) and not prof.stats(t.id).mastered]
    if due_weak:
        t = min(due_weak, key=lambda t: (prof.stats(t.id).mastery, TOPICS.index(t)))
        st = prof.stats(t.id)
        return Advice("review", t, f"Spaced repetition: '{t.title}' is due and your mastery is "
                                   f"{st.mastery:.0%} (needs {MASTERED:.0%}).")
    overdue = [t for t in core if prof.stats(t.id).mastered and prof.stats(t.id).due
               and (today - date.fromisoformat(prof.stats(t.id).due)).days >= OVERDUE_DAYS]
    if overdue:
        t = min(overdue, key=lambda t: prof.stats(t.id).due)
        return Advice("review", t, f"'{t.title}' was mastered but its review is overdue since "
                                   f"{prof.stats(t.id).due}: refresh it before it fades.")
    for t in core:
        st = prof.stats(t.id)
        if st.mastered:
            continue
        weak = _weak_prereq(t, prof)
        if weak is not None:
            ws = prof.stats(weak.id)
            act = "learn" if not ws.started else "practice"
            return Advice(act, weak, f"'{t.title}' builds on '{weak.title}', where your mastery is "
                                     f"{ws.mastery:.0%}: strengthen it first.")
        if not st.started:
            return Advice("learn", t, f"Next new topic in the course order ({t.ref}).")
        return Advice("practice", t, f"Keep practising: mastery {st.mastery:.0%} "
                                     f"after {st.attempts} exercise(s).")
    due = [t for t in core if prof.stats(t.id).is_due(today)]
    if due:
        t = min(due, key=lambda t: prof.stats(t.id).mastery)
        return Advice("review", t, "Everything is mastered; this one is due for a refresh.")
    official = _unsolved_official(prof)
    if official:
        sid, title, n = official
        return Advice("official", None, f"All topics are mastered: now the official material. {title} still has "
                                        f"{n} exercise(s) you have not solved.", source=sid)
    return Advice("exam", None, "All topics mastered and nothing is due: time for a timed mock exam!")


def _unsolved_official(prof: Profile):
    """(source id, title, #unsolved) of the first sheet/quiz with unsolved items, or None."""
    try:
        from .bank import get_bank
        bank = get_bank()
    except Exception:
        return None
    for src in sorted(bank.sources.values(), key=lambda s: ({"sheet": 0, "quiz": 1}.get(s.kind, 9), s.id)):
        if src.kind not in ("sheet", "quiz"):
            continue
        todo = [i for i in src.items if not prof.item_done(i.id)]
        if todo:
            return src.id, src.title, len(todo)
    return None
