"""Real past exam papers (temi d'esame): exam rules, scoring in points, the study plan.

The rules come from "Informazioni sull'esame" (Informazioni_esame_GAL.pdf, A.A. 2026/27):
  * prova scritta: 1 h 45 min, 10 multiple-choice questions (1 point each) + open exercises
    (22 points) = 32 points; at least 16 points give access to the prova teorica (oral);
  * prova in itinere (optional): 60 min, 5 multiple-choice questions + 1 exercise = 16 points,
    on the first half of the course (chapters 1-3); at the first winter appello the written
    score is max(PC, 2/3 (PI + PC)).
Older papers (another lecturer, 2021-2025) have the same two parts (Parte A: test, Parte B:
exercises with printed points); they are timed with the current rules and their score is
also reported scaled to 32.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

RULES = {
    "appello": {"minutes": 105, "points": 32, "mcq": 10, "exercises": 22,
                "label": "Prova scritta (appello): 1 h 45 min, 10 domande (10 pt) + esercizi (22 pt) = 32 pt"},
    "itinere": {"minutes": 60, "points": 16, "mcq": 5, "exercises": 11,
                "label": "Prova in itinere: 60 min, 5 domande (5 pt) + 1 esercizio (11 pt) = 16 pt"},
}
ORAL_THRESHOLD = 16        # points (out of 32) needed to be admitted to the prova teorica
ORAL_NOTE = ("Prova teorica (orale): definizioni, esempi, controesempi, teoremi e dimostrazioni "
             "su tutto il programma; incide sul voto finale.")


def written_score(pc: float, pi: float | None = None) -> float:
    """Score of the written exam at the first winter appello, given the itinere score PI."""
    return pc if pi is None else max(pc, 2 * (pi + pc) / 3)


def papers(bank) -> list:
    """Sources that are exam papers, newest first."""
    return sorted((s for s in bank.sources.values() if s.kind == "exam" and s.items),
                  key=lambda s: (s.date, s.id), reverse=True)


def paper_format(src) -> str:
    return (src.paper or {}).get("format", "appello")


def paper_minutes(src) -> int:
    return int((src.paper or {}).get("minutes", RULES[paper_format(src)]["minutes"]))


def max_points(src) -> float:
    return float(sum(i.points or 0 for i in src.items))


def full_points(src) -> float:
    """What the paper is worth on its own scale (32 for an appello, 16 for an itinere)."""
    return float((src.paper or {}).get("points", RULES[paper_format(src)]["points"]))


@dataclass
class ItemScore:
    item: object
    fraction: float = 0.0          # in [0, 1]
    answered: bool = False
    note: str = ""

    @property
    def points(self) -> float:
        return round((self.item.points or 0) * self.fraction, 2)


@dataclass
class PaperResult:
    source: object
    scores: list[ItemScore] = field(default_factory=list)
    minutes_used: float = 0.0

    @property
    def points(self) -> float:
        return round(sum(s.points for s in self.scores), 2)

    @property
    def max(self) -> float:
        return max_points(self.source)

    @property
    def scaled(self) -> float:
        """Points on the paper's own full scale (32 or 16)."""
        mx = self.max
        return round(full_points(self.source) * self.points / mx, 1) if mx else 0.0

    def section_points(self, section: str) -> tuple[float, float]:
        sc = [s for s in self.scores if (s.item.section or "") == section]
        return round(sum(s.points for s in sc), 2), float(sum(s.item.points or 0 for s in sc))

    def verdict(self) -> str:
        if paper_format(self.source) == "itinere":
            return f"{self.scaled}/16 in the prova in itinere format."
        ok = self.scaled >= ORAL_THRESHOLD
        return (f"{self.scaled}/32: " + ("admitted to the prova teorica (≥ 16)." if ok
                                         else "not yet admitted to the prova teorica (< 16)."))


# ------------------------------------------------------------------ study plan
FINAL_WINDOW = 14          # last days before the exam: a paper every other day
WEEKDAYS = (2, 5)          # before that: Wednesday and Saturday
ITINERE_SHARE = 1 / 3      # the first third of the sessions use 60-minute itinere papers


def session_dates(start: date, exam: date) -> list[date]:
    """2 per week (Wed, Sat) until FINAL_WINDOW days before the exam, then every other day,
    ending two days before the exam (the last day is for rest and the theory)."""
    out: list[date] = []
    window = exam - timedelta(days=FINAL_WINDOW)
    d = start
    while d < window:
        if d.weekday() in WEEKDAYS:
            out.append(d)
        d += timedelta(days=1)
    d = exam - timedelta(days=2)
    tail = []
    while d >= max(window, start):
        tail.append(d)
        d -= timedelta(days=2)
    return out + sorted(tail)


def paper_half(src) -> int:
    """1 = first half of the course (an 'itinere 1'), 2 = second half / whole course."""
    meta = src.paper or {}
    if "half" in meta:
        return int(meta["half"])
    t = (src.title + " " + src.id).lower()
    if paper_format(src) == "itinere" and not any(k in t for k in ("itinere 2", "itinere ii", "i2")):
        return 1
    return 2


def _own(src) -> bool:
    return (src.paper or {}).get("lecturer") == "Frapporti"


def paper_order(bank) -> tuple[list, list]:
    """(itinere-style papers on the first half, full papers), each in order of preference.
    Itinere: this year's first (a re-take shows progress), then the lecturer's previous ones,
    then the older lecturer's, newest first. Full papers: the lecturer's own, oldest to newest
    (the most recent appello is the final dress rehearsal), preceded by the older lecturer's
    papers, newest first."""
    ps = papers(bank)                      # newest first
    first = [s for s in ps if paper_half(s) == 1]
    itinere = [s for s in first if _own(s)] + [s for s in first if not _own(s)]
    rest = [s for s in ps if paper_half(s) != 1]
    old = [s for s in rest if not _own(s)]
    own = sorted([s for s in rest if _own(s)], key=lambda s: (paper_format(s) != "itinere", s.date, s.id))
    return itinere, old + own


def make_plan(bank, start: date, exam: date, done: set[str] = frozenset()) -> list[dict]:
    dates = session_dates(start, exam)
    itinere, full = paper_order(bank)
    itinere = [s for s in itinere if s.id not in done]
    full = [s for s in full if s.id not in done]
    n_it = min(len(itinere), round(len(dates) * ITINERE_SHARE))
    n_full = len(dates) - n_it
    own = [s for s in full if _own(s)]
    old = [s for s in full if not _own(s)]
    own = own[-n_full:] if n_full < len(own) else own          # the lecturer's latest papers always fit
    old = old[:max(0, n_full - len(own))]                      # then the most recent older ones
    chosen = itinere[:n_it] + list(reversed(old)) + own        # older papers chronologically, then own
    if len(chosen) < len(dates):     # not enough papers: fewer sessions, keep the latest dates
        dates = dates[len(dates) - len(chosen):]
    return [{"date": d.isoformat(), "paper": s.id, "done": False} for d, s in zip(dates, chosen)]


def due_session(plan: list[dict], today: date) -> dict | None:
    for s in plan:
        if not s.get("done") and date.fromisoformat(s["date"]) <= today:
            return s
    return None


def next_session(plan: list[dict], today: date) -> dict | None:
    for s in plan:
        if not s.get("done") and date.fromisoformat(s["date"]) >= today:
            return s
    return None
