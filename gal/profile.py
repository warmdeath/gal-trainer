"""Persistent progress profile: mastery per topic + spaced repetition (Leitner boxes).

Stored as JSON in $GAL_TRAINER_HOME/profile.json (default ~/.gal_trainer/).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

INTERVALS = [0, 1, 2, 4, 7, 14]  # days until the next review, per Leitner box
MASTERED = 0.75                  # mastery needed to call a topic "mastered"
MIN_ATTEMPTS = 3                 # ... and at least this many exercises
PREREQ_OK = 0.6                  # mastery of a prerequisite needed to move on
DEFAULT_EXAM_DATE = "2027-01-18"
PAST_WEIGHT = 2.0                # a real past exam counts like 2 ordinary results (exam format)
SELF_GRADES = {"y": 1.0, "yes": 1.0, "s": 1.0, "si": 1.0, "sì": 1.0, "p": 0.5, "partly": 0.5,
               "n": 0.0, "no": 0.0}


def parse_grade(txt: str) -> float | None:
    """y / p / n, a fraction of 1 (0.7), a percentage (70%) or points (5/8) -> score in [0, 1]."""
    t = txt.strip().lower()
    if t in SELF_GRADES:
        return SELF_GRADES[t]
    try:
        if t.endswith("%"):
            v = float(t[:-1]) / 100
        elif "/" in t:
            a, b = t.split("/", 1)
            v = float(a) / float(b)
        else:
            v = float(t.replace(",", "."))
    except (ValueError, ZeroDivisionError):
        return None
    return v if 0.0 <= v <= 1.0 else None


def home_dir() -> Path:
    return Path(os.environ.get("GAL_TRAINER_HOME", Path.home() / ".gal_trainer"))


@dataclass
class TopicStats:
    mastery: float = 0.0
    attempts: int = 0
    correct: int = 0
    box: int = 0
    due: str | None = None       # ISO date of the next review
    last: str | None = None      # ISO datetime of the last exercise
    lesson_seen: bool = False

    @property
    def started(self) -> bool:
        return self.attempts > 0

    @property
    def mastered(self) -> bool:
        """High mastery, enough attempts, and the latest answer was not a failure (box 0)."""
        return self.mastery >= MASTERED and self.attempts >= MIN_ATTEMPTS and self.box >= 1

    def is_due(self, today: date) -> bool:
        return self.started and (self.due is None or date.fromisoformat(self.due) <= today)


@dataclass
class Profile:
    topics: dict[str, TopicStats] = field(default_factory=dict)
    exams: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    exam_date: str = DEFAULT_EXAM_DATE
    bank: dict[str, dict] = field(default_factory=dict)   # official items: {id: {attempts, best, last}}
    past_exams: list[dict] = field(default_factory=list)  # real past papers, graded by hand
    plan: list[dict] = field(default_factory=list)        # scheduled timed papers [{date, paper, done}]
    path: Path | None = None

    # ------------------------------------------------------------ persistence
    @staticmethod
    def load(path: Path | None = None) -> "Profile":
        path = path or home_dir() / "profile.json"
        if not path.exists():
            return Profile(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            backup = path.with_suffix(".corrupt.json")
            path.replace(backup)
            return Profile(path=path)
        p = Profile(path=path)
        p.topics = {k: TopicStats(**v) for k, v in raw.get("topics", {}).items()}
        p.exams = raw.get("exams", [])
        p.history = raw.get("history", [])
        p.exam_date = raw.get("exam_date", DEFAULT_EXAM_DATE)
        p.bank = raw.get("bank", {})
        p.past_exams = raw.get("past_exams", [])
        p.plan = raw.get("plan", [])
        return p

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "topics": {k: asdict(v) for k, v in self.topics.items()},
                "exams": self.exams, "history": self.history[-500:], "exam_date": self.exam_date,
                "bank": self.bank, "past_exams": self.past_exams, "plan": self.plan}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".profile", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ updates
    def stats(self, topic_id: str) -> TopicStats:
        if topic_id not in self.topics:
            self.topics[topic_id] = TopicStats()
        return self.topics[topic_id]

    def record(self, topic_id: str, score: float, kind: str = "", now: datetime | None = None,
               source: str = "practice", weight: float = 1.0) -> TopicStats:
        """Register one exercise with score in [0, 1]. `weight` w > 1 moves mastery as much as
        w identical results would (alpha_w = 1 - (1 - alpha)^w), but counts as one attempt."""
        now = now or datetime.now()
        score = max(0.0, min(1.0, float(score)))
        st = self.stats(topic_id)
        st.attempts += 1
        if score >= 0.999:
            st.correct += 1
        alpha = max(0.25, 1.0 / st.attempts)  # running mean at first, then moving average
        if weight != 1.0:
            alpha = 1.0 - (1.0 - alpha) ** weight if alpha < 1.0 else 1.0
        st.mastery = round(st.mastery + alpha * (score - st.mastery), 4)
        if score >= 0.999:
            st.box = min(st.box + 1, len(INTERVALS) - 1)
        elif score < 0.5:
            st.box = 0
        st.due = (now.date() + timedelta(days=INTERVALS[st.box])).isoformat()
        st.last = now.isoformat(timespec="seconds")
        entry = {"t": st.last, "topic": topic_id, "kind": kind, "score": round(score, 3), "source": source}
        if weight != 1.0:
            entry["weight"] = weight
        self.history.append(entry)
        return st

    def record_item(self, item_id: str, score: float, now: datetime | None = None) -> dict:
        """Progress on one official exercise (also counted in the topic's mastery by the caller)."""
        now = now or datetime.now()
        st = self.bank.setdefault(item_id, {"attempts": 0, "best": 0.0, "last": None})
        st["attempts"] += 1
        st["best"] = round(max(st["best"], float(score)), 3)
        st["last"] = now.isoformat(timespec="seconds")
        return st

    def item_done(self, item_id: str) -> bool:
        return self.bank.get(item_id, {}).get("best", 0.0) >= 0.999

    def mark_lesson(self, topic_id: str) -> None:
        self.stats(topic_id).lesson_seen = True

    def add_exam(self, result: dict) -> None:
        self.exams.append(result)

    def add_past_exam(self, paper: str, exercises: list[dict], taken: str | None = None,
                      grader: str = "self", minutes: float | None = None, total: tuple | None = None,
                      now: datetime | None = None, **extra) -> dict:
        """A real past paper done and graded outside the trainer (by the student or another agent).
        exercises: [{"topics": [ids], "got": float, "max": float}] (may be empty if only the total
        is known: total = (got, max)). Each exercise feeds the mastery of its topics with weight
        PAST_WEIGHT, because past papers show the real exam format."""
        now = now or datetime.now()
        got = sum(e["got"] for e in exercises) if exercises else float(total[0])
        mx = sum(e["max"] for e in exercises) if exercises else float(total[1])
        if mx <= 0 or not 0 <= got <= mx:
            raise ValueError("points must satisfy 0 <= got <= max, max > 0")
        rec = {"paper": paper, "taken": taken or now.date().isoformat(), "grader": grader,
               "points": round(got, 2), "max": round(mx, 2), "grade": round(30 * got / mx, 1),
               "exercises": exercises, "recorded": now.isoformat(timespec="seconds")}
        if minutes is not None:
            rec["minutes_used"] = minutes
        rec.update(extra)
        for e in exercises:
            for tid in e.get("topics", []):
                self.record(tid, e["got"] / e["max"], "past-exam", now=now, source="past-exam",
                            weight=PAST_WEIGHT)
        self.past_exams.append(rec)
        return rec

    def papers_done(self) -> set[str]:
        return {e["paper_id"] for e in self.past_exams if e.get("paper_id")}

    def readiness(self, last: int = 5) -> tuple[float, int] | None:
        """Weighted mean grade /30 of the latest `last` exam results: mock exams weight 1, real
        past papers PAST_WEIGHT. Returns (grade, number of results used) or None."""
        res = [(e.get("date", ""), float(e["grade"]), 1.0) for e in self.exams if "grade" in e]
        res += [(e.get("taken", ""), float(e["grade"]), PAST_WEIGHT) for e in self.past_exams]
        res = sorted(res, key=lambda r: r[0])[-last:]
        if not res:
            return None
        w = sum(r[2] for r in res)
        return round(sum(r[1] * r[2] for r in res) / w, 1), len(res)

    def days_to_exam(self, today: date | None = None) -> int | None:
        try:
            return (date.fromisoformat(self.exam_date) - (today or date.today())).days
        except ValueError:
            return None
