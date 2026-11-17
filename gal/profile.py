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
        return p

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "topics": {k: asdict(v) for k, v in self.topics.items()},
                "exams": self.exams, "history": self.history[-500:], "exam_date": self.exam_date}
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
               source: str = "practice") -> TopicStats:
        """Register one exercise with score in [0, 1]."""
        now = now or datetime.now()
        score = max(0.0, min(1.0, float(score)))
        st = self.stats(topic_id)
        st.attempts += 1
        if score >= 0.999:
            st.correct += 1
        alpha = max(0.25, 1.0 / st.attempts)  # running mean at first, then moving average
        st.mastery = round(st.mastery + alpha * (score - st.mastery), 4)
        if score >= 0.999:
            st.box = min(st.box + 1, len(INTERVALS) - 1)
        elif score < 0.5:
            st.box = 0
        st.due = (now.date() + timedelta(days=INTERVALS[st.box])).isoformat()
        st.last = now.isoformat(timespec="seconds")
        self.history.append({"t": st.last, "topic": topic_id, "kind": kind, "score": round(score, 3),
                             "source": source})
        return st

    def mark_lesson(self, topic_id: str) -> None:
        self.stats(topic_id).lesson_seen = True

    def add_exam(self, result: dict) -> None:
        self.exams.append(result)

    def days_to_exam(self, today: date | None = None) -> int | None:
        try:
            return (date.fromisoformat(self.exam_date) - (today or date.today())).days
        except ValueError:
            return None
