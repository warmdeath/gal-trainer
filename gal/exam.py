"""Timed, exam-style mock tests mixing topics (PoliMi GAL written-exam style).

A GAL written exam typically has a few multi-part exercises: a linear system with a
parameter, subspaces / Grassmann, a linear map (ker, Im, matrix), diagonalisation,
Euclidean geometry (projections, distances). The blueprint below mirrors that; one
exam-flagged generator is drawn from each slot. Score is reported out of 30.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .exercise import Exercise
from .profile import Profile
from .topics import BY_ID

BLUEPRINT: list[list[str]] = [
    ["rank", "systems", "inverse", "determinant"],            # systems, parameters, det
    ["sum_intersection", "cart_param", "bases"],               # subspaces
    ["linear_maps", "rep_matrix"],                             # linear maps
    ["diagonalization", "eigen"],                              # endomorphisms
    ["projections", "gram_schmidt", "scalar_products"],        # Euclidean spaces
    ["affine", "cross_distances", "lines_planes", "isometries"],  # geometry
]


@dataclass
class ExamItem:
    exercise: Exercise
    answers: list[str | None] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    msgs: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        n = len(self.exercise.parts)
        return sum(self.scores) / n if n else 0.0


@dataclass
class Exam:
    items: list[ExamItem]
    minutes: int
    started: float = 0.0

    def start(self) -> None:
        self.started = time.monotonic()

    def remaining(self) -> float:
        return self.minutes * 60 - (time.monotonic() - self.started)

    def expired(self) -> bool:
        return self.remaining() <= 0

    def grade30(self) -> float:
        if not self.items:
            return 0.0
        return round(30 * sum(i.score for i in self.items) / len(self.items), 1)


def build_exam(rng: random.Random, prof: Profile | None = None, n: int = 6,
               minutes: int | None = None, topics: list[str] | None = None) -> Exam:
    """One exercise per blueprint slot (cycling if n > slots). Within a slot, weaker topics
    are more likely. `topics` restricts the pool (e.g. only chapters studied so far)."""
    slots = [list(s) for s in BLUEPRINT]
    if topics:
        slots = [[t for t in s if t in topics] for s in slots]
        slots = [s for s in slots if s] or [list(topics)]
    items = []
    order = list(range(len(slots)))
    rng.shuffle(order)
    for k in range(n):
        slot = slots[order[k % len(order)]]
        weights = [1.5 - (prof.stats(t).mastery if prof else 0.0) for t in slot]
        tid = rng.choices(slot, weights=weights)[0]
        ex = BY_ID[tid].random_exercise(rng, exam_only=True)
        items.append(ExamItem(ex))
    return Exam(items, minutes if minutes is not None else 10 * n)
