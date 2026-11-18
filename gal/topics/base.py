"""Topic = lesson + worked examples + exercise generators."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..exercise import Exercise


@dataclass
class Gen:
    key: str
    title: str
    fn: Callable[[random.Random], Exercise]
    exam: bool = False  # used in mock exams
    weight: float = 1.0

    def __call__(self, rng: random.Random) -> Exercise:
        ex = self.fn(rng)
        ex.kind = self.key
        return ex


@dataclass
class Topic:
    id: str
    ref: str          # section of Note_GAL26.pdf, e.g. "§1.2.2"
    title: str        # English
    title_it: str     # Italian, as in the notes
    chapter: str
    prereqs: list[str]
    lesson: list[tuple[str, str]]          # (heading, body)
    examples: list[Callable[[], Exercise]]  # worked examples, mostly from the notes
    gens: list[Gen] = field(default_factory=list)
    extra: bool = False  # not (yet) in the notes

    def gen(self, key: str) -> Gen:
        for g in self.gens:
            if g.key == key:
                return g
        raise KeyError(key)

    def random_exercise(self, rng: random.Random, exam_only: bool = False) -> Exercise:
        pool = [g for g in self.gens if g.exam or not exam_only] or self.gens
        g = rng.choices(pool, weights=[g.weight for g in pool])[0]
        ex = g(rng)
        ex.topic = self.id
        return ex
