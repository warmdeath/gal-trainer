"""Generic checks for EVERY exercise generator:
- generation is deterministic for a given seed and never crashes;
- the model answer, typed back through the parser, is accepted (round trip);
- prompt and solution are non-empty.
Independent sympy verification lives in tests/test_verify_*.py."""
import random

import pytest

from gal.topics import TOPICS

ALL = [(t, g) for t in TOPICS for g in t.gens]
SEEDS = range(40)


@pytest.mark.parametrize("topic,gen", ALL, ids=[f"{t.id}:{g.key}" for t, g in ALL])
def test_round_trip(topic, gen):
    for seed in SEEDS:
        ex = gen(random.Random(seed))
        assert ex.prompt.strip() and ex.solution
        for part in ex.parts:
            typed = part.answer.typed()
            res = part.answer.check(typed)
            assert res.ok, (topic.id, gen.key, seed, part.label, typed, res.msg)
        ex2 = gen(random.Random(seed))
        assert ex2.prompt == ex.prompt


@pytest.mark.parametrize("topic", TOPICS, ids=[t.id for t in TOPICS])
def test_worked_examples_and_lesson(topic):
    assert topic.lesson and topic.gens
    for make in topic.examples:
        ex = make()
        for part in ex.parts:
            assert part.answer.check(part.answer.typed()).ok
