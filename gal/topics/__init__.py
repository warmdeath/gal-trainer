"""Registry of all topics, in the order of Note_GAL26.pdf (conics and the AI bonus = extras, at the end)."""
from __future__ import annotations

from .base import Gen, Topic
from . import ai, ch0, ch1, ch2, ch3, ch4, ch5, conics

TOPICS: list[Topic] = [*ch0.TOPICS, *ch1.TOPICS, *ch2.TOPICS, *ch3.TOPICS, *ch4.TOPICS,
                       *ch5.TOPICS, *conics.TOPICS, *ai.TOPICS]
BY_ID: dict[str, Topic] = {t.id: t for t in TOPICS}


def get(topic_id: str) -> Topic:
    return BY_ID[topic_id]
