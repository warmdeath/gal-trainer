"""Registry of all topics, in the order of Note_GAL26.pdf."""
from __future__ import annotations

from .base import Gen, Topic
from . import ch0, ch1

TOPICS: list[Topic] = [*ch0.TOPICS, *ch1.TOPICS]
BY_ID: dict[str, Topic] = {t.id: t for t in TOPICS}


def get(topic_id: str) -> Topic:
    return BY_ID[topic_id]
