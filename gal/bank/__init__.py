"""The practice bank: the course's official material (exercise sheets, quizzes, handwritten
esercitazioni) transcribed as data files, one TOML file per source in gal/bank/data/.

Every item has a `task`; auto-checked tasks are built by gal.bank.tasks, `open` items are
self-assessed against a model answer. New PDFs are turned into skeleton sources by
`gal ingest` (gal/bank/ingest.py) and can then be upgraded item by item.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from ..exercise import Exercise, Part
from .tasks import TaskError, build_parts

DATA_DIR = Path(__file__).parent / "data"


def bank_dir() -> Path:
    return Path(os.environ.get("GAL_BANK_DIR", DATA_DIR))


def material_root() -> Path:
    """Folder that contains the downloaded material folders (default: the repository root)."""
    env = os.environ.get("GAL_MATERIAL_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2]


@dataclass
class Item:
    id: str
    ref: str
    topic: str
    task: str
    text: str
    data: dict
    source: "Source"
    page: int | None = None
    exam: str = ""       # date of the past exam this exercise comes from ('' if none)
    points: float | None = None   # exam papers: points of this item
    section: str = ""             # exam papers: "I" (test) / "II" (exercises)
    setup: str = ""               # shared statement of the exercise (inherited by its sub-items)
    solved_in: list[tuple["Source", list[int]]] = field(default_factory=list)  # (esercitazione, pages)

    @property
    def auto(self) -> bool:
        return self.task != "open"

    @property
    def from_exam(self) -> bool:
        """Exercises the sheets mark as taken from a past exam ('Dal tema d'esame ...')."""
        return bool(self.exam) or "tema d'esame" in self.text.lower()

    @property
    def exercise_key(self) -> str:
        """'F06-E4c' -> 'F06-E4': the sub-items of one exercise share it."""
        m = re.match(r"^(.*?-[A-Z]+\d+)", self.id)
        return m.group(1) if m else self.id

    def title(self) -> str:
        return f"{self.source.title} — {self.ref}"


@dataclass
class Source:
    id: str
    title: str
    kind: str            # sheet | quiz | esercitazione | exam
    files: list[dict]    # [{path, sha256, pages}]
    date: str = ""
    status: str = "transcribed"   # transcribed | skeleton
    note: str = ""
    items: list[Item] = field(default_factory=list)
    solutions: list[dict] = field(default_factory=list)  # for esercitazioni: [{item, pages}]
    paper: dict = field(default_factory=dict)   # exam papers: format, minutes, points, lecturer, ...


@dataclass
class Bank:
    sources: dict[str, Source]
    items: dict[str, Item]

    def by_topic(self, topic_id: str, auto_only: bool = False) -> list[Item]:
        return [i for i in self.items.values() if i.topic == topic_id and (i.auto or not auto_only)]

    def of_kind(self, kind: str) -> list[Item]:
        return [i for s in self.sources.values() if s.kind == kind for i in s.items]

    def resolve_source(self, ref: str) -> Source | None:
        r = ref.strip().upper().replace(" ", "")
        if r in self.sources:
            return self.sources[r]
        # 'foglio 2', 'f2', 'quiz 3', 'q3', 'es 4'
        import re
        m = re.match(r"^(F|FOGLIO|SHEET|Q|QUIZ|E|ES|ESERCITAZIONE)0*(\d+)$", r)
        if m:
            pref = {"F": "F", "FOGLIO": "F", "SHEET": "F", "Q": "Q", "QUIZ": "Q",
                    "E": "ES", "ES": "ES", "ESERCITAZIONE": "ES"}[m.group(1)]
            key = f"{pref}{int(m.group(2)):02d}"
            return self.sources.get(key)
        return None


def load_bank(path: Path | None = None) -> Bank:
    path = path or bank_dir()
    sources: dict[str, Source] = {}
    items: dict[str, Item] = {}
    for f in sorted(path.glob("*.toml")):
        raw = tomllib.loads(f.read_text(encoding="utf-8"))
        s = raw["source"]
        src = Source(s["id"], s["title"], s["kind"], s.get("files", []), s.get("date", ""),
                     s.get("status", "transcribed"), s.get("note", ""), paper=s.get("paper", {}))
        if src.id in sources:
            raise ValueError(f"duplicate source id {src.id} in {f.name}")
        setups: dict[str, str] = {}
        for it in raw.get("item", []):
            item = Item(it["id"], it.get("ref", ""), it.get("topic", ""), it.get("task", "open"),
                        it.get("text", "").strip(), it.get("data", {}), src, it.get("page"),
                        str(it.get("exam", "")), it.get("points"), str(it.get("section", "")))
            if item.id in items:
                raise ValueError(f"duplicate item id {item.id}")
            if it.get("setup"):
                setups[item.exercise_key] = it["setup"].strip()
            item.setup = setups.get(item.exercise_key, "")
            items[item.id] = item
            src.items.append(item)
        src.solutions = raw.get("solution", [])
        sources[src.id] = src
    for src in sources.values():
        for sol in src.solutions:
            for iid in sol["items"]:
                if iid in items:
                    items[iid].solved_in.append((src, sol.get("pages", [])))
    return Bank(sources, items)


def build(item: Item, with_setup: bool = True) -> Exercise:
    """with_setup=False leaves out the shared statement (paper mode prints it once)."""
    extra, parts, steps = build_parts(item.task, item.data, item.text)
    keep = item.data.get("keep")
    if keep:   # ask only the parts whose label starts with one of these prefixes
        parts = [(l, a) for l, a in parts if any(l.startswith(k) for k in keep)]
        if not parts:
            raise TaskError(f"{item.id}: 'keep' {keep} matches no part")
    from ..core.fmt import reflow
    head = reflow(item.setup) + "\n\n" if item.setup and with_setup else ""
    prompt = f"[{item.title()}]\n{head}{reflow(item.text)}"
    if extra:
        prompt += "\n\n" + extra
    if not item.auto and item.solved_in:
        steps = steps + [solved_in_text(item)]
    elif item.solved_in:
        steps = steps + ["", solved_in_text(item)]
    ex = Exercise(item.topic, f"bank:{item.task}", prompt, [Part(lbl, a) for lbl, a in parts], steps,
                  dict(item.data, item_id=item.id))
    return ex


def past_exam_exercises(bank: Bank, topics: list[str] | None = None) -> list[Exercise]:
    """One chained multi-part Exercise per past-exam exercise of the sheets, like on the real
    paper: the shared set-up text, then every auto-checked sub-question (labelled a), b), ...).
    Self-assessed sub-questions (proofs) are left out: a mock exam is graded automatically."""
    groups: dict[str, list[Item]] = {}
    for it in bank.items.values():
        if it.from_exam:
            groups.setdefault(it.exercise_key, []).append(it)
    out = []
    for key, its in groups.items():
        auto = [i for i in its if i.auto]
        if not auto or (topics is not None and not any(i.topic in topics for i in auto)):
            continue
        first = its[0]
        when = f" (tema d'esame del {first.exam})" if first.exam else ""
        prompt = f"[{first.source.title} — {key.split('-')[-1].replace('E', 'Esercizio ')}{when}]"
        parts, steps, ids, owner = [], [], [], []
        for it in auto:
            try:
                ex = build(it)
            except TaskError:
                continue
            body = ex.prompt.split("\n", 1)[1] if "\n" in ex.prompt else ex.prompt
            prompt += f"\n\n— {it.ref} —\n{body}"
            words = it.ref.split(maxsplit=2)     # 'Esercizio 1 b) c)' -> 'b) c)'
            sub = words[2] if len(words) == 3 else it.ref or it.id
            for p in ex.parts:
                parts.append(Part(f"{sub} {p.label}".strip(), p.answer))
                owner.append(it.id)
            steps += [f"— {it.ref} —"] + list(ex.solution)
            ids.append(it.id)
        if parts:
            out.append(Exercise(auto[0].topic, "bank:exam", prompt, parts, steps,
                                {"item_ids": ids, "part_items": owner, "exam": first.exam}))
    return out


_BANK: Bank | None = None


def get_bank() -> Bank:
    global _BANK
    if _BANK is None:
        _BANK = load_bank()
    return _BANK


def reload_bank() -> Bank:
    global _BANK
    _BANK = load_bank()
    return _BANK


def solved_in_text(item: Item) -> str:
    out = []
    for src, pages in item.solved_in:
        path = src.files[0]["path"] if src.files else ""
        pg = f", p. {', '.join(str(p) for p in pages)}" if pages else ""
        out.append(f"Solved in class (handwritten): {src.title}{pg}  →  {path}")
    return "\n".join(out)


__all__ = ["Bank", "Item", "Source", "TaskError", "build", "get_bank", "load_bank", "past_exam_exercises", "reload_bank",
           "bank_dir", "material_root"]
