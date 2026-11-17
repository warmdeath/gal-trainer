"""Exam papers: rules, timed runs graded in points, the study plan."""
import io
import random
from datetime import date, timedelta

import pytest
from rich.console import Console

from gal.bank import build, get_bank
from gal.cli import App
from gal.papers import (RULES, due_session, make_plan, max_points, paper_order, papers, session_dates,
                        written_score)
from gal.planner import advise
from gal.profile import Profile

BANK = get_bank()
PAPERS = papers(BANK)


class Script:
    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, label):
        return self.answers.pop(0) if self.answers else "q"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_app(answers, tmp_path, clock=None):
    buf = io.StringIO()
    app = App(Profile.load(tmp_path / "p.json"), console=Console(file=buf, width=120, color_system=None),
              ask=Script(answers), rng=random.Random(0), clock=clock or Clock())
    app.official_share = 0
    return app, buf


def perfect_answers(src):
    out = [""]                                     # start
    opens = 0
    for it in src.items:
        for p in build(it, with_setup=False).parts:
            if getattr(p.answer, "self_assessed", False):
                out.append("")                      # 'written on paper'
                opens += 1
            else:
                out.append(p.answer.typed())
    return out + ["y"] * opens


def test_rules_from_the_exam_info():
    assert RULES["appello"]["minutes"] == 105 and RULES["appello"]["points"] == 32
    assert RULES["itinere"]["minutes"] == 60 and RULES["itinere"]["points"] == 16
    assert written_score(20) == 20
    assert written_score(20, 14) == pytest.approx(2 * 34 / 3)   # the itinere helps: 22.67 > 20
    assert written_score(28, 4) == 28                              # ... or is ignored if it would hurt
    assert written_score(12, 16) == pytest.approx(2 * 28 / 3)


@pytest.mark.parametrize("src", PAPERS, ids=lambda s: s.id)
def test_paper_structure(src):
    fmt = (src.paper or {}).get("format")
    assert fmt in RULES
    assert all(i.points and i.points > 0 for i in src.items), "every item of a paper needs points"
    assert all(i.section in ("I", "II") for i in src.items)
    if (src.paper or {}).get("lecturer") == "Frapporti":
        mcq = [i for i in src.items if i.section == "I"]
        assert len(mcq) == RULES[fmt]["mcq"]
        assert max_points(src) == pytest.approx(RULES[fmt]["points"], abs=0.05)
    for f in src.files:
        assert f.get("sha256"), f"{src.id}: run `gal ingest` to fill the hashes"


@pytest.mark.parametrize("src", PAPERS, ids=lambda s: s.id)
def test_perfect_run_scores_full_marks(src, tmp_path):
    app, buf = make_app(perfect_answers(src), tmp_path)
    res = app.run_paper(src, timed=False)
    assert res.points == pytest.approx(max_points(src), abs=0.05)
    rec = app.prof.past_exams[-1]
    assert rec["paper_id"] == src.id and rec["grade"] == 30.0


def test_timed_run_stops_when_time_is_up(tmp_path):
    src = BANK.sources["P2602AV1"]
    clock = Clock()
    answers = perfect_answers(src)

    class Late(Script):
        def __call__(self, label):
            clock.t += 15 * 60            # every answer takes 15 minutes
            return super().__call__(label)
    app, buf = make_app([], tmp_path, clock)
    app.ask = Late(answers[:12] + ["y"] * 10)
    res = app.run_paper(src, timed=True)
    out = buf.getvalue()
    assert "Time is up" in out
    assert 0 < res.points < max_points(src)
    assert any(sc.note == "not reached" for sc in res.scores)


def test_wrong_answers_and_self_grades(tmp_path):
    src = BANK.sources["P2602AV1"]
    ans = perfect_answers(src)
    ans[1] = "a"                   # D1 wrong (key c)
    ans[-1] = "n"                  # last proof graded wrong
    ans[-2] = "p"
    app, buf = make_app(ans, tmp_path)
    res = app.run_paper(src, timed=False)
    assert res.points == pytest.approx(32 - 1 - 2.2 - 1.1, abs=0.01)
    assert "Parte I (test): 9/10" in buf.getvalue()
    assert app.prof.bank["P2602AV1-D1"]["best"] == 0.0
    assert app.prof.stats("vector_spaces").attempts == 1


def test_session_dates_and_plan():
    start, exam = date(2026, 11, 17), date(2027, 1, 18)
    ds = session_dates(start, exam)
    assert ds == sorted(ds) and ds[-1] == exam - timedelta(days=2) and ds[0] >= start
    window = exam - timedelta(days=14)
    assert all(d.weekday() in (2, 5) for d in ds if d < window)
    tail = [d for d in ds if d >= window]
    assert all((b - a).days == 2 for a, b in zip(tail, tail[1:]))
    plan = make_plan(BANK, start, exam)
    assert len({s["paper"] for s in plan}) == len(plan)
    assert all(s["paper"] in BANK.sources for s in plan)
    itinere, full = paper_order(BANK)
    if full:
        own = [s for s in full if (s.paper or {}).get("lecturer") == "Frapporti" and s.paper.get("format") == "appello"]
        if own:
            assert plan[-1]["paper"] == own[-1].id          # the most recent appello is the last rehearsal


def test_plan_command_and_next(tmp_path):
    app, buf = make_app([], tmp_path)
    app.prof.exam_date = (date.today() + timedelta(days=60)).isoformat()
    app.command("plan")
    assert app.prof.plan and "Study plan" in buf.getvalue()
    first = app.prof.plan[0]
    adv = advise(app.prof, date.fromisoformat(first["date"]))
    assert adv.action == "paper" and adv.source == first["paper"]
    assert due_session(app.prof.plan, date.fromisoformat(first["date"]) - timedelta(days=1)) is None


def test_paper_marks_plan_session_done(tmp_path):
    src = BANK.sources["P2602AV1"]
    app, buf = make_app(perfect_answers(src), tmp_path)
    app.prof.plan = [{"date": date.today().isoformat(), "paper": src.id, "done": False}]
    app.command("paper")
    assert app.prof.plan[0]["done"] and app.prof.plan[0]["result"] == "32.0/32"
    assert src.id in app.prof.papers_done()


def test_papers_list_and_exam_info(tmp_path):
    app, buf = make_app([], tmp_path)
    app.command("papers")
    app.command("examinfo")
    out = buf.getvalue()
    assert "P2602AV1" in out and "1 h 45 min" in out and "max(PC, 2/3 (PI + PC))" in out
