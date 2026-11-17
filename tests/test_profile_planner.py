import pytest
import random
from datetime import date, datetime, timedelta

from gal.exam import BLUEPRINT, build_exam
from gal.planner import advise
from gal.profile import Profile
from gal.topics import BY_ID, TOPICS


def test_record_and_persistence(tmp_path):
    p = Profile.load(tmp_path / "p.json")
    now = datetime(2026, 11, 17, 10, 0)
    for s in (1, 1, 1):
        st = p.record("matrices", s, now=now)
    assert st.mastered and st.box == 3
    assert st.due == (now.date() + timedelta(days=4)).isoformat()
    st = p.record("matrices", 0, now=now)
    assert st.box == 0 and st.due == now.date().isoformat() and not st.mastered
    p.save()
    q = Profile.load(tmp_path / "p.json")
    assert q.topics["matrices"].attempts == 4
    assert q.topics["matrices"].mastery == p.topics["matrices"].mastery


def test_corrupt_profile_is_backed_up(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{not json")
    p = Profile.load(f)
    assert p.topics == {}
    assert (tmp_path / "p.corrupt.json").exists()


def test_next_follows_course_order_then_reviews():
    p = Profile()
    today = date(2026, 11, 17)
    adv = advise(p, today)
    assert adv.action == "learn" and adv.topic.id == TOPICS[0].id
    now = datetime(2026, 11, 17, 9)
    for _ in range(3):
        p.record("vectors", 1, now=now)
    adv = advise(p, today)
    assert adv.topic.id == "lines_planes" and adv.action == "learn"
    # a wrong answer makes a topic due today → spaced repetition has priority
    p.record("lines_planes", 1, now=now)
    p.record("vectors", 0, now=now)
    p.record("vectors", 0, now=now)
    adv = advise(p, today)
    assert adv.action == "review" and adv.topic.id == "vectors"


def test_next_overdue_mastered_topic():
    p = Profile()
    old = datetime(2026, 10, 1)
    for _ in range(3):
        p.record("vectors", 1, now=old)
    adv = advise(p, date(2026, 11, 17))
    assert adv.action == "review" and adv.topic.id == "vectors"


def test_next_exam_when_all_mastered():
    p = Profile()
    now = datetime(2026, 11, 17)
    for t in TOPICS:
        for _ in range(6):
            p.record(t.id, 1, now=now)
    adv = advise(p, date(2026, 11, 17))
    assert adv.action == "official" and adv.source == "F01"   # unsolved sheets come first
    from gal.bank import get_bank
    for iid in get_bank().items:
        p.record_item(iid, 1)
    assert advise(p, date(2026, 11, 17)).action == "exam"


def test_prerequisites_exist_and_precede():
    order = {t.id: i for i, t in enumerate(TOPICS)}
    for t in TOPICS:
        for pre in t.prereqs:
            assert pre in BY_ID
            assert order[pre] < order[t.id], (t.id, pre)


def test_build_exam():
    for slot in BLUEPRINT:
        for tid in slot:
            assert any(g.exam for g in BY_ID[tid].gens), tid
    ex = build_exam(random.Random(1), Profile(), 6)
    assert len(ex.items) == 6 and ex.minutes == 60
    assert len({i.exercise.topic for i in ex.items}) >= 4
    ex = build_exam(random.Random(2), None, 3, 15, topics=["systems", "rank"])
    assert {i.exercise.topic for i in ex.items} <= {"systems", "rank"}


def test_weighted_record_moves_mastery_like_repeated_results():
    from gal.profile import PAST_WEIGHT
    a, b = Profile(), Profile()
    for p in (a, b):
        for _ in range(6):
            p.record("rank", 1.0)
    a.record("rank", 0.0, weight=PAST_WEIGHT)
    b.record("rank", 0.0)
    b.record("rank", 0.0)
    assert abs(a.stats("rank").mastery - b.stats("rank").mastery) < 1e-3
    assert a.stats("rank").attempts == 7


def test_parse_grade():
    from gal.profile import parse_grade
    assert [parse_grade(x) for x in ("y", "p", "n", "sì", "0.7", "70%", "3/4", "0,5")] == \
        [1.0, 0.5, 0.0, 1.0, 0.7, 0.7, 0.75, 0.5]
    assert parse_grade("2") is None and parse_grade("abc") is None and parse_grade("1/0") is None


def test_past_exam_record_and_readiness():
    p = Profile()
    rec = p.add_past_exam("tema 07/02/2024", [
        {"topics": ["systems"], "got": 8, "max": 8},
        {"topics": ["diagonalization", "eigen"], "got": 2, "max": 8},
        {"topics": [], "got": 6, "max": 8}], taken="2026-11-20", grader="agent")
    assert rec["points"] == 16 and rec["max"] == 24 and rec["grade"] == 20.0
    assert p.stats("systems").mastery == 1.0 and p.stats("eigen").mastery == 0.25
    assert p.stats("eigen").box == 0                       # weak topic: due again now
    assert p.history[-1]["source"] == "past-exam" and p.history[-1]["weight"] == 2.0
    p.add_exam({"date": "2026-11-21", "n": 6, "grade": 26.0})
    assert p.readiness() == (round((20 * 2 + 26) / 3, 1), 2)
    with pytest.raises(ValueError):
        p.add_past_exam("bad", [], total=(31, 30))


def test_past_exams_persist(tmp_path):
    p = Profile(path=tmp_path / "p.json")
    p.add_past_exam("tema A", [], total=(24, 30))
    p.save()
    q = Profile.load(tmp_path / "p.json")
    assert q.past_exams[0]["grade"] == 24.0 and q.topics == {}
