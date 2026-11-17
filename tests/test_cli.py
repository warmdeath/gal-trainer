"""Drive the whole UI with scripted input."""
import io
import random

from rich.console import Console

from gal.cli import App, resolve_topic
from gal.profile import Profile
from gal.topics import TOPICS


class Script:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, label):
        self.prompts.append(label)
        return self.answers.pop(0) if self.answers else "quit"


def make_app(answers, tmp_path, seed=0):
    buf = io.StringIO()
    con = Console(file=buf, width=120, color_system=None)
    s = Script(answers)
    prof = Profile.load(tmp_path / "p.json")
    app = App(prof, console=con, ask=s, rng=random.Random(seed))
    app.official_share = 0  # keep random exercises predictable in these tests
    return app, buf, s


def first_exercise(topic_id, seed=0):
    t = resolve_topic(topic_id)
    return t.random_exercise(random.Random(seed))


def test_resolve_topic():
    assert resolve_topic("det").id == "determinant"
    assert resolve_topic("1").id == TOPICS[0].id
    assert resolve_topic("gram").id == "gram_schmidt"
    assert resolve_topic("zzz") is None


def test_practice_correct_and_wrong(tmp_path):
    ex = first_exercise("determinant", 0)
    right = ex.parts[0].answer.typed()
    app, buf, _ = make_app(["practice det", right, "", "123456", "q", "quit"], tmp_path)
    app.run()
    out = buf.getvalue()
    assert "✓ correct" in out
    assert "✗ not correct" in out and "Step-by-step solution" in out
    st = app.prof.stats("determinant")
    assert st.attempts == 2 and st.correct == 1
    assert (tmp_path / "p.json").exists()


def test_unreadable_input_is_not_penalised(tmp_path):
    ex = first_exercise("matrices", 0)
    app, buf, _ = make_app(["p matrices", "this is not a matrix ((", "?", ex.parts[0].answer.typed(), "q", "quit"],
                           tmp_path)
    app.run()
    out = buf.getvalue()
    assert "I can't read that" in out and "Format:" in out
    assert app.prof.stats("matrices").attempts == 1 and app.prof.stats("matrices").correct == 1


def test_learn_next_topics_stats_help(tmp_path):
    app, buf, _ = make_app(["help", "topics", "learn vectors", "", "", "", "n", "next", "n", "stats",
                            "examdate 2027-01-20", "bogus", "quit"], tmp_path)
    app.run()
    out = buf.getvalue()
    assert "Worked example" in out and "Solution" in out
    assert "Next: learn" in out
    assert "Progress" in out and app.prof.exam_date == "2027-01-20"
    assert "Unknown command" in out
    assert app.prof.stats("vectors").lesson_seen


def test_exam_flow(tmp_path):
    app, buf, s = make_app(["exam 2 30", ""] + ["skip"] * 12 + [""] + ["quit"], tmp_path)
    app.run()
    out = buf.getvalue()
    assert "Mock exam result: 0.0/30" in out
    assert len(app.prof.exams) == 1 and app.prof.exams[0]["n"] == 2


def test_exam_with_correct_answers(tmp_path):
    # rebuild the same exam to know the answers: App uses its rng for build_exam
    from gal.exam import build_exam
    app, buf, script = make_app(["exam 2 30", ""], tmp_path, seed=5)
    probe = build_exam(random.Random(5), Profile(), 2, 30, past=app._past_exam_exercises(None))
    script.answers += [p.answer.typed() for item in probe.items for p in item.exercise.parts] + ["1", "", "quit"]
    app.run()
    assert "Mock exam result: 30.0/30" in buf.getvalue()


def test_ai_chapter_runs(tmp_path):
    app, buf, _ = make_app([""] * 7 + ["n", "quit"], tmp_path)
    app.command("ai")
    out = buf.getvalue()
    assert "Live training" in out and "Learned predictions" in out


def test_reset(tmp_path):
    app, buf, _ = make_app(["yes"], tmp_path)
    app.prof.record("vectors", 1)
    app.command("reset")
    assert app.prof.topics == {}


def test_reflow_keeps_formulas_and_bullets():
    from gal.cli import reflow
    txt = "A long\nparagraph.\n• bullet one\n  continues\n• bullet two\n   x = 1 + t   (formula)\nPlain after."
    out = reflow(txt).split("\n")
    assert out == ["A long paragraph.", "• bullet one continues", "• bullet two",
                   "   x = 1 + t   (formula)", "Plain after."]


def test_quit_at_go_prompt_does_not_start(tmp_path):
    app, buf, _ = make_app(["quit"], tmp_path)
    app.next()
    assert "Intuition" not in buf.getvalue()


# ------------------------------------------------------------------ official material
from gal.bank import build, get_bank  # noqa: E402


def typed_answers(item_id):
    return [p.answer.typed() for p in build(get_bank().items[item_id]).parts]


def test_sheets_table(tmp_path):
    app, buf, _ = make_app([], tmp_path)
    app.command("sheets")
    out = buf.getvalue()
    assert "Foglio di esercizi 2" in out and "Quiz 4" in out and "handwritten" in out


def test_work_through_sheet_records_items(tmp_path):
    first = get_bank().sources["F02"].items[0]
    answers = typed_answers(first.id) + ["q"]
    app, buf, _ = make_app(answers, tmp_path)
    app.command("sheet 2")
    out = buf.getvalue()
    assert "Foglio di esercizi 2" in out and "✓ correct" in out
    assert app.prof.item_done(first.id)
    assert app.prof.stats(first.topic).attempts == 1
    # next time the solved item is skipped
    app2, buf2, s2 = make_app(["q"], tmp_path)
    app2.prof = app.prof
    app2.command("sheet 2")
    assert first.ref not in buf2.getvalue().split("Exercise 1")[1][:200]


def test_open_item_self_assessment(tmp_path):
    app, buf, _ = make_app(["", "p"], tmp_path)
    app.command("item Q01-C3")
    out = buf.getvalue()
    assert "Model answer" in out and "Lemma 1.15" in out
    assert app.prof.bank["Q01-C3"]["best"] == 0.5


def test_quiz_drill_and_solved_in_link(tmp_path):
    app, buf, s = make_app(["a"] * 30, tmp_path)
    app.command("quiz")
    assert "Quiz drill" in buf.getvalue()
    assert sum(v["attempts"] for v in app.prof.bank.values()) >= 1
    app, buf, _ = make_app(["0"] * 10 + ["q"], tmp_path)
    app.command("item F05-E5")
    assert "Solved in class (handwritten): Esercitazione 5" in buf.getvalue()


def test_practice_mixes_official_items(tmp_path):
    app, buf, _ = make_app(["skip", "", "skip", "", "skip", "", "skip", "q"], tmp_path, seed=3)
    app.official_share = 1.0
    t = resolve_topic("systems")
    app.practice(t)
    assert "Foglio di esercizi" in buf.getvalue()


def test_material_notice(tmp_path, monkeypatch):
    import gal.bank.ingest as ing
    monkeypatch.setattr(ing, "pending", lambda *a, **k: [tmp_path / "Foglio_esercizi_09.pdf"])
    app, buf, _ = make_app([], tmp_path)
    app.material_notice()
    assert "New course material" in buf.getvalue() and "ingest" in buf.getvalue()


def test_stats_shows_official_progress(tmp_path):
    app, buf, _ = make_app([], tmp_path)
    app.prof.record_item("F01-E1a", 1)
    app.command("stats")
    out = buf.getvalue()
    assert "Official exercises: 1/" in out and "F01 1/" in out


def test_exam_with_past_exam_exercise_credits_items(tmp_path):
    from gal.bank import get_bank, past_exam_exercises
    from gal.exam import Exam, ExamItem
    app, buf, script = make_app([], tmp_path)
    past = [e for e in past_exam_exercises(get_bank()) if "F06-E4a" in e.data["item_ids"]][0]
    ex = Exam([ExamItem(past)], 30)
    item = ex.items[0]
    item.scores = [1.0] * len(past.parts)
    item.answers = [p.answer.typed() for p in past.parts]
    item.msgs = [""] * len(past.parts)
    app._exam_report(ex, 1.0)
    assert all(app.prof.item_done(i) for i in ("F06-E4a", "F06-E4c", "F06-E4d"))
    assert not app.prof.item_done("F06-E4b")


def test_past_add_interactive(tmp_path):
    app, buf, _ = make_app(["tema 07/02/2024", "2026-11-20", "agent", "90", "2",
                            "systems, rank", "7/8", "eigen", "2/8"], tmp_path)
    app.command("past add")
    out = buf.getvalue()
    rec = app.prof.past_exams[0]
    assert rec["paper"] == "tema 07/02/2024" and rec["grader"] == "agent" and rec["minutes_used"] == 90
    assert rec["grade"] == round(30 * 9 / 16, 1)
    assert app.prof.stats("rank").attempts == 1 and app.prof.stats("eigen").mastery == 0.25
    assert "To review first: eigen" in out
    app.command("past")
    assert "Exam readiness" in buf.getvalue()


def test_past_add_quick_and_rm(tmp_path):
    app, buf, _ = make_app([], tmp_path)
    app.command("past add tema giugno 2025 27/33")
    assert app.prof.past_exams[0]["paper"] == "tema giugno 2025"
    assert app.prof.past_exams[0]["grade"] == round(30 * 27 / 33, 1)
    app.command("stats")
    assert "Real past exams" in buf.getvalue() and "Exam readiness" in buf.getvalue()
    app.command("past rm 1")
    assert app.prof.past_exams == []


def test_past_add_rejects_bad_input_then_accepts(tmp_path):
    app, buf, _ = make_app(["tema X", "", "", "", "1", "nosuchtopic", "rank", "9/8", "5/8"], tmp_path)
    app.command("past add")
    out = buf.getvalue()
    assert "Unknown topic" in out and "got/max" in out
    assert app.prof.past_exams[0]["exercises"][0] == {"n": 1, "topics": ["rank"], "got": 5.0, "max": 8.0}


def test_grade_series_of_proofs(tmp_path):
    app, buf, _ = make_app([], tmp_path)
    app.command("grade F07-E13 F01-E2 y F05-E2a p q01-c1 3/4")
    bank = app.prof.bank
    assert bank["F07-E13"]["best"] == 1.0 and bank["F05-E2a"]["best"] == 0.5
    assert any(k.startswith("F01-E2") for k in bank)
    assert "Recorded:" in buf.getvalue()
    app.command("grade F07-E13 maybe")
    assert "Not understood" in buf.getvalue()


def test_grade_without_value_shows_model_answer(tmp_path):
    app, buf, _ = make_app(["boh", "p"], tmp_path)
    app.command("grade F07-E13")
    out = buf.getvalue()
    assert "Model answer" in out and app.prof.bank["F07-E13"]["best"] == 0.5


def test_proofs_list(tmp_path):
    app, buf, _ = make_app([], tmp_path)
    app.prof.record_item("F07-E13", 0.5)
    app.command("proofs 7")
    out = buf.getvalue()
    assert "F07-E13" in out and "50%" in out
