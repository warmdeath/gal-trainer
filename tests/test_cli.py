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
    return App(prof, console=con, ask=s, rng=random.Random(seed)), buf, s


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
    probe = build_exam(random.Random(5), Profile(), 2, 30)
    answers = [p.answer.typed() for item in probe.items for p in item.exercise.parts]
    app, buf, _ = make_app(["exam 2 30", ""] + answers + ["1", "", "quit"], tmp_path, seed=5)
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
