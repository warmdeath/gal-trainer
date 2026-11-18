"""Course material ↔ bank consistency, and the ingest pipeline for future material."""
import shutil
from pathlib import Path

import pytest

from gal.bank import build, load_bank, material_root
from gal.bank.ingest import (find_pdfs, guess_topic, ingest, pending, sha256, split_items)
from gal.topics import BY_ID

ROOT = material_root()
DATA = Path(__file__).resolve().parents[1] / "gal" / "bank" / "data"
HAVE_MATERIAL = bool(find_pdfs(ROOT))
needs_material = pytest.mark.skipif(not HAVE_MATERIAL, reason="course PDFs not present")
needs_extractor = pytest.mark.skipif(not (shutil.which("pdftotext") or __import__("importlib").util.find_spec("pypdf")),
                                     reason="no PDF text extractor")


@needs_material
def test_every_pdf_is_in_the_bank():
    """If this fails you added new material: run `gal ingest` (see CLAUDE.md)."""
    missing = [p.name for p in pending(ROOT, DATA)]
    assert not missing, f"PDFs not yet ingested: {missing}. Run `gal ingest`."


@needs_material
def test_every_source_file_exists_with_matching_hash():
    bank = load_bank(DATA)
    for src in bank.sources.values():
        for f in src.files:
            p = ROOT / f["path"]
            assert p.exists(), f"{src.id}: missing {f['path']}"
            assert f["sha256"] == sha256(p), f"{src.id}: {f['path']} changed — review it and run `gal ingest --rehash`"


def test_split_and_topics():
    text = ("Esercizio 1. Si calcoli il determinante della matrice.\n"
            "Esercizio 2. Si determini se la matrice A è diagonalizzabile e i suoi autovalori.\n"
            "Domanda 3. Sia f un'applicazione lineare iniettiva: nucleo e immagine?\n")
    items = split_items(text)
    assert [(w, n) for w, n, _, _ in items] == [("Esercizio", 1), ("Esercizio", 2), ("Domanda", 3)]
    assert [guess_topic(b) for _, _, b, _ in items] == ["determinant", "diagonalization", "linear_maps"]
    assert split_items("Esercizio 1. a\fEsercizio 2. b")[1][3] == 2


def _altered_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes() + b"\n% revised\n")


@needs_material
@needs_extractor
def test_ingest_new_duplicate_and_changed(tmp_path):
    bdir = tmp_path / "data"
    shutil.copytree(DATA, bdir)
    root = tmp_path / "root"
    # the existing material (so nothing is 'new')
    for p in find_pdfs(ROOT):
        dst = root / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)
    sheet1 = ROOT / "Esercizi-20261117" / "Foglio_esercizi_01.pdf"
    sheet2 = ROOT / "Esercizi-20261117" / "Foglio_esercizi_02.pdf"
    quiz1 = ROOT / "Quiz-20261117" / "QUIZ1.pdf"
    later = root / "Esercizi-20270110"
    later.mkdir()
    shutil.copy(sheet1, later / "Foglio_esercizi_01.pdf")                       # re-downloaded copy
    _altered_copy(sheet2, later / "Foglio_esercizi_09.pdf")                     # genuinely new sheet
    _altered_copy(quiz1, root / "Quiz-20270110" / "QUIZ5.pdf")                  # new quiz
    _altered_copy(sheet1, root / "Esercizi-20270111" / "Foglio_esercizi_01.pdf")  # revised version
    rep = ingest(root, bdir)
    new = {sid: n for _, sid, n in rep.new}
    assert set(new) == {"F09", "Q05"} and new["F09"] >= 5 and new["Q05"] >= 10
    assert any(sid == "F01" for _, sid in rep.duplicates)
    assert any(sid == "F01" for _, sid in rep.changed)
    bank = load_bank(bdir)
    assert bank.sources["F09"].status == "skeleton" and bank.sources["Q05"].kind == "quiz"
    for item in bank.sources["F09"].items + bank.sources["Q05"].items:
        assert item.topic in BY_ID and item.task == "open"
        ex = build(item)
        assert ex.parts[0].answer.check("y").ok
    assert {i.id for i in bank.sources["Q05"].items} >= {"Q05-C1", "Q05-D1"}
    # idempotent: a second run finds nothing new
    rep2 = ingest(root, bdir)
    assert not rep2.new
    assert [p.name for p in pending(root, bdir)] == ["Foglio_esercizi_01.pdf"]  # only the unresolved revision
    # accepting the revision
    rep3 = ingest(root, bdir, rehash=True)
    assert "F01" in rep3.rehashed
