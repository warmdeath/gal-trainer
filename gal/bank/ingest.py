"""Ingest new course material (PDFs) into the practice bank.

    gal ingest            scan the material folders, create skeleton sources for new PDFs
    gal ingest --rehash   also accept revised versions of already transcribed PDFs

Material folders (inside GAL_MATERIAL_DIR, default: the repository root) are the ones the
course website produces when downloading a folder, e.g. `Esercizi-20261117`, `Quiz-20261201`,
`Esercitazioni-20261117`, plus anything under `material/`. Re-downloading a folder later (with
a new date suffix) is fine: files are recognised by their SHA-256, so duplicates are ignored.

For a NEW pdf the ingester
  1. extracts its text (pdftotext if installed, else the optional `pypdf` package),
  2. splits it into items at "Esercizio N." / "Domanda N." / "Compito N.",
  3. guesses the topic of each item from Italian keywords,
  4. writes gal/bank/data/<ID>.toml with status = "skeleton" and every item as task = "open".
Skeleton items are usable immediately (self-assessed). Upgrading an item to an auto-checked
task = edit its `task` and `data` (see CLAUDE.md) and run the tests.
Handwritten esercitazioni get a source without items; link their pages to items by hand.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import bank_dir, load_bank, material_root

FOLDER_PATTERNS = ["Esercizi*", "Esercitazioni*", "Quiz*", "Esami*", "Temi*", "material"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def find_pdfs(root: Path | None = None) -> list[Path]:
    root = root or material_root()
    out = []
    for pat in FOLDER_PATTERNS:
        for d in sorted(root.glob(pat)):
            if d.is_dir():
                out += sorted(p for p in d.rglob("*") if p.suffix.lower() == ".pdf")
    return out


def kind_of(path: Path) -> str:
    s = (path.parent.name + "/" + path.name).lower()
    name, anc = path.name.lower(), "/".join(path.parts[-4:-1]).lower()
    if "soluzion" in name or path.parent.name.lower() == "soluzioni":
        return "solutions"
    if "appello" in name or "itinere" in name or "esam" in anc or "tema" in anc:
        return "exam"
    if "esercitazion" in s:
        return "esercitazione"
    if "quiz" in s:
        return "quiz"
    if "esam" in s or "tema" in s:
        return "exam"
    return "sheet"


PREFIX = {"sheet": "F", "quiz": "Q", "esercitazione": "ES", "exam": "P", "solutions": "S"}


def exam_tag(path: Path) -> str:
    """'2026-02 Appello - V1' -> '2602AV1', '2024-01 Itinere II - Parte B' -> '2401I2PB'."""
    stem = path.stem
    m = re.search(r"(20\d\d)-(\d\d)", stem)
    ym = f"{m.group(1)[2:]}{m.group(2)}" if m else "".join(re.findall(r"\d+", stem))[-4:]
    low = stem.lower()
    tag = ym
    if "itinere" in low:
        k = re.search(r"itinere\s*(ii|i|\d)?\b", low)
        num = {"i": "1", "ii": "2"}.get(k.group(1), k.group(1)) if k and k.group(1) else ""
        tag += "I" + (num or "")
    else:
        tag += "A"
    v = re.search(r"\bv(\d)\b|versione\s+([a-z])", low)
    if v:
        tag += "V" + (v.group(1) or v.group(2).upper())
    pt = re.search(r"parte\s+([ab])\b", low)
    if pt:
        tag += "P" + pt.group(1).upper()
    return tag


def propose_id(path: Path, kind: str, taken: set[str]) -> str:
    if kind in ("exam", "solutions"):
        base = PREFIX[kind] + exam_tag(path)
        cand, k = base, 1
        while cand in taken:
            k += 1
            cand = f"{base}v{k}"
        return cand
    nums = re.findall(r"\d+", path.stem)
    # esercitazioni are named '2026-10-07 Esercitazione 2': the LAST number is the session
    n = int(nums[-1]) if nums else 1
    base = f"{PREFIX[kind]}{n:02d}"
    cand, k = base, 1
    while cand in taken:
        k += 1
        cand = f"{base}v{k}"
    return cand


# ------------------------------------------------------------------ text extraction
def extract_text(path: Path) -> tuple[str, int]:
    """(text, pages). pdftotext keeps the layout best; pypdf is the pure-Python fallback."""
    pages = 0
    if shutil.which("pdfinfo"):
        try:
            info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=60).stdout
            m = re.search(r"Pages:\s+(\d+)", info)
            pages = int(m.group(1)) if m else 0
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("pdftotext"):
        try:
            res = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                return res.stdout, pages
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in r.pages), pages or len(r.pages)
    except ImportError:
        raise RuntimeError("no PDF text extractor: install poppler-utils (pdftotext) or `pip install pypdf`") from None


HEADER = re.compile(r"(?:^|\f)[ \t]*(Esercizio|Domanda|Compito)\s+(\d+)\s*\.?", re.MULTILINE)


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    text = _CONTROL.sub("", text)
    lines = [re.sub(r"\s{3,}", "  ", l.strip()) for l in text.splitlines()]
    out = "\n".join(l for l in lines if l)
    return out.replace("6=", "≠").replace(" ", " ")


NUMBERED = re.compile(r"(?:^|\f)[ \t]*(\d{1,2})\.[ \t]+(?=\S)", re.MULTILINE)


def split_items(text: str, numbered: bool = False) -> list[tuple[str, int, str, int]]:
    """[(word, number, body, page)] in document order (pages are separated by form feeds).
    numbered=True: if there are no Esercizio/Domanda headers, split at '1. ', '2. ', ...
    (the test part of older exam papers) and call them Domanda."""
    ms = list(HEADER.finditer(text))
    if not ms and numbered:
        out, expect = [], 1
        cand = [m for m in NUMBERED.finditer(text)]
        seq = []
        for m in cand:                       # keep the increasing sequence 1, 2, 3, ...
            if int(m.group(1)) == expect:
                seq.append(m)
                expect += 1
        for i, m in enumerate(seq):
            end = seq[i + 1].start() if i + 1 < len(seq) else len(text)
            page = text.count("\f", 0, m.start(1)) + 1
            out.append(("Domanda", int(m.group(1)), clean(text[m.end():end]), page))
        return out
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        body = clean(text[m.end():end])
        page = text.count("\f", 0, m.start(1)) + 1
        out.append((m.group(1), int(m.group(2)), body, page))
    return out


# ------------------------------------------------------------------ topic guessing
TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("diagonalization", ("diagonalizzabil", "diagonalizzazione", "simili", "molteplicit")),
    ("eigen", ("autovalor", "autovettor", "polinomio caratteristico", "spettro", "autospazi")),
    ("gram_schmidt", ("gram-schmidt", "ortonormalizzazione", "base ortonormale")),
    ("projections", ("proiezione", "complemento ortogonale", "perpendicolare")),
    ("isometries", ("isometri", "matrice ortogonale", "rotazione", "riflessione")),
    ("cross_distances", ("prodotto vettoriale", "distanza tra", "distanza di")),
    ("bilinear", ("forma bilineare", "matrice di gram")),
    ("scalar_products", ("prodotto scalare", "norma", "angolo", "distanza")),
    ("rep_matrix", ("matrice rappresentativa", "cambiamento di base", "cambio di base", "matrice di trasformazione")),
    ("linear_maps", ("applicazion", "mappa lineare", "mappe lineari", "nucleo", "immagine", "iniettiv",
                     "suriettiv", "isomorfism", "kernel")),
    ("affine", ("rett", "piano", "piani", "posizione reciproca", "fascio", "sghemb")),
    ("sum_intersection", ("grassmann", "u + w", "u ∩ w", "somma diretta", "intersezione")),
    ("cart_param", ("forma cartesiana", "equazioni cartesiane", "equazione cartesiana")),
    ("bases", ("base", "dimensione", "coordinate")),
    ("span_independence", ("linearmente indipendent", "generatori", "combinazione lineare", "span")),
    ("vector_spaces", ("sottospazio", "spazio vettoriale")),
    ("determinant", ("determinante",)),
    ("inverse", ("inversa", "invertibil")),
    ("rank", ("rango", "rouché", "rouche")),
    ("systems", ("sistema", "sistemi", "meg", "gauss", "soluzion")),
]


def guess_topic(text: str) -> str:
    t = text.lower()
    best, score = "matrices", 0
    for topic, words in TOPIC_RULES:
        s = sum(t.count(w) for w in words)
        if s > score:
            best, score = topic, s
    return best


# ------------------------------------------------------------------ TOML writing
def q(s: str) -> str:
    """TOML basic string (JSON escaping is valid TOML)."""
    return json.dumps(s, ensure_ascii=False)


def paper_meta(path: Path) -> str:
    """TOML inline table for an exam paper skeleton (format guessed from the file name)."""
    low = path.stem.lower()
    fmt = "itinere" if "itinere" in low else "appello"
    minutes = 60 if fmt == "itinere" else 105
    points = 16 if fmt == "itinere" else 32
    half = 2 if fmt == "appello" or re.search(r"itinere\s*(2|ii)\b", low) else 1
    return (f'paper = {{ format = "{fmt}", minutes = {minutes}, points = {points}, lecturer = "?", half = {half}, '
            'points_estimated = true }')


def skeleton_toml(sid: str, title: str, kind: str, rel: str, digest: str, pages: int,
                  items: list[tuple[str, int, str]], meta: str = "") -> str:
    lines = ["# Generated by `gal ingest`: items are self-check ('open') until transcribed.",
             "# To auto-check an item: set its task + data (see CLAUDE.md, 'Adding course material').",
             "[source]", f"id = {q(sid)}", f"title = {q(title)}", f"kind = {q(kind)}", 'status = "skeleton"',
             f"files = [{{ path = {q(rel)}, sha256 = {q(digest)}, pages = {pages} }}]"] + ([meta] if meta else []) + [""]
    pts = {}
    if kind == "exam" and items:
        n_ex = sum(1 for w, *_ in items if w == "Esercizio")
        n_mcq = len(items) - n_ex
        total = 16 if "itinere" in meta else 32
        ex_pts = round((total - n_mcq) / n_ex, 2) if n_ex else 0
        pts = {"Domanda": 1, "Esercizio": max(ex_pts, 1), "Compito": max(ex_pts, 1)}
    letter = {"Esercizio": "E", "Domanda": "D", "Compito": "C"}
    seen: set[str] = set()
    for word, n, body, page in items:
        iid = f"{sid}-{letter[word]}{n}"
        while iid in seen:
            iid += "b"
        seen.add(iid)
        note = f"\n(Automatically extracted: formulas and matrices may be garbled — check the PDF, page {page}.)"
        extra = [f'section = "{"I" if word == "Domanda" else "II"}"', f"points = {pts[word]}"] if pts else []
        lines += ["[[item]]", f"id = {q(iid)}", f"ref = {q(f'{word} {n}')}", f"topic = {q(guess_topic(body))}",
                  f"page = {page}"] + extra + ['task = "open"', f"text = {q((body or '(no text extracted)') + note)}",
                  'data = { solution = "(see the PDF / its solutions)" }', ""]
    return "\n".join(lines)


def title_for(path: Path, kind: str, sid: str) -> str:
    stem = path.stem.replace("_", " ")
    if kind == "sheet":
        m = re.search(r"(\d+)", stem)
        return f"Foglio di esercizi {int(m.group(1))}" if m else stem
    if kind == "quiz":
        m = re.search(r"(\d+)", stem)
        return f"Quiz {int(m.group(1))}" if m else stem
    return stem


# ------------------------------------------------------------------ main entry
@dataclass
class Report:
    new: list[tuple[Path, str, int]] = field(default_factory=list)      # (pdf, source id, #items)
    duplicates: list[tuple[Path, str]] = field(default_factory=list)   # (pdf, known source id)
    changed: list[tuple[Path, str]] = field(default_factory=list)      # same file name, new content
    rehashed: list[str] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)

    def text(self) -> str:
        out = []
        for p, sid, n in self.new:
            out.append(f"NEW      {p.name} → {sid} ({n} item(s), self-check until transcribed)")
        for p, sid in self.changed:
            out.append(f"CHANGED  {p.name}: differs from the version transcribed in {sid} "
                       "(review it; `gal ingest --rehash` accepts it)")
        for p, sid in self.duplicates:
            out.append(f"same     {p.name} = {sid} (re-downloaded copy, ignored)")
        for sid in self.rehashed:
            out.append(f"rehashed {sid}")
        for p, e in self.errors:
            out.append(f"ERROR    {p.name}: {e}")
        return "\n".join(out) or "Nothing new: every PDF is already in the bank."


def known_files(bdir: Path):
    bank = load_bank(bdir)
    by_hash, by_name = {}, {}
    for src in bank.sources.values():
        for f in src.files:
            if f.get("sha256"):
                by_hash[f["sha256"]] = src.id
            by_name[Path(f["path"]).name] = (src.id, f.get("sha256", ""))
    return bank, by_hash, by_name


def pending(root: Path | None = None, bdir: Path | None = None) -> list[Path]:
    """PDFs that are neither known (by hash) nor duplicates: what `ingest` would add/flag."""
    root, bdir = root or material_root(), bdir or bank_dir()
    _, by_hash, by_name = known_files(bdir)
    out = []
    for p in find_pdfs(root):
        if sha256(p) not in by_hash:
            out.append(p)
    return out


def fill_missing_hashes(root: Path, bdir: Path) -> list[str]:
    """Write the sha256 of transcribed sources whose files list has sha256 = "" (first run)."""
    done = []
    for f in sorted(bdir.glob("*.toml")):
        txt = f.read_text(encoding="utf-8")
        changed = False

        def repl(m):
            nonlocal changed
            path = root / m.group(1)
            if path.exists():
                changed = True
                return f'path = "{m.group(1)}", sha256 = "{sha256(path)}"'
            return m.group(0)
        new = re.sub(r'path = "([^"]+)", sha256 = ""', repl, txt)
        if changed:
            f.write_text(new, encoding="utf-8")
            done.append(f.stem)
    return done


def ingest(root: Path | None = None, bdir: Path | None = None, rehash: bool = False) -> Report:
    root, bdir = root or material_root(), bdir or bank_dir()
    rep = Report()
    rep.rehashed = fill_missing_hashes(root, bdir)
    bank, by_hash, by_name = known_files(bdir)
    taken = set(bank.sources)
    for p in find_pdfs(root):
        digest = sha256(p)
        if digest in by_hash:
            known_src = bank.sources[by_hash[digest]]
            if all((root / f["path"]) != p for f in known_src.files):
                rep.duplicates.append((p, by_hash[digest]))
            continue
        rel = p.relative_to(root).as_posix()
        if p.name in by_name:
            sid, old = by_name[p.name]
            if rehash:
                src_file = bdir / f"{sid}.toml"
                txt = src_file.read_text(encoding="utf-8")
                txt = re.sub(r'sha256 = "' + re.escape(old) + '"', f'sha256 = "{digest}"', txt) if old else txt
                src_file.write_text(txt, encoding="utf-8")
                rep.rehashed.append(sid)
            else:
                rep.changed.append((p, sid))
            continue
        kind = kind_of(p)
        sid = propose_id(p, kind, taken)
        taken.add(sid)
        try:
            text, pages = extract_text(p)
        except RuntimeError as e:
            rep.errors.append((p, str(e)))
            text, pages = "", 0
        items = [] if kind in ("esercitazione", "solutions") else split_items(text, numbered=kind == "exam")
        meta = paper_meta(p) if kind == "exam" else ""
        (bdir / f"{sid}.toml").write_text(
            skeleton_toml(sid, title_for(p, kind, sid), kind, rel, digest, pages, items, meta), encoding="utf-8")
        by_hash[digest] = sid
        rep.new.append((p, sid, len(items)))
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gal ingest", description=__doc__.split("\n\n")[0])
    ap.add_argument("--rehash", action="store_true", help="accept revised versions of known PDFs")
    ap.add_argument("--material", default=None, help="folder containing the material folders")
    ns = ap.parse_args(argv)
    rep = ingest(Path(ns.material) if ns.material else None, rehash=ns.rehash)
    print(rep.text())
    return 0
