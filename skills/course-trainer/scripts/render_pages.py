#!/usr/bin/env python3
"""Render PDF pages to PNG (to read handwriting / formulas) and optional contact sheets.

    python render_pages.py notes.pdf out_dir [--dpi 60] [--first 1] [--last N] [--contact 6]

Needs `pdftoppm` (poppler-utils). --contact K also writes contact sheets with K pages each
(needs Pillow), useful to map a long handwritten file quickly before reading pages in full.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def render(pdf: Path, out: Path, dpi: int, first: int | None, last: int | None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["pdftoppm", "-r", str(dpi), "-png"]
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    stem = out / pdf.stem.replace(" ", "_")
    subprocess.run(cmd + [str(pdf), str(stem)], check=True)
    return sorted(out.glob(stem.name + "-*.png"))


def contact(pages: list[Path], out: Path, per_sheet: int) -> list[Path]:
    from PIL import Image, ImageDraw
    sheets = []
    for k in range(0, len(pages), per_sheet):
        group = [Image.open(p) for p in pages[k:k + per_sheet]]
        w = max(i.width for i in group)
        h = max(i.height for i in group)
        cols = 3 if per_sheet > 4 else 2
        rows = (len(group) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (img, p) in enumerate(zip(group, pages[k:k + per_sheet])):
            x, y = (j % cols) * w, (j // cols) * h
            sheet.paste(img, (x, y))
            draw.text((x + 8, y + 8), p.stem.rsplit("-", 1)[-1], fill="red")
        path = out / f"contact-{k // per_sheet + 1:02d}.png"
        sheet.save(path)
        sheets.append(path)
    return sheets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dpi", type=int, default=60)
    ap.add_argument("--first", type=int)
    ap.add_argument("--last", type=int)
    ap.add_argument("--contact", type=int, default=0, help="pages per contact sheet (0 = none)")
    a = ap.parse_args()
    pages = render(a.pdf, a.out, a.dpi, a.first, a.last)
    print(f"{len(pages)} page image(s) in {a.out}")
    if a.contact:
        try:
            for s in contact(pages, a.out, a.contact):
                print("contact sheet:", s)
        except ImportError:
            print("Pillow not installed: skipping contact sheets", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
