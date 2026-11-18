#!/usr/bin/env python3
"""Extract multiple-choice keys from official solution PDFs where the chosen option is
marked typographically (a crossed/ticked box that pdftotext renders as `C@`).

    python mc_keys.py "Soluzioni/2026-02 Appello - soluzioni.pdf" [--word Domanda]

Prints, per version block, `1C 2D 3B ...`; `?` where no mark was found (some files circle
the answer instead — read those pages as images). ALWAYS confirm against the paper's own
option order: some solution files list the correct option first.
"""
import argparse
import re
import subprocess


def keys(pdf: str, word: str) -> list[tuple[str, list[str]]]:
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
    heads = [m.start() for m in re.finditer(r"^.*\bsoluzioni\s*$", text, re.M | re.I)] or [0]  # version headers end with "soluzioni"
    out = []
    for i, start in enumerate(heads):
        end = heads[i + 1] if i + 1 < len(heads) else len(text)
        block = text[start:end]
        title = block.splitlines()[0].strip()[-60:] if block else ""
        parts = re.split(rf"{word}\s+(\d+)\.", block)
        ks = []
        for j in range(1, len(parts), 2):
            marks = re.findall(r"\b([A-E])@", parts[j + 1])
            ks.append(f"{parts[j]}{''.join(marks) or '?'}")
        out.append((title, ks))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--word", default="Domanda", help="question header word (Domanda, Esercizio, ...)")
    a = ap.parse_args()
    for title, ks in keys(a.pdf, a.word):
        print("##", title)
        print(" ".join(ks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
