# CLAUDE.md — gal-trainer

## Goal
Terminal trainer (Python, rich + prompt_toolkit) for Politecnico di Milano
"Geometria e Algebra Lineare". `Note_GAL26.pdf` is the source of truth: topic order,
notation (M_K(m,n), rg(A), Sol(A|b), MEG, Span, ker, Im, M_B^C(f), p_A(t)=det(tI-A),
ma/mg, ⟨·,·⟩, W⊥, π_W ...) and Italian terminology follow it. See SPEC.md for the request.
Correctness first: exact arithmetic (fractions.Fraction, Surd for q·√n), never floats
(except the AI bonus chapter, which is explicitly numerical).

## Architecture (work in progress — kept updated)
- `gal/core/` exact engine: `matrix.py` (Matrix over Q), `gauss.py` (MEG / Gauss-Jordan with
  step logs, rank, det, inverse, solve, kernel), `poly.py` (Poly, charpoly, rational roots),
  `surd.py` (q·√n), `fmt.py` (text rendering), `parse.py` (student answer parser).
