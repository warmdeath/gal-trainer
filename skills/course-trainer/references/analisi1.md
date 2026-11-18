# Playbook: Analisi Matematica 1

Read in Phase 0. Analysis differs from linear algebra in one decisive way: answers are
**expressions** (a limit, a derivative, a primitive, a Taylor polynomial, a general
solution), not numbers or matrices. That changes the engine, the checkers and the traps.

## Contents
Topic checklist · Engine · Input syntax · Answer types · Exercise kinds and generators ·
Typical mistakes · Verification · Theory (oral) · Exam format

## Topic checklist
Follow the order of the course's notes; this list only helps you spot gaps. Typical first
course (Ingegneria, PoliMi style): insiemi numerici, sup/inf/max/min, principio di
induzione; numeri complessi (forma algebrica, trigonometrica, esponenziale, radici
n-esime, equazioni); funzioni elementari e grafici; successioni (limiti, monotonia,
confronto, limiti notevoli, ordini di infinito); limiti di funzioni e continuità
(teoremi: zeri, Weierstrass, valori intermedi); asintotici, o-piccolo, ~;
derivate (regole, teoremi di Fermat, Rolle, Lagrange, de l'Hôpital); formula di Taylor
(Peano, Lagrange) e sviluppi notevoli; studio di funzione (dominio, simmetrie, limiti,
asintoti, monotonia, convessità, grafico); integrali (Riemann, teorema fondamentale,
primitive: sostituzione, per parti, razionali); integrali generalizzati (criteri di
confronto e confronto asintotico, parametro α); serie numeriche (geometrica, armonica
generalizzata, criteri: confronto, confronto asintotico, rapporto, radice, Leibniz,
assoluta convergenza); equazioni differenziali (1° ordine lineari e a variabili
separabili, 2° ordine lineari a coefficienti costanti con metodo di somiglianza,
problema di Cauchy). Some courses add funzioni integrali or series of functions — only
if the notes do.

## Engine: a CAS at runtime
Use **sympy at runtime** (move it from dev to main dependencies). Exactness still rules:
keys are sympy expressions built from exact rationals, `pi`, `E`, `sqrt`, `log` — never
floats. Wrap heavy calls (`simplify`, `limit`, `integrate`, `dsolve`) in a time budget
(subprocess or signal-based timeout); a hang in front of the student is unacceptable.
Generate questions so that the CAS work is cheap: compute the answer while designing the
question (e.g. pick the primitive F, differentiate it to get the integrand).

## Input syntax (Italian conventions!)
Parse with `sympy.parsing.sympy_parser.parse_expr` + `implicit_multiplication_application`,
`convert_xor`, and a local dictionary. Accept, all meaning the obvious thing:
`sen`/`sin`, `tg`/`tan`, `arctg`/`arctan`, `arcsen`, `sh`/`sinh`, `ch`/`cosh`, `log` **and**
`ln` (in Italian courses `log` is the natural log; say so in the format hint), `e^x`, `exp(x)`,
`sqrt`, `√`, `|x|` (→ `Abs`), `pi`/`π`, `inf`/`+inf`/`-inf`/`∞`/`oo`, `x^2`, `2x`, `x(x+1)`,
`e^(2x)`, `1/2x` (document how it parses: ask the student for parentheses). Intervals:
`(-inf, 2) U [3, 5]`, `R`, `R \ {0}`, `x>2`. Complex: `3+4i`, `2e^(i pi/3)`, `2(cos(pi/3)+i sin(pi/3))`.
"Does not exist": `non esiste`, `ne`, `dne`. Never count unparseable input as wrong.

## Answer types
| type | correct when | notes |
|---|---|---|
| ExprAns | `simplify(student − key) == 0`, else numeric agreement at ≥ 8 random points of the domain (mpmath, 30 digits) | domain-aware: sample where both are defined |
| LimitAns | same value, incl. `+∞`, `−∞`, `∞` (unsigned, if the course uses it), "non esiste" | distinguish 0⁺/0⁻ only if the question asks |
| PrimitiveAns | `d/dx(student) − integrand ≡ 0` (any constant; +C optional) | `ln|x|` vs `ln x`: accept per the question's interval |
| DefIntegralAns / SeriesSumAns | exact value equivalence | |
| TaylorAns | polynomials equal after truncating to the asked order | report "missing term x^k" |
| AsymptoticAns (~, o-piccolo, principal part) | `limit(student/key) == 1` (equivalence) or same order and coefficient | |
| IntervalSetAns | sympy `Interval`/`Union`/`FiniteSet` equality (domains, inequalities, α-sets) | |
| ConvergenceChoice | converge / diverge (+∞/−∞) / irregolare; series: assolutamente / semplicemente | plus an optional `open` part "criterio usato" |
| ParamSetAns | the set of α (or x) for convergence/existence | boundary points matter: test them |
| ODEGeneralAns | substituting into the ODE gives 0 identically **and** the free constants are independent (Jacobian w.r.t. the constants has full rank) | any letters for constants |
| CauchyAns | exact function equivalence on the interval | |
| ComplexAns / ComplexSetAns | equality of complex numbers / of the set of roots | any form (algebraic, polar, exponential) |
| SupInfAns | value or "non esiste"; plus yes/no "è massimo?" | |
| Study of function | one Part per item: domain (IntervalSet), parity (choice), limits at the ends (Limit), asymptotes (list of lines `y = mx + q`, `x = a`), stationary points + nature, monotonicity intervals, convexity intervals | the graph itself stays self-assessed |

## Exercise kinds and generators (design the answer first)
- **Limits**: build from limiti notevoli / Taylor so the value is rational or a simple
  surd: `(1 − cos(ax))/(bx²) → a²/(2b)`; `(e^{ax} − 1 − ax)/x²`; ratio of polynomials with
  chosen leading terms; `n`-sequences with `(1 + a/n)^{bn}`. Keep a "trap" variant where
  de l'Hôpital does not apply.
- **Derivatives**: random composition of 2–3 elementary functions with integer
  parameters; ask f′(x₀) at a point where it's nice, or the tangent line.
- **Primitives / integrals**: pick F (nice), differentiate to get f; for rational
  functions pick roots and partial-fraction coefficients first; by parts with `x·e^x`,
  `x·sin x`, `ln x`; substitution with a visible inner derivative.
- **Improper integrals / series with parameter α**: from `1/x^α`, `1/(n^α)`,
  `log` factors and asymptotic comparison; the answer is an interval in α — test the
  boundary separately.
- **Taylor**: compose standard expansions (e^x, sin, cos, log(1+x), (1+x)^a, arctan)
  up to order 3–5; ask the polynomial or a limit solved with it.
- **ODEs**: choose the characteristic roots (distinct real, double, complex) and the
  forcing term (polynomial, exponential, trig, resonant case); build y_p and the Cauchy
  data from a chosen solution.
- **Complex numbers**: equations `z^n = w` with w on nice angles; `|z − a| = r` loci as
  open self-check with a model sketch description.
- **Study of function**: families with nice stationary points (`x e^{-x}`,
  `(x²−1)/(x+2)`, `ln(x)/x`, `x − 2 arctan x`).
- Official sheets and past papers go to the bank exactly as in other courses; add task
  kinds `limit`, `derivative`, `primitive`, `defint`, `taylor`, `series_conv`,
  `improper_conv`, `ode`, `cauchy`, `complex_roots`, `domain`, `asymptotes`,
  `study` (multi-part), `supinf`.

## Typical mistakes worth diagnosing
de l'Hôpital on a non-indeterminate form; chain-rule factor missing; sign of `∫ sin`;
forgetting the absolute value in `ln|x|`; Taylor truncated at the wrong order or `o()`
dropped too early; equivalence `~` used inside a sum (cancellation); ratio test equal to 1
read as convergence; Leibniz without checking monotonicity; boundary α of `Σ 1/n^α`
(α = 1); resonance in ODEs (missing factor x); wrong argument of complex roots (branch);
oblique asymptote with q computed before m; domain of `log`/`sqrt` compositions.
Implement them as `mistakes=[(wrong_value, explanation)]` computed alongside the key.

## Verification (tests)
Recompute every key independently: `sympy.limit`, `diff`, check primitives by
differentiating, `integrate` for definite integrals, `series(...).removeO()` for Taylor,
`dsolve` + `checkodesol` for ODEs, `solveset` for domains/inequalities. sympy's series
convergence support is limited: for `series_conv`/`improper_conv` verify with your own
criterion implementation on the generated families, plus numeric partial sums where they
discriminate. Every generator: round trip `check(typed())` over 40 seeds; test the
parser on all the Italian spellings above.

## Theory and oral
Analysis exams often have theory questions (enunciati, dimostrazioni, controesempi).
Build `open` items from the digest's definitions and theorems — the model answer is the
notes' own wording — and schedule them with spaced repetition like topics.

## Exam format
Do not assume: read the course's rules sheet (duration, parts, points, theory part,
itinere, pass mark) into `papers.RULES`. Analysis papers often have exercises with
several sub-points and a theory section: map them to `section` I/II/III with points.
