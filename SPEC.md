# SPEC — original request (saved verbatim)

The file Note_GAL26.pdf in the repo is the official course material. Read all of it first. Follow its topic order, notation, definitions and Italian terminology exactly, cover everything in it, and base the exercise style on its examples. It overrides the topic list below wherever they differ.

First, save this prompt as SPEC.md and create a CLAUDE.md (goals, architecture, how to run tests), kept updated. Work autonomously, commit after each milestone, write tests, don't ask questions; document decisions in REVIEW.md.

Build a terminal trainer in Python (rich + prompt_toolkit) for a first-year university Linear Algebra and Geometry course (Politecnico di Milano "GAL"), for a student who missed two months of lectures and has the exam in January.

- Topics, in dependency order: vectors, matrices and operations, linear systems and Gaussian elimination, rank, determinants, inverses, vector spaces and subspaces, linear independence, bases and dimension, linear maps, kernel and image, change of basis, eigenvalues and eigenvectors, diagonalization, scalar products and orthogonality, Gram-Schmidt, lines and planes in space, conics basics.
- Each topic: a short lesson (intuition first, then the formal definition, with the Italian term next to the English one), worked examples, then unlimited randomly generated exercises with nice numbers.
- Exact arithmetic with fractions. The student types the answer; on a mistake, show the full step-by-step solution and identify the likely error.
- Every generator and solver must be verified in tests against sympy. Correctness is the top priority.
- Persistent progress profile: mastery per topic, spaced repetition of weak topics, a `next` command that picks what to practice, and timed exam-style mock tests mixing topics.
- Final bonus chapter: "why this matters for AI": build a tiny neural network from scratch using the matrix operations learned, explained step by step.
- README with Windows/WSL setup. REVIEW.md at the end with architecture, simplifications and next ideas.
