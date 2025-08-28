# min-ratio-cycle

An optimized Python library for finding the **minimum cost-to-time ratio cycle** in a directed graph.

> Lawler-style parametric search with NumPy-accelerated negative‑cycle detection and an exact Stern–Brocot mode for integer weights.

---

## Table of Contents

- [min-ratio-cycle](#min-ratio-cycle)
  - [Table of Contents](#table-of-contents)
  - [Highlights](#highlights)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Analytics \& Visualization](#analytics--visualization)
  - [How It Works (Short)](#how-it-works-short)
  - [Background \& References](#background--references)
  - [Benchmarks](#benchmarks)
  - [Documentation](#documentation)
  - [Testing \& Quality](#testing--quality)
  - [Development](#development)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Maintainer](#maintainer)

---

## Highlights

* **Parametric search** over $\lambda$ with fast relaxations on weights $w_\lambda(e) = c(e) - \lambda\,t(e)$.
* **Exact mode** via Stern–Brocot for integer inputs (returns a rational $\lambda^*$).
* **Robustness**: topology and weight validation, clear exceptions, recovery hints.
* **Analytics**: sensitivity studies, stability regions, simple confidence intervals.
* **Visualization helpers** for cycles/ratios.
* **Benchmarks** with optional NetworkX comparisons.
* **Ergonomic results**: iterable `(cycle, cost, time, ratio)` pattern.

---

## Installation

This project uses Poetry.

```bash
poetry install
```

Tips

* Prefer a system BLAS (OpenBLAS/MKL) for faster NumPy.
* Enable pre-commit hooks after install:

```bash
poetry run pre-commit install
```

---

## Quick Start

```python
# Example API (module namespace: min_ratio_cycle)
from min_ratio_cycle.solver import MinRatioCycleSolver

# Create a 3-node directed graph
solver = MinRatioCycleSolver(n_nodes=3)

# Add edges (u -> v) with cost and time
solver.add_edge(0, 1, cost=2, time=1)
solver.add_edge(1, 2, cost=3, time=2)
solver.add_edge(2, 0, cost=1, time=1)

# Solve
cycle, cost, time, ratio = solver.solve()
print("Cycle:", cycle)
print("Cost:", cost, " Time:", time, " Ratio:", ratio)
```

* `cycle`: list of node indices forming the minimum ratio cycle.
* `ratio = cost / time`: minimum cost-per-time among all directed cycles.

> If your graph is strictly integer-weighted (costs and times), you can enable the exact mode to avoid floating-point drift.

```python
cycle, cost, time, ratio = solver.solve(exact=True)  # Stern–Brocot search
```

---

## Analytics & Visualization

Sensitivity and simple confidence intervals:

```python
from min_ratio_cycle.analytics import sensitivity_analysis, confidence_interval

# +10% cost on edge (0 -> 1)
perturb = {(0, 1): {"cost": +0.10}}
summary = sensitivity_analysis(solver, perturb)
ci = confidence_interval(samples=[ratio for _ in range(10)])
```

Visualize solution:

```python
result = solver.solve(return_object=True)
result.visualize_solution(show_cycle=True)
```

---

## How It Works (Short)

We search for the scalar parameter $\lambda$ such that no directed cycle has negative **mean cost** in the reweighted graph $G_\lambda$, where each edge weight is
$\; w_\lambda(e) = c(e) - \lambda\,t(e).$

* For a given $\lambda$, we run negative‑cycle detection (Bellman–Ford style relaxations) on $w_\lambda$.
* The minimum feasible $\lambda$ with **no** negative cycle equals the **minimum cost-to-time ratio** over all cycles.
* For integer inputs, an **exact** Stern–Brocot search avoids floating error and returns $\lambda^*$ as a rational.

This design offers practical speed (vectorized relaxations) and correctness (exact arithmetic when applicable).

---

## Background & References

This library is based on and follows the problem formulation from:

> **Karl Bringmann, Thomas Dueholm Hansen, Sebastian Krinninger** (ICALP 2017; arXiv:1704.08122),
> *Improved Algorithms for Computing the Cycle of Minimum Cost‑to‑Time Ratio in Directed Graphs.*

**Core ideas used here**

* Adopt the **parametric reduction**: reweight edges as $c - \lambda t$ and test for negative cycles.
* Implement a practical **decision oracle** with NumPy‑accelerated relaxations.
* Use **bisection** for $\lambda$ in floating mode and **Stern–Brocot** in exact integer mode.

**BibTeX** (please cite if you use this library in research):

```bibtex
@article{bringmann2017improved,
  title   = {Improved Algorithms for Computing the Cycle of Minimum Cost-to-Time Ratio in Directed Graphs},
  author  = {Karl Bringmann and Thomas Dueholm Hansen and Sebastian Krinninger},
  journal = {arXiv:1704.08122},
  year    = {2017},
  note    = {Accepted to ICALP 2017}
}
```

---

## Benchmarks

We include pytest markers to separate performance runs and comparisons.

```bash
# Micro-benchmarks
poetry run pytest -m benchmark

# Optional: parallelize
poetry run pytest -n auto -m benchmark
```

Programmatic entry:

```python
from min_ratio_cycle.benchmarks import benchmark_solver
runtime_s, ratio = benchmark_solver(solver)
print(f"{runtime_s:.6f}s -> ratio={ratio}")
```

Tips

* Pin NumPy and BLAS for stable timing.
* For very large graphs, consider tighter early‑exit tolerances.

---

## Documentation

Sphinx docs live under `docs/`. Build locally:

```bash
poetry run sphinx-build -b html docs docs/_build/html
```

A Read the Docs configuration is included for easy hosting.

---

## Testing & Quality

We use `pytest`, property-based tests, coverage, type checks, linting, and security scans (configured in `pyproject.toml`).

```bash
# Unit & property tests
poetry run pytest --cov=min_ratio_cycle

# Type checks
poetry run mypy min_ratio_cycle

# Style & lint
poetry run black . && poetry run isort .
poetry run flake8 .

# Security scan
poetry run bandit -r min_ratio_cycle
```

Enable hooks:

```bash
poetry run pre-commit install
```

---

## Development

* Package module: `min_ratio_cycle/`
* Tests: `tests/` with markers `slow`, `benchmark`, `property`, `integration`
* Build/publish: Poetry; convenience targets in `Makefile`

---

## Troubleshooting

* **Floating‑point sensitivity**: use `exact=True` for integer data.
* **Non‑positive times**: all `time` values must be strictly positive; validation fails early.
* **Large graphs**: ensure sufficient RAM; reduce warmups in benchmarks.
* **Unexpected ratios**: check units and ensure no edge has negative time.

---

## License

MIT — see `LICENSE`.

---

## Maintainer

**Diogo Ribeiro (DiogoRibeiro7)**
ESMAD – Instituto Politécnico do Porto
Personal: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)
Professional: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
ORCID: [https://orcid.org/0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
