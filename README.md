# min-ratio-cycle

[![CI](https://github.com/DiogoRibeiro7/min-ratio-cycle/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/min-ratio-cycle/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiogoRibeiro7/min-ratio-cycle/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/min-ratio-cycle)
[![Docs](https://readthedocs.org/projects/min-ratio-cycle/badge/?version=latest)](https://min-ratio-cycle.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?logo=python\&logoColor=white)

An optimized Python library for finding the **minimum cost-to-time ratio cycle** in a directed graph.

> Lawler-style parametric search with NumPy-accelerated negative‑cycle detection and an exact Stern–Brocot mode for integer weights.

---

## Table of Contents

- [min-ratio-cycle](#min-ratio-cycle)
  - [Table of Contents](#table-of-contents)
  - [Highlights](#highlights)
  - [Installation](#installation)
    - [From PyPI (after publishing)](#from-pypi-after-publishing)
    - [From source (development)](#from-source-development)
  - [Quick Start](#quick-start)
  - [Analytics \& Visualization](#analytics--visualization)
  - [How It Works (Short)](#how-it-works-short)
  - [Theory](#theory)
  - [Background \& References](#background--references)
  - [How to cite](#how-to-cite)
  - [Benchmarks](#benchmarks)
  - [Documentation](#documentation)
  - [Testing \& Quality](#testing--quality)
  - [Development](#development)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Maintainer](#maintainer)
  - [API surface (stable)](#api-surface-stable)
  - [Contributing](#contributing)
  - [Code of Conduct](#code-of-conduct)

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

### From PyPI (after publishing)

```bash
pip install min-ratio-cycle
```

### From source (development)

```bash
poetry install
poetry run pre-commit install
```

**Supported Python**: 3.10, 3.11, 3.12 (see `pyproject.toml`).

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

## Theory

We implement the classic **Lawler reduction** by reweighting edges with
$w_\lambda(e) = c(e) - \lambda\,t(e)$. A **negative-cycle oracle** on the
reweighted graph $G_\lambda$ acts as the decision procedure: if a negative cycle
exists, then $\lambda < \lambda^*$; if not, $\lambda \geq \lambda^*$.
We search for \lambda^\* by **bisection** in floating-point mode and via
**Stern–Brocot** in exact integer mode. Our implementation is **weakly polynomial**
(depends on numeric magnitudes) and optimized for practical performance. The
Bringmann–Hansen–Krinninger results are **strongly polynomial** under specific
models/assumptions and use parallelizable oracles; this library prioritizes
clarity and reproducibility over matching those asymptotics.

## Background & References

This library follows the problem formulation in:

> **Karl Bringmann, Thomas Dueholm Hansen, Sebastian Krinninger** (ICALP 2017; arXiv:1704.08122),
> *Improved Algorithms for Computing the Cycle of Minimum Cost‑to‑Time Ratio in Directed Graphs.*

**Core ideas used here**

* Adopt the **parametric reduction**: reweight edges as $c - \lambda t$ and test for negative cycles.
* Implement a practical **decision oracle** with NumPy‑accelerated relaxations.
* Use **bisection** for $\lambda$ in floating mode and **Stern–Brocot** in exact integer mode.
* For theoretical equivalence/stopping criteria, see **Lemma 2.2** and the **parametric search** summary (Section 2) in the paper.

**BibTeX (paper)**

```bibtex
@article{bringmann2017improved,
  title   = {Improved Algorithms for Computing the Cycle of Minimum Cost-to-Time Ratio in Directed Graphs},
  author  = {Karl Bringmann and Thomas Dueholm Hansen and Sebastian Krinninger},
  journal = {arXiv:1704.08122},
  year    = {2017},
  note    = {Accepted to ICALP 2017}
}
```

## How to cite

If you use this software, please cite the package and the paper.

* **Software**: see **[CITATION.cff](./CITATION.cff)** (GitHub renders multiple formats automatically).
* **Paper**: Bringmann–Hansen–Krinninger (2017), BibTeX:

```bibtex
@article{bringmann2017improved,
  title   = {Improved Algorithms for Computing the Cycle of Minimum Cost-to-Time Ratio in Directed Graphs},
  author  = {Karl Bringmann and Thomas Dueholm Hansen and Sebastian Krinninger},
  journal = {arXiv:1704.08122},
  year    = {2017},
  note    = {Accepted to ICALP 2017}
}
```

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

---

## API surface (stable)

**Public modules**

* `min_ratio_cycle.solver`

  * `class MinRatioCycleSolver(n_nodes: int)`

    * `add_edge(u: int, v: int, *, cost: float, time: float) -> None`
    * `add_edges(edges: Iterable[Edge]) -> None`
    * `solve(*, exact: bool = False, tol: float | None = None, max_iter: int | None = None, lambda_lower: float | None = None, lambda_upper: float | None = None, return_object: bool = False)` →

      * if `return_object=False`: `(cycle: list[int], cost: float, time: float, ratio: float)`
      * if `return_object=True`: a `Result` object with attributes `cycle`, `cost`, `time`, `ratio` and method `visualize_solution(show_cycle: bool = True)`
  * `@dataclass Edge(u: int, v: int, cost: float, time: float)`

* `min_ratio_cycle.analytics`

  * `sensitivity_analysis(solver: MinRatioCycleSolver, perturb: dict) -> dict`
  * `confidence_interval(samples: Iterable[float], alpha: float = 0.05) -> tuple[float, float]`

**Common `solve()` kwargs**

* `exact` *(bool)*: enable exact Stern–Brocot search for integer inputs. Default `False`.
* `tol` *(float | None)*: numeric tolerance for the decision oracle (negative‑cycle checks). If `None`, a sensible default is used.
* `max_iter` *(int | None)*: cap iterations/relaxations for the oracle (safety on large graphs).
* `lambda_lower`, `lambda_upper` *(float | None)*: optional bracket for the search over `λ`.
* `return_object` *(bool)*: return a rich result object with helpers.

> The API above is considered **stable**; breaking changes will be versioned with SemVer and documented in the changelog.

---

## Contributing

Please read **[CONTRIBUTING.md](./CONTRIBUTING.md)** for how to set up your environment, run tests, style/typing requirements, and our PR checklist. In short:

```bash
# 1) Setup
poetry install
poetry run pre-commit install

# 2) Test & quality
poetry run pytest --cov=min_ratio_cycle
poetry run mypy min_ratio_cycle
poetry run black . && poetry run isort .
poetry run flake8 .
poetry run bandit -r min_ratio_cycle
```

We use **Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`). Please add tests for new features and keep docs building (`sphinx-build`).

## Code of Conduct

This project adheres to the **[Code of Conduct](./CODE_OF_CONDUCT.md)**. By participating, you agree to uphold it. Enforcement contact: `dfr@esmad.ipp.pt`.
