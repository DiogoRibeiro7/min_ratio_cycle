# Min Ratio Cycle Solver

An optimized Python library for finding the **minimum cost-to-time ratio cycle** in a directed graph.

## Features

- Lawler parametric search with NumPy-accelerated Bellman–Ford relaxations
- Stern–Brocot exact mode for integer weights
- Comprehensive pre- and post-solve validation of topology, weight ranges, conditioning, and ratio correctness
- Rich exception hierarchy with recovery hints (`GraphStructureError`, `NumericalInstabilityError`, `ResourceExhaustionError`)
- Resource limits and graceful degradation with exact-mode fallbacks or relaxed tolerances
- Advanced analytics for sensitivity studies, stability region estimation, confidence intervals, convergence rates, and statistical comparisons
- Static and interactive visualisation helpers
- Benchmark suite with DIMACS loaders, regression baselines, and optional NetworkX comparisons
- Iterable solver results for intuitive tuple unpacking `(cycle, cost, time, ratio)`

## Installation

```bash
poetry install
```

## Quick start

```python
from min_ratio_cycle.solver import MinRatioCycleSolver, Edge

solver = MinRatioCycleSolver(3)
solver.add_edges([
    Edge(0, 1, cost=2, time=1),
    Edge(1, 2, cost=3, time=2),
    Edge(2, 0, cost=1, time=1),
])

cycle, cost, time, ratio = solver.solve()
print(cycle, ratio)
```

## Analytics

```python
from min_ratio_cycle.analytics import sensitivity_analysis, confidence_interval

perturb = {(0, 1): 0.1}
summary = sensitivity_analysis(solver, perturb)
ci = confidence_interval([ratio for _ in range(5)])
```

## Visualisation

```python
result = solver.solve()
result.visualize_solution(show_cycle=True)
```

## Benchmarking

```python
from min_ratio_cycle.benchmarks import benchmark_solver

runtime, ratio = benchmark_solver(solver)
```

## Documentation

The full user and API guides live in `docs/`. Build the Sphinx documentation:

```bash
poetry run sphinx-build -b html docs docs/_build/html
```

## Testing

```bash
poetry run pytest
```

## Maintainer

Diogo Ribeiro (DiogoRibeiro7)
ESMAD - Instituto Politécnico do Porto
Personal: diogo.debastos.ribeiro@gmail.com
Professional: dfr@esmad.ipp.pt
ORCID: https://orcid.org/0009-0001-2022-7072
