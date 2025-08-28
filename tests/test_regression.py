"""
Regression tests to track performance, quality, and API stability.
"""

from pathlib import Path

import min_ratio_cycle as mrc
from min_ratio_cycle.analytics import compare_solutions
from min_ratio_cycle.benchmarks import (
    benchmark_solver,
    load_baseline,
    load_dimacs_graph,
)


def test_solver_regressions():
    baseline_path = Path(__file__).with_name("baselines") / "triangle.json"
    baseline = load_baseline(baseline_path)

    lines = [
        "c simple triangle",
        "p 3 3",
        "a 1 2 1 1",
        "a 2 3 1 1",
        "a 3 1 1 1",
    ]

    times = []
    ratios = []
    for _ in range(3):
        solver = load_dimacs_graph(lines)
        stats = benchmark_solver(solver, compare=False)
        times.append(stats["time"])
        ratios.append(stats["ratio"])

    # Performance regression: mean runtime should not increase drastically
    t_stat = compare_solutions(times, baseline["times"])
    baseline_mean = sum(baseline["times"]) / len(baseline["times"])
    run_mean = sum(times) / len(times)
    assert t_stat < 5.0
    assert run_mean <= 1.5 * baseline_mean

    # Solution quality regression: ratios should match baseline
    assert all(abs(r - baseline["ratio"]) < 1e-9 for r in ratios)

    # Numerical stability regression: ratios constant across runs
    assert max(ratios) - min(ratios) < 1e-12

    # API compatibility regression: exported names unchanged
    assert set(mrc.__all__) == set(baseline["api"])
