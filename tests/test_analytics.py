import matplotlib

matplotlib.use("Agg")
import pytest

from min_ratio_cycle.analytics import (
    compare_solutions,
    confidence_interval,
    convergence_rate,
)
from min_ratio_cycle.benchmarks import benchmark_solver, load_dimacs_graph
from min_ratio_cycle.solver import MinRatioCycleSolver


def _basic_solver() -> MinRatioCycleSolver:
    solver = MinRatioCycleSolver(3)
    solver.add_edge(0, 1, 1, 1)
    solver.add_edge(1, 2, 1, 1)
    solver.add_edge(2, 0, 1, 1)
    return solver


def test_sensitivity_and_stability():
    solver = _basic_solver()
    solver.solve()
    results = solver.sensitivity_analysis([{0: (0.5, 0.0)}])
    assert len(results) == 1
    stability = solver.stability_region(0.05)
    assert len(stability) == 3


def test_visualization_tools():
    solver = _basic_solver()
    solver.solve()
    fig, ax = solver.visualize_solution()
    assert fig is not None and ax is not None
    fig2 = solver.create_interactive_plot()
    assert fig2 is not None


def test_statistical_functions():
    ci_low, ci_high = confidence_interval([1, 2, 3, 4, 5])
    assert ci_low < ci_high
    rate = convergence_rate([1, 0.5, 0.25, 0.125])
    assert rate > 0
    t_stat = compare_solutions([1, 2, 3], [1, 2, 4])
    assert isinstance(t_stat, float)


def test_benchmark_utils():
    dimacs = [
        "p 5 7",
        "a 1 2 3 1",
        "a 2 3 4 1",
        "a 3 1 2 1",
        "a 2 4 1 1",
        "a 4 5 3 2",
        "a 5 2 0 1",
        "a 3 5 2 2",
    ]
    solver = load_dimacs_graph(dimacs)
    stats = benchmark_solver(solver)
    assert stats["diff"] == pytest.approx(0.0, abs=1e-9)
    assert stats["time"] >= 0.0


def test_benchmark_no_compare():
    dimacs = [
        "c sample graph",
        "p sp 3 3",
        "a 1 2 1 1",
        "a 2 3 1 1",
        "a 3 1 1 1",
    ]
    solver = load_dimacs_graph(dimacs)
    stats = benchmark_solver(solver, compare=False)
    assert stats["diff"] is None
    assert stats["ratio"] == pytest.approx(1.0, abs=1e-9)
