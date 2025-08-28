import time

import psutil

from min_ratio_cycle.solver import MinRatioCycleSolver


def _build_cycle_graph(n: int) -> MinRatioCycleSolver:
    solver = MinRatioCycleSolver(n)
    for i in range(n):
        solver.add_edge(i, (i + 1) % n, 1, 1)
        solver.add_edge((i + 1) % n, i, 1, 1)
    return solver


def test_solver_benchmark(large_graph, benchmark):
    """
    Benchmark the solver on a moderately sized graph.
    """
    benchmark(lambda: large_graph.solve())
    assert benchmark.stats["mean"] < 0.6


def test_no_memory_leak():
    """
    Solve many small graphs and ensure memory usage stays bounded.
    """
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    for _ in range(50):
        solver = _build_cycle_graph(5)
        solver.solve()
    rss_after = proc.memory_info().rss
    assert rss_after - rss_before < 5 * 1024 * 1024


def test_complexity_scaling():
    """
    Empirically verify near-quadratic scaling with graph size.
    """

    def time_solver(n):
        solver = _build_cycle_graph(n)
        start = time.perf_counter()
        solver.solve()
        return time.perf_counter() - start

    t_small = time_solver(20)
    t_large = time_solver(40)
    assert t_large < t_small * 5
