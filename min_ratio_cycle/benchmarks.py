"""Benchmark utilities and regression testing for the solver."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from .analytics import compare_solutions
from .solver import MinRatioCycleSolver


def load_dimacs_graph(lines: Iterable[str]) -> MinRatioCycleSolver:
    """Parse a directed graph from DIMACS ``.sp`` style lines.

    Parameters
    ----------
    lines : Iterable[str]
        Iterable of lines from a DIMACS file containing ``p`` and ``a``
        records.  Only the ``sp`` problem type is supported.

    Returns
    -------
    MinRatioCycleSolver
        Solver initialised with vertices and edges described by ``lines``.

    Raises
    ------
    ValueError
        If the input lines are malformed or missing required fields.

    Examples
    --------
    >>> with open("graph.dimacs") as fh:  # doctest: +SKIP
    ...     solver = load_dimacs_graph(fh)
    >>> solver.n  # doctest: +SKIP
    5
    """
    n_vertices = 0
    edges: List[Tuple[int, int, float, float]] = []
    for line in lines:
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[0] == "p":
            # DIMACS lines may look like "p <n> <m>" or "p sp <n> <m>".
            if len(parts) == 3:
                n_vertices = int(parts[1])
            else:
                n_vertices = int(parts[2])
        elif parts[0] == "a":
            u = int(parts[1]) - 1
            v = int(parts[2]) - 1
            cost = float(parts[3])
            time = float(parts[4])
            edges.append((u, v, cost, time))
    solver = MinRatioCycleSolver(n_vertices)
    solver.add_edges(edges)
    return solver


def compare_with_networkx(solver: MinRatioCycleSolver) -> float:
    """Compare solver's ratio against a brute-force enumeration.

    Parameters
    ----------
    solver : MinRatioCycleSolver
        Solver containing the graph to test.

    Returns
    -------
    float
        Absolute difference between the solver's ratio and the optimum
        found by :func:`networkx.simple_cycles`.

    Raises
    ------
    RuntimeError
        If ``solver.solve`` fails to produce a result.

    See Also
    --------
    benchmark_solver : Convenience wrapper that optionally invokes this
        comparison when benchmarking.
    """

    G = nx.DiGraph()
    for e in solver._edges:
        G.add_edge(e.u, e.v, cost=float(e.cost), time=float(e.time))

    best_ratio = float("inf")
    for cycle in nx.simple_cycles(G):
        cost = sum(G[u][v]["cost"] for u, v in zip(cycle, cycle[1:] + [cycle[0]]))
        time = sum(G[u][v]["time"] for u, v in zip(cycle, cycle[1:] + [cycle[0]]))
        ratio = cost / time
        if ratio < best_ratio:
            best_ratio = ratio

    result = solver.solve()
    return abs(result.ratio - best_ratio)


def benchmark_solver(
    solver: MinRatioCycleSolver, *, compare: bool = True
) -> Dict[str, Optional[float]]:
    """Run the solver and measure runtime and optional accuracy.

    Parameters
    ----------
    solver : MinRatioCycleSolver
        Initialised solver containing the graph to solve.
    compare : bool, optional
        If ``True``, compute the absolute ratio difference against
        :func:`compare_with_networkx`, by default ``True``.

    Returns
    -------
    dict
        Mapping with ``ratio`` from the solver, ``time`` taken in seconds, and
        ``diff`` if ``compare`` is ``True`` otherwise ``None``.

    See Also
    --------
    compare_with_networkx : Brute-force comparison routine.
    """

    start = perf_counter()
    result = solver.solve()
    elapsed = perf_counter() - start
    diff = compare_with_networkx(solver) if compare else None
    return {"ratio": result.ratio, "time": elapsed, "diff": diff}


def load_baseline(path: str | Path) -> Dict[str, object]:
    """Load a JSON baseline mapping.

    Parameters
    ----------
    path : str or Path
        File location of the baseline to read.

    Returns
    -------
    dict
        Baseline data with keys such as ``times`` and ``ratio``.

    Raises
    ------
    OSError
        Propagated if the file cannot be opened.
    """
    p = Path(path)
    with p.open() as fh:
        return json.load(fh)


def save_baseline(path: str | Path, data: Dict[str, object]) -> None:
    """Persist ``data`` as a JSON baseline.

    Parameters
    ----------
    path : str or Path
        Destination file for the baseline.
    data : dict
        Mapping to serialise.  Must be JSON serialisable.

    Raises
    ------
    OSError
        Propagated if the file cannot be written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(data, fh, indent=2)


def regression_check(
    times: Sequence[float],
    ratio: float,
    baseline: Dict[str, object],
    *,
    ratio_tol: float = 1e-9,
    t_threshold: float = 2.0,
) -> Dict[str, bool]:
    """Compare new measurements against a stored baseline.

    Parameters
    ----------
    times : Sequence[float]
        Runtime measurements in seconds for the current run.
    ratio : float
        Ratio produced by the solver for the current run.
    baseline : dict
        Mapping containing ``times`` and ``ratio`` from a previous run.
    ratio_tol : float, optional
        Allowed deviation in ratio before flagging a regression,
        by default ``1e-9``.
    t_threshold : float, optional
        Maximum allowed Welch t-statistic difference before flagging a
        performance regression, by default ``2.0``.

    Returns
    -------
    dict
        ``{"performance": bool, "quality": bool}`` where ``True`` indicates
        a regression was detected.
    """

    perf_reg = False
    if baseline.get("times"):
        try:
            t_stat = compare_solutions(times, baseline["times"])
            perf_reg = t_stat > t_threshold
        except ValueError:
            perf_reg = mean(times) > mean(baseline["times"]) * 1.5

    qual_reg = False
    if baseline.get("ratio") is not None:
        qual_reg = abs(ratio - float(baseline["ratio"])) > ratio_tol

    return {"performance": perf_reg, "quality": qual_reg}
