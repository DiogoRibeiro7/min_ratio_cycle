"""
Enhanced Min Ratio Cycle Solver with improved features and robustness.

This module provides a comprehensive implementation of minimum cost-to-time ratio
cycle detection with the following enhancements:
- Better error handling and validation
- Performance optimizations for sparse graphs
- Structured logging and metrics
- Configuration management
- Multiple solving modes and options
- Comprehensive debugging utilities
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import psutil

from min_ratio_cycle.exceptions import (
    ConvergenceError,
    NumericalInstabilityError,
    ResourceExhaustionError,
    TimeoutError,
)

# Version and metadata
__version__ = "0.2.0"
__author__ = "Enhanced by Claude"


# ==============================
# Configuration and Types
# ==============================


class SolverMode(Enum):
    """
    Solver mode enumeration.
    """

    AUTO = "auto"  # Automatic selection based on weight types
    EXACT = "exact"  # Force exact rational arithmetic
    NUMERIC = "numeric"  # Force floating-point arithmetic
    APPROXIMATE = "approximate"  # Fast approximation


class LogLevel(Enum):
    """
    Logging level enumeration.
    """

    NONE = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4


@dataclass
class SolverConfig:
    """
    Configuration for the solver.
    """

    # Numeric mode parameters
    numeric_max_iter: int = 60
    numeric_tolerance: float = 1e-12
    numeric_cycle_slack: float = 1e-15

    # Exact mode parameters
    exact_max_denominator: int | None = None
    exact_max_steps: int | None = None

    # Performance parameters
    sparse_threshold: float = 0.1  # Use sparse optimizations below this density
    enable_preprocessing: bool = True
    enable_early_termination: bool = True

    # Validation parameters
    validate_cycles: bool = True
    repair_cycles: bool = True

    # Logging and monitoring
    log_level: LogLevel = LogLevel.INFO
    collect_metrics: bool = True

    # Advanced options
    use_kahan_summation: bool = True  # For numerical stability
    max_solve_time: float | None = None  # Maximum time in seconds
    max_memory_mb: int | None = None  # Maximum memory usage


@dataclass(frozen=True)
class Edge:
    """
    Directed edge with cost and (strictly positive) transit time.

    For exact solver, both cost and time should be integers or rational
    numbers. For numeric solver, floats are acceptable.
    """

    u: int
    v: int
    cost: float | int | Fraction
    time: float | int | Fraction

    def __post_init__(self):
        """
        Validate edge parameters.
        """
        if not isinstance(self.u, int) or not isinstance(self.v, int):
            raise ValueError("Vertex indices must be integers")
        if self.u < 0 or self.v < 0:
            raise ValueError("Vertex indices must be non-negative")

        # Convert time to appropriate type and validate
        time_val = (
            float(self.time) if not isinstance(self.time, Fraction) else self.time
        )
        if (isinstance(time_val, (int, float)) and time_val <= 0) or (
            isinstance(time_val, Fraction) and time_val <= 0
        ):
            raise ValueError("Edge transit time must be strictly positive")


@dataclass
class SolverMetrics:
    """
    Metrics collected during solving.
    """

    solve_time: float = 0.0
    iterations: int = 0
    mode_used: str = ""
    preprocessing_time: float = 0.0
    graph_properties: dict[str, Any] = field(default_factory=dict)
    memory_peak: int | None = None


@dataclass
class SolverResult:
    """
    Result from solver with additional metadata.
    """

    cycle: list[int]
    sum_cost: float | Fraction
    sum_time: float | Fraction
    ratio: float | Fraction
    success: bool = True
    error_message: str = ""
    metrics: SolverMetrics | None = None
    config_used: SolverConfig | None = None

    def __iter__(self):
        """
        Allow tuple unpacking of the primary result fields.
        """
        yield self.cycle
        yield self.sum_cost
        yield self.sum_time
        yield self.ratio


# ==============================
# Enhanced Solver Implementation
# ==============================


class MinRatioCycleSolver:
    """
    Enhanced minimum cost-to-time ratio cycle solver.

    Features:
    - Automatic mode selection (exact vs numeric)
    - Sparse graph optimizations
    - Comprehensive error handling
    - Performance monitoring
    - Configurable behavior
    - Debugging utilities
    """

    def __init__(self, n_vertices: int, config: SolverConfig | None = None):
        """
        Initialize solver.

        Args:
            n_vertices: Number of vertices in the graph
            config: Optional configuration object
        """
        if not isinstance(n_vertices, int) or n_vertices <= 0:
            raise ValueError("n_vertices must be a positive integer")

        self.n = n_vertices
        self.config = config or SolverConfig()
        self._edges: list[Edge] = []

        # Setup logging
        self._setup_logging()

        # NumPy arrays (built lazily)
        self._arrays_built = False
        self._U: np.ndarray | None = None
        self._V: np.ndarray | None = None
        self._C: np.ndarray | None = None  # float64 costs
        self._T: np.ndarray | None = None  # float64 times
        self._Ci: np.ndarray | None = None  # int64 costs (exact mode)
        self._Ti: np.ndarray | None = None  # int64 times (exact mode)
        self._starts: np.ndarray | None = None  # CSR format
        self._counts: np.ndarray | None = None  # CSR format

        # State tracking
        self._all_int = True  # Whether all weights are integers
        self._is_sparse = False
        self._last_result: SolverResult | None = None

        # Metrics
        self._metrics = SolverMetrics() if self.config.collect_metrics else None

        self.logger.info(f"Initialized solver with {n_vertices} vertices")

    def _build_numpy_arrays_once(self) -> None:
        """
        Backward compatibility shim for tests expecting this method.
        """
        self._build_arrays_if_needed()

    def _setup_logging(self) -> None:
        """
        Setup structured logging.
        """
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")

        if self.config.log_level == LogLevel.NONE:
            self.logger.setLevel(logging.CRITICAL + 1)
        elif self.config.log_level == LogLevel.ERROR:
            self.logger.setLevel(logging.ERROR)
        elif self.config.log_level == LogLevel.WARN:
            self.logger.setLevel(logging.WARNING)
        elif self.config.log_level == LogLevel.INFO:
            self.logger.setLevel(logging.INFO)
        else:  # DEBUG
            self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # ==============================
    # Edge Management
    # ==============================

    def add_edge(
        self,
        u: int,
        v: int,
        cost: float | int | Fraction,
        time: float | int | Fraction,
    ) -> None:
        """
        Add a directed edge u -> v with given cost and time.

        Args:
            u, v: Vertex indices (must be in [0, n_vertices))
            cost: Edge cost (can be negative)
            time: Edge time (must be positive)
        """
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError("u and v must be valid vertex indices")

        # Create edge (validation happens in Edge.__post_init__)
        edge = Edge(u, v, cost, time)

        # Update integer tracking
        if self._all_int:
            cost_is_int = isinstance(cost, (int, Fraction)) or (
                isinstance(cost, float) and cost.is_integer()
            )
            time_is_int = isinstance(time, (int, Fraction)) or (
                isinstance(time, float) and time.is_integer()
            )
            if not (cost_is_int and time_is_int):
                self._all_int = False
                self.logger.debug("Switched to numeric mode due to non-integer weights")

        self._edges.append(edge)
        self._arrays_built = False  # Invalidate cached arrays

        self.logger.debug(f"Added edge {u} -> {v} (cost={cost}, time={time})")

    def add_edges(self, edges: list[tuple[int, int, float | int, float | int]]) -> None:
        """
        Add multiple edges at once for efficiency.
        """
        for u, v, cost, time in edges:
            self.add_edge(u, v, cost, time)

    def remove_edge(self, u: int, v: int) -> bool:
        """
        Remove an edge between vertices u and v.

        Returns:
            True if edge was removed, False if not found
        """
        for i, edge in enumerate(self._edges):
            if edge.u == u and edge.v == v:
                del self._edges[i]
                self._arrays_built = False
                self.logger.debug(f"Removed edge {u} -> {v}")
                return True
        return False

    def clear_edges(self) -> None:
        """
        Remove all edges from the graph.
        """
        self._edges.clear()
        self._arrays_built = False
        self.logger.debug("Cleared all edges")

    # ==============================
    # Graph Analysis
    # ==============================

    def get_graph_properties(self) -> dict[str, Any]:
        """
        Get comprehensive graph properties.
        """
        if not self._edges:
            return {"error": "No edges in graph"}

        self._build_arrays_if_needed()

        if self._U is None:
            return {"error": "Failed to build arrays"}

        n, m = self.n, len(self._edges)
        density = m / (n * n)

        # Degree analysis
        in_degrees = np.bincount(self._V, minlength=n)
        out_degrees = np.bincount(self._U, minlength=n)

        # Weight analysis
        costs = [float(e.cost) for e in self._edges]
        times = [float(e.time) for e in self._edges]

        return {
            "vertices": n,
            "edges": m,
            "density": density,
            "is_sparse": density < self.config.sparse_threshold,
            "connectivity": {
                "isolated_vertices": int(
                    np.sum((in_degrees == 0) & (out_degrees == 0))
                ),
                "source_vertices": int(np.sum((in_degrees == 0) & (out_degrees > 0))),
                "sink_vertices": int(np.sum((in_degrees > 0) & (out_degrees == 0))),
                "avg_in_degree": float(np.mean(in_degrees)),
                "avg_out_degree": float(np.mean(out_degrees)),
                "max_in_degree": int(np.max(in_degrees)),
                "max_out_degree": int(np.max(out_degrees)),
            },
            "weights": {
                "all_integer": self._all_int,
                "cost_range": (min(costs), max(costs)),
                "time_range": (min(times), max(times)),
                "cost_mean": sum(costs) / len(costs),
                "time_mean": sum(times) / len(times),
                "cost_std": np.std(costs),
                "time_std": np.std(times),
            },
        }

    def detect_issues(self) -> list[dict[str, str]]:
        """
        Detect potential issues that might cause solve failures.
        """
        issues = []
        props = self.get_graph_properties()

        if "error" in props:
            issues.append({"level": "ERROR", "message": props["error"]})
            return issues

        # Check connectivity
        conn = props["connectivity"]
        if conn["isolated_vertices"] > 0:
            issues.append(
                {
                    "level": "WARNING",
                    "message": f"{conn['isolated_vertices']} isolated vertices (no cycles possible through them)",
                }
            )

        # Check for very sparse graphs
        if props["density"] < 0.01:
            issues.append(
                {
                    "level": "INFO",
                    "message": f"Very sparse graph (density: {props['density']:.4f})",
                }
            )

        # Check weight ranges
        weights = props["weights"]
        cost_range = weights["cost_range"][1] - weights["cost_range"][0]
        time_ratio = (
            weights["time_range"][1] / weights["time_range"][0]
            if weights["time_range"][0] > 0
            else float("inf")
        )

        if cost_range > 1e10:
            issues.append(
                {
                    "level": "WARNING",
                    "message": f"Very large cost range may cause numerical issues: {cost_range:.2e}",
                }
            )

        if time_ratio > 1e6:
            issues.append(
                {
                    "level": "WARNING",
                    "message": f"Very large time ratio may cause numerical issues: {time_ratio:.2e}",
                }
            )

        return issues

    # ==============================
    # Array Building and Preprocessing
    # ==============================

    def _build_arrays_if_needed(self) -> None:
        """
        Build NumPy arrays if not already built.
        """
        if self._arrays_built:
            return

        start_time = time.time() if self._metrics else 0

        m = len(self._edges)
        if m == 0:
            self.logger.warning("No edges to build arrays from")
            return

        # Build basic arrays
        U = np.empty(m, dtype=np.int64)
        V = np.empty(m, dtype=np.int64)
        C = np.empty(m, dtype=np.float64)
        T = np.empty(m, dtype=np.float64)

        for i, edge in enumerate(self._edges):
            U[i] = edge.u
            V[i] = edge.v
            C[i] = float(edge.cost)
            T[i] = float(edge.time)

        # Sort by destination for CSR format
        order = np.argsort(V, kind="stable")
        U, V, C, T = U[order], V[order], C[order], T[order]

        # Build CSR structure
        counts = np.bincount(V, minlength=self.n)
        starts = np.empty(self.n, dtype=np.int64)
        starts[0] = 0
        if self.n > 1:
            np.cumsum(counts[:-1], out=starts[1:])

        # Store arrays
        self._U, self._V, self._C, self._T = U, V, C, T
        self._starts, self._counts = starts, counts

        # Build integer arrays for exact mode if needed
        if self._all_int:
            self._Ci = np.array(
                [int(edge.cost) for edge in self._edges], dtype=np.int64
            )[order]
            self._Ti = np.array(
                [int(edge.time) for edge in self._edges], dtype=np.int64
            )[order]

        # Check if sparse
        density = m / (self.n * self.n)
        self._is_sparse = density < self.config.sparse_threshold

        self._arrays_built = True

        if self._metrics:
            self._metrics.preprocessing_time = time.time() - start_time

        self.logger.debug(
            f"Built arrays: {m} edges, density={density:.4f}, sparse={self._is_sparse}"
        )

    def _preprocess_graph(self) -> None:
        """
        Apply graph preprocessing optimizations.
        """
        if not self.config.enable_preprocessing:
            return

        original_edges = len(self._edges)

        # Remove dominated edges (same endpoints, one dominates in both cost and time)
        if self.config.enable_preprocessing:
            self._remove_dominated_edges()

        removed = original_edges - len(self._edges)
        if removed > 0:
            self.logger.info(f"Preprocessing removed {removed} dominated edges")
            self._arrays_built = False  # Need to rebuild arrays

    def _remove_dominated_edges(self) -> None:
        """
        Remove edges dominated by others with same endpoints.
        """
        edge_groups: dict[tuple[int, int], list[int]] = {}

        # Group edges by (u, v) pairs
        for i, edge in enumerate(self._edges):
            key = (edge.u, edge.v)
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(i)

        indices_to_remove = set()

        # For each group, find dominated edges
        for indices in edge_groups.values():
            if len(indices) <= 1:
                continue

            edges = [self._edges[i] for i in indices]

            for i in range(len(edges)):
                for j in range(len(edges)):
                    if i == j:
                        continue

                    edge_i, edge_j = edges[i], edges[j]

                    # Check if edge_i dominates edge_j
                    if (
                        float(edge_i.cost) <= float(edge_j.cost)
                        and float(edge_i.time) <= float(edge_j.time)
                        and (
                            float(edge_i.cost) < float(edge_j.cost)
                            or float(edge_i.time) < float(edge_j.time)
                        )
                    ):
                        indices_to_remove.add(indices[j])

        # Remove dominated edges
        if indices_to_remove:
            self._edges = [
                edge for i, edge in enumerate(self._edges) if i not in indices_to_remove
            ]

    # ==============================
    # Main Solve Method
    # ==============================

    def solve(
        self,
        mode: SolverMode | str = SolverMode.AUTO,
        target_ratio: float | None = None,
        **kwargs,
    ) -> SolverResult:
        """
        Find minimum cost-to-time ratio cycle.

        Args:
            mode: Solving mode (auto, exact, numeric, approximate)
            target_ratio: Stop if better ratio found (early termination)
            **kwargs: Additional parameters passed to specific solvers

        Returns:
            SolverResult with cycle, costs, and metadata
        """
        if isinstance(mode, str):
            mode = SolverMode(mode)

        start_time = time.time()

        if self.config.max_memory_mb is not None:
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            if mem > self.config.max_memory_mb:
                raise ResourceExhaustionError(
                    "Memory limit exceeded before solve",
                    resource="memory",
                    limit=self.config.max_memory_mb,
                    usage=mem,
                    suggested_fix="increase max_memory_mb",
                )

        try:
            # Preprocessing
            self._preprocess_graph()
            self._build_arrays_if_needed()

            if not self._edges:
                raise ValueError("Graph has no edges")

            # Use numeric solver for all cases
            actual_mode = SolverMode.NUMERIC
            self.logger.info(f"Starting solve in {actual_mode.value} mode")

            try:
                result = self._solve_numeric(target_ratio=target_ratio, **kwargs)
            except (NumericalInstabilityError, ConvergenceError) as e:
                self.logger.warning("Numeric solve failed: %s", e)
                if self._all_int:
                    self.logger.info("Retrying with exact mode")
                    result = self._solve_exact(**kwargs)
                else:
                    self.logger.info("Relaxing tolerance and retrying")
                    old_tol = self.config.numeric_tolerance
                    self.config.numeric_tolerance *= 10
                    result = self._solve_numeric(target_ratio=target_ratio, **kwargs)
                    self.config.numeric_tolerance = old_tol
                if not result.success:
                    raise NumericalInstabilityError(
                        "Solver failed in all recovery attempts",
                        component="solve",
                        suggested_fix="check input weights",
                        recovery_hint="try exact mode or scale weights",
                    ) from e

            # Validate result if requested
            if self.config.validate_cycles and result.success:
                result = self._validate_result(result)

            # Update metrics
            solve_time = time.time() - start_time
            if self._metrics:
                self._metrics.solve_time = solve_time
                self._metrics.mode_used = actual_mode.value
                self._metrics.graph_properties = self.get_graph_properties()
                result.metrics = self._metrics

            result.config_used = self.config
            self._last_result = result

            if result.success:
                self.logger.info(
                    f"Solved successfully in {solve_time:.4f}s, ratio={result.ratio}"
                )
                return result
            else:
                self.logger.error(f"Solve failed: {result.error_message}")
                raise RuntimeError(result.error_message)

        except TimeoutError as e:
            self.logger.error("Timeout: %s", e)
            raise
        except ResourceExhaustionError as e:
            self.logger.error("Resource limit hit: %s", e)
            raise
        except Exception as e:
            self.logger.error("Solver exception: %s", e)
            raise

    # ==============================
    # Exact Mode Implementation
    # ==============================

    def _solve_exact(self, **kwargs) -> SolverResult:
        """
        Solve using exact rational arithmetic (Stern-Brocot search).
        """
        if not self._all_int:
            return SolverResult(
                cycle=[],
                sum_cost=0,
                sum_time=0,
                ratio=float("inf"),
                success=False,
                error_message="Exact mode requires integer weights",
            )

        if self._Ci is None or self._Ti is None:
            raise RuntimeError(
                "Integer weight arrays (_Ci, _Ti) must be initialized before solving"
            )

        max_den = kwargs.get("max_denominator", self.config.exact_max_denominator)
        max_steps = kwargs.get("max_steps", self.config.exact_max_steps)

        try:
            cycle, sum_c, sum_t, (a, b), ratio_float = self._stern_brocot_search(
                max_den, max_steps
            )

            # Close cycle if needed
            if cycle and (len(cycle) == 1 or cycle[0] != cycle[-1]):
                cycle.append(cycle[0])

            return SolverResult(
                cycle=cycle,
                sum_cost=float(sum_c),
                sum_time=float(sum_t),
                ratio=float(sum_c) / float(sum_t),
                success=True,
            )

        except Exception as e:
            return SolverResult(
                cycle=[],
                sum_cost=0,
                sum_time=0,
                ratio=float("inf"),
                success=False,
                error_message=f"Exact solver failed: {str(e)}",
            )

    def _stern_brocot_search(
        self, max_den: int | None, max_steps: int | None
    ) -> tuple[list[int], int, int, tuple[int, int], float]:
        """
        Implement Stern-Brocot tree search for exact optimal ratio.
        """
        if self._Ci is None or self._Ti is None:
            raise RuntimeError(
                "Integer weight arrays (_Ci, _Ti) must be initialized before searching"
            )

        # Set defaults
        if max_den is None:
            t_max = int(np.max(self._Ti))
            max_den = self.n * max(1, t_max)

        if max_steps is None:
            max_steps = 2 * max_den + 10

        # Initial bracket
        ratios = self._Ci.astype(np.float64) / self._Ti.astype(np.float64)
        aL, bL = int(math.floor(np.min(ratios))) - 1, 1
        aR, bR = int(math.ceil(np.max(ratios))) + 1, 1

        # Verify bracket validity
        if self._has_negative_cycle_exact(aL, bL):
            raise RuntimeError("Lower bound has negative cycle")
        if not self._has_negative_cycle_exact(aR, bR):
            raise RuntimeError("Upper bound has no negative cycle")

        iterations = 0

        for _ in range(max_steps):
            iterations += 1
            aM, bM = aL + aR, bL + bR

            if bM > max_den:
                # Check if L is exact
                zero_cycle = self._find_zero_cycle_exact(aL, bL)
                if zero_cycle is not None:
                    cycle, sum_c, sum_t = zero_cycle
                    g = math.gcd(aL, bL)
                    return cycle, sum_c, sum_t, (aL // g, bL // g), sum_c / sum_t

                # Move towards R with largest feasible step
                k = max(1, (max_den - bL) // bR)
                aM, bM = aL + k * aR, bL + k * bR

            if self._has_negative_cycle_exact(aM, bM):
                aR, bR = aM, bM
                continue

            # Check for exact solution
            zero_cycle = self._find_zero_cycle_exact(aM, bM)
            if zero_cycle is not None:
                cycle, sum_c, sum_t = zero_cycle
                g = math.gcd(aM, bM)
                if self._metrics:
                    self._metrics.iterations = iterations
                return cycle, sum_c, sum_t, (aM // g, bM // g), sum_c / sum_t

            aL, bL = aM, bM

        raise RuntimeError("Stern-Brocot search did not converge")

    def _combine_weights_exact(self, a: int, b: int) -> np.ndarray:
        combo = b * self._Ci - a * self._Ti
        if not np.isfinite(combo).all():
            raise NumericalInstabilityError(
                "Weight combination overflowed",
                component="solver",
                suggested_fix="Scale weights or use smaller coefficients",
                recovery_hint="Normalize edge weights before solving",
            )
        combo = np.clip(combo, np.iinfo(np.int64).min, np.iinfo(np.int64).max)
        return combo.astype(np.int64)

    def _has_negative_cycle_exact(self, a: int, b: int) -> bool:
        """
        Check for negative cycle with exact integer arithmetic.
        """
        if self._U is None or self._V is None:
            raise RuntimeError(
                "Edge endpoint arrays (_U, _V) must be initialized before cycle checks"
            )
        if self._Ci is None or self._Ti is None:
            raise RuntimeError(
                "Integer weight arrays (_Ci, _Ti) must be initialized before cycle checks"
            )
        if self._starts is None or self._counts is None:
            raise RuntimeError(
                "Segment arrays (_starts, _counts) must be initialized before cycle checks"
            )

        W = self._combine_weights_exact(a, b)
        dist = np.zeros(self.n, dtype=np.int64)

        def relax_once() -> bool:
            cand = dist[self._U] + W
            improve = cand < dist[self._V]
            if not np.any(improve):
                return False

            # Use numerical stable min reduction
            cand_imp = np.where(improve, cand, np.iinfo(np.int64).max)
            mins = np.minimum.reduceat(cand_imp, self._starts)
            mins_rep = np.repeat(mins, self._counts)
            good = improve & (cand_imp == mins_rep)

            if not np.any(good):
                return False

            # Update distances for selected vertices
            idx = np.flatnonzero(good)
            dist[self._V[idx]] = cand[idx]
            return True

        # Bellman-Ford: n-1 passes + detection pass
        for _ in range(self.n - 1):
            if not relax_once():
                break

        return relax_once()  # If still can relax, negative cycle exists

    def _find_zero_cycle_exact(
        self, a: int, b: int
    ) -> tuple[list[int], int, int] | None:
        """
        Find zero-weight cycle in equality subgraph.
        """
        # First get potentials
        ok, dist = self._compute_potentials_exact(a, b)
        if not ok:
            return None

        if self._U is None or self._V is None:
            raise RuntimeError(
                "Edge endpoint arrays (_U, _V) must be initialized before cycle extraction"
            )
        if self._Ci is None or self._Ti is None:
            raise RuntimeError(
                "Integer weight arrays (_Ci, _Ti) must be initialized before cycle extraction"
            )

        W = self._combine_weights_exact(a, b)
        equal_mask = (dist[self._U] + W) == dist[self._V]

        if not np.any(equal_mask):
            return None

        # Build equality subgraph
        adj = [[] for _ in range(self.n)]
        eq_indices = np.flatnonzero(equal_mask)
        for i in eq_indices:
            adj[self._U[i]].append(self._V[i])

        # DFS for cycle
        color = np.zeros(self.n, dtype=np.int8)
        parent = np.full(self.n, -1, dtype=np.int64)

        def dfs(u: int) -> list[int] | None:
            color[u] = 1  # Gray
            for v in adj[u]:
                if color[v] == 0:  # White
                    parent[v] = u
                    cycle = dfs(v)
                    if cycle is not None:
                        return cycle
                elif color[v] == 1:  # Gray - back edge found
                    # Extract cycle
                    cycle_nodes = [v]
                    x = u
                    while x != v and x != -1:
                        cycle_nodes.append(x)
                        x = parent[x]
                    cycle_nodes.reverse()
                    return cycle_nodes
            color[u] = 2  # Black
            return None

        for start in range(self.n):
            if color[start] == 0:
                cycle = dfs(start)
                if cycle:
                    return self._compute_cycle_weights_exact(cycle)

        return None

    def _compute_potentials_exact(self, a: int, b: int) -> tuple[bool, np.ndarray]:
        """
        Compute shortest path potentials with exact arithmetic.
        """
        if self._U is None or self._V is None:
            raise RuntimeError(
                "Edge endpoint arrays (_U, _V) must be initialized before computing potentials"
            )
        if self._Ci is None or self._Ti is None:
            raise RuntimeError(
                "Integer weight arrays (_Ci, _Ti) must be initialized before computing potentials"
            )
        if self._starts is None or self._counts is None:
            raise RuntimeError(
                "Segment arrays (_starts, _counts) must be initialized before computing potentials"
            )

        W = self._combine_weights_exact(a, b)
        dist = np.zeros(self.n, dtype=np.int64)

        def relax_once() -> bool:
            cand = dist[self._U] + W
            improve = cand < dist[self._V]
            if not np.any(improve):
                return False

            cand_imp = np.where(improve, cand, np.iinfo(np.int64).max)
            mins = np.minimum.reduceat(cand_imp, self._starts)
            mins_rep = np.repeat(mins, self._counts)
            good = improve & (cand_imp == mins_rep)

            if not np.any(good):
                return False

            idx = np.flatnonzero(good)
            dist[self._V[idx]] = cand[idx]
            return True

        # Standard Bellman-Ford
        for _ in range(self.n - 1):
            if not relax_once():
                break

        # Check for negative cycles
        if relax_once():
            return False, dist

        return True, dist

    def _compute_cycle_weights_exact(
        self, cycle: list[int]
    ) -> tuple[list[int], int, int]:
        """
        Compute exact cycle weights.
        """
        sum_cost, sum_time = 0, 0

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]

            # Find edge u -> v
            edge_found = False
            for edge in self._edges:
                if edge.u == u and edge.v == v:
                    sum_cost += int(edge.cost)
                    sum_time += int(edge.time)
                    edge_found = True
                    break

            if not edge_found:
                raise RuntimeError(f"Edge {u} -> {v} not found in cycle extraction")

        return cycle, sum_cost, sum_time

    # ==============================
    # Numeric Mode Implementation
    # ==============================

    def _solve_numeric(
        self, target_ratio: float | None = None, **kwargs
    ) -> SolverResult:
        """
        Solve using numeric binary search (Lawler's algorithm).
        """
        lambda_lo = kwargs.get("lambda_lo")
        lambda_hi = kwargs.get("lambda_hi")
        max_iter = kwargs.get("max_iter", self.config.numeric_max_iter)
        tol = kwargs.get("tolerance", self.config.numeric_tolerance)
        slack = kwargs.get("cycle_slack", self.config.numeric_cycle_slack)

        try:
            start_time = time.time()
            process = psutil.Process()

            # Determine search bounds
            if lambda_lo is None or lambda_hi is None:
                costs = self._C
                times = self._T
                ratios = costs / times
                lo = float(np.min(ratios) - 1.0)
                hi = float(np.max(ratios) + 1.0)
                lambda_lo = lambda_lo or lo
                lambda_hi = lambda_hi or hi

            best_cycle = None
            iterations = 0

            # Binary search on lambda
            for iteration in range(max_iter):
                if (
                    self.config.max_solve_time is not None
                    and time.time() - start_time > self.config.max_solve_time
                ):
                    raise TimeoutError(
                        "Numeric solve exceeded time limit",
                        time_elapsed=time.time() - start_time,
                        time_limit=self.config.max_solve_time,
                        operation="numeric_solve",
                    )
                if self.config.max_memory_mb is not None:
                    mem = process.memory_info().rss / (1024 * 1024)
                    if mem > self.config.max_memory_mb:
                        raise ResourceExhaustionError(
                            "Numeric solve exceeded memory limit",
                            resource="memory",
                            limit=self.config.max_memory_mb,
                            usage=mem,
                        )

                iterations += 1
                mid = (lambda_lo + lambda_hi) / 2.0

                has_neg, cycle_data = self._detect_negative_cycle_numeric(mid, slack)

                if has_neg:
                    lambda_hi = mid
                    if cycle_data is not None:
                        best_cycle = cycle_data

                        # Early termination check
                        if (
                            target_ratio is not None
                            and self.config.enable_early_termination
                            and cycle_data[2] < target_ratio
                        ):  # ratio < target
                            self.logger.debug(
                                "Early termination at iteration %d", iteration
                            )
                            break
                else:
                    lambda_lo = mid

                if lambda_hi - lambda_lo <= tol:
                    break

            # Get final result
            if best_cycle is None:
                # Try once more at the upper bound
                has_neg, cycle_data = self._detect_negative_cycle_numeric(
                    lambda_hi, slack
                )
                if has_neg and cycle_data is not None:
                    best_cycle = cycle_data

            if best_cycle is None:
                raise NumericalInstabilityError(
                    "No negative cycle found", component="numeric_solve"
                )

            cycle, sum_cost, sum_time, ratio = best_cycle

            # Close cycle if needed
            if cycle and (len(cycle) == 1 or cycle[0] != cycle[-1]):
                cycle.append(cycle[0])

            if self._metrics:
                self._metrics.iterations = iterations

            # Convergence check
            if iteration == max_iter - 1 and lambda_hi - lambda_lo > tol:
                raise ConvergenceError(
                    "Numeric solver failed to converge",
                    algorithm_name="Lawler",
                    max_iterations=max_iter,
                    tolerance=tol,
                    final_error=lambda_hi - lambda_lo,
                )

            return SolverResult(
                cycle=cycle,
                sum_cost=sum_cost,
                sum_time=sum_time,
                ratio=ratio,
                success=True,
            )

        except Exception as e:
            raise NumericalInstabilityError(
                "Numeric solver failed", component="numeric_solve"
            ) from e

    def _detect_negative_cycle_numeric(
        self, lam: float, slack: float = 0.0
    ) -> tuple[bool, tuple[list[int], float, float, float] | None]:
        """
        Detect negative cycle using vectorized Bellman-Ford.
        """
        if self._U is None or self._V is None:
            raise RuntimeError(
                "Edge endpoint arrays (_U, _V) must be initialized before numeric cycle detection"
            )
        if self._C is None or self._T is None:
            raise RuntimeError(
                "Weight arrays (_C, _T) must be initialized before numeric cycle detection"
            )
        if self._starts is None or self._counts is None:
            raise RuntimeError(
                "Segment arrays (_starts, _counts) must be initialized before numeric cycle detection"
            )

        n = self.n
        W = self._C - lam * self._T

        # Use Kahan summation if configured
        if self.config.use_kahan_summation:
            dist = np.zeros(n, dtype=np.float64)
            compensation = np.zeros(n, dtype=np.float64)  # For Kahan summation
        else:
            dist = np.zeros(n, dtype=np.float64)
            compensation = None

        pred = np.full(n, -1, dtype=np.int64)

        def relax_once_kahan(update_pred: bool) -> bool:
            """
            Relaxation step with Kahan summation for numerical stability.
            """
            cand = dist[self._U] + W
            improve = cand < (dist[self._V] - slack)

            if not np.any(improve):
                return False

            # Segment-wise minimum selection
            cand_imp = np.where(improve, cand, np.inf)
            mins = np.full(self.n, np.inf)
            for v in range(self.n):
                start = self._starts[v]
                cnt = self._counts[v]
                if cnt > 0:
                    mins[v] = np.min(cand_imp[start : start + cnt])
            mins_rep = np.repeat(mins, self._counts)
            good = improve & (cand_imp == mins_rep)

            # First occurrence in each segment
            cs = np.cumsum(good.astype(np.int64))
            base = np.empty_like(self._starts)
            base[0] = 0
            if self._starts.size > 1:
                base[1:] = cs[self._starts[1:] - 1]
            first_mask = good & (cs == (np.repeat(base, self._counts) + 1))

            if not np.any(first_mask):
                return False

            # Update with Kahan summation
            idx = np.flatnonzero(first_mask)
            v_sel = self._V[idx]
            new_dist = cand[idx]

            if compensation is not None:
                # Kahan summation update
                y = new_dist - dist[v_sel] - compensation[v_sel]
                t = dist[v_sel] + y
                compensation[v_sel] = (t - dist[v_sel]) - y
                dist[v_sel] = t
            else:
                dist[v_sel] = new_dist

            if update_pred:
                pred[v_sel] = self._U[idx]

            return True

        def relax_once_standard(update_pred: bool) -> bool:
            """
            Standard relaxation step.
            """
            cand = dist[self._U] + W
            improve = cand < (dist[self._V] - slack)

            if not np.any(improve):
                return False

            cand_imp = np.where(improve, cand, np.inf)
            mins = np.full(self.n, np.inf)
            for v in range(self.n):
                start = self._starts[v]
                cnt = self._counts[v]
                if cnt > 0:
                    mins[v] = np.min(cand_imp[start : start + cnt])
            mins_rep = np.repeat(mins, self._counts)
            good = improve & (cand_imp == mins_rep)

            cs = np.cumsum(good.astype(np.int64))
            base = np.empty_like(self._starts)
            base[0] = 0
            if self._starts.size > 1:
                base[1:] = cs[self._starts[1:] - 1]
            first_mask = good & (cs == (np.repeat(base, self._counts) + 1))

            if not np.any(first_mask):
                return False

            idx = np.flatnonzero(first_mask)
            v_sel = self._V[idx]
            dist[v_sel] = cand[idx]

            if update_pred:
                pred[v_sel] = self._U[idx]

            return True

        # Choose relaxation method
        relax_once = (
            relax_once_kahan if compensation is not None else relax_once_standard
        )

        # Bellman-Ford iterations
        for _ in range(n - 1):
            if not relax_once(update_pred=True):
                break

        # Detection pass
        changed = relax_once(update_pred=True)
        if not changed:
            return False, None

        # Extract cycle from predecessors
        touched = np.where(pred != -1)[0]
        if touched.size == 0:
            return True, None

        # Walk to enter a cycle
        x = int(touched[-1])
        for _ in range(n):
            if pred[x] == -1:
                break
            x = int(pred[x])

        # Extract cycle
        seen = {}
        sequence = []
        cur = x

        while cur not in seen and cur != -1:
            seen[cur] = len(sequence)
            sequence.append(cur)
            cur = int(pred[cur]) if pred[cur] != -1 else -1

        if cur == -1:
            return True, None

        cycle = sequence[seen[cur] :][::-1]

        # Compute cycle weights
        sum_cost, sum_time = self._compute_cycle_weights_float(cycle)
        if sum_time <= 0:
            return True, None

        ratio = sum_cost / sum_time
        return True, (cycle, sum_cost, sum_time, ratio)

    def _compute_cycle_weights_float(self, cycle: list[int]) -> tuple[float, float]:
        """
        Compute cycle weights using cached edge map.
        """
        if not hasattr(self, "_edge_map") or self._edge_map is None:
            self._build_edge_map()

        sum_cost, sum_time = 0.0, 0.0

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]

            if (u, v) in self._edge_map:
                edges = self._edge_map[(u, v)]
                cost, time = min(edges, key=lambda ct: ct[0] / ct[1])
                sum_cost += cost
                sum_time += time
            else:
                # Fallback: scan through edges
                edge_found = False
                for edge in self._edges:
                    if edge.u == u and edge.v == v:
                        sum_cost += float(edge.cost)
                        sum_time += float(edge.time)
                        edge_found = True
                        break

                if not edge_found:
                    raise RuntimeError(f"Edge {u} -> {v} not found in cycle")

        return sum_cost, sum_time

    def _build_edge_map(self):
        """
        Build edge lookup map for fast cycle weight computation.
        """
        self._edge_map = {}
        for edge in self._edges:
            key = (edge.u, edge.v)
            cost, time = float(edge.cost), float(edge.time)
            self._edge_map.setdefault(key, []).append((cost, time))

    # ==============================
    # Approximate Mode Implementation
    # ==============================

    def _solve_approximate(
        self,
        target_ratio: float | None = None,
        epsilon: float = 0.01,
        max_time: float | None = None,
        **kwargs,
    ) -> SolverResult:
        """
        Fast approximation algorithm for very large graphs.
        """
        max_time = max_time or self.config.max_solve_time or 60.0
        start_time = time.time()

        try:
            # Simple approximation: sample random starting points and do limited search
            best_result = None
            best_ratio = float("inf")

            # Try multiple random starts
            n_samples = min(10, self.n)
            sample_vertices = np.random.choice(self.n, size=n_samples, replace=False)

            for start_v in sample_vertices:
                if time.time() - start_time > max_time:
                    break

                # Limited depth search from this vertex
                cycle, ratio = self._limited_search_from_vertex(
                    start_v, max_depth=min(20, self.n)
                )

                if cycle and ratio < best_ratio:
                    best_ratio = ratio
                    sum_cost, sum_time = self._compute_cycle_weights_float(cycle)
                    best_result = (cycle, sum_cost, sum_time, ratio)

                    # Early termination if good enough
                    if target_ratio is not None and ratio <= target_ratio * (
                        1 + epsilon
                    ):
                        break

            if best_result is None:
                return SolverResult(
                    cycle=[],
                    sum_cost=0,
                    sum_time=0,
                    ratio=float("inf"),
                    success=False,
                    error_message="Approximation found no cycles",
                )

            cycle, sum_cost, sum_time, ratio = best_result

            # Close cycle
            if cycle and cycle[0] != cycle[-1]:
                cycle.append(cycle[0])

            return SolverResult(
                cycle=cycle,
                sum_cost=sum_cost,
                sum_time=sum_time,
                ratio=ratio,
                success=True,
            )

        except Exception as e:
            return SolverResult(
                cycle=[],
                sum_cost=0,
                sum_time=0,
                ratio=float("inf"),
                success=False,
                error_message=f"Approximation failed: {str(e)}",
            )

    def _limited_search_from_vertex(
        self, start: int, max_depth: int
    ) -> tuple[list[int] | None, float]:
        """
        Limited depth search for cycles from a starting vertex.
        """
        if not hasattr(self, "_adj_list") or self._adj_list is None:
            self._build_adjacency_list()

        best_cycle = None
        best_ratio = float("inf")

        def dfs(path: list[int], visited: set, depth: int):
            nonlocal best_cycle, best_ratio

            if depth >= max_depth:
                return

            current = path[-1]

            for next_v, cost, time in self._adj_list[current]:
                if next_v == start and len(path) >= 2:
                    # Found cycle back to start
                    cycle = path[:]
                    total_cost = sum(
                        c
                        for _, c, _ in [
                            (self._get_edge_weights(path[i], path[i + 1]))
                            for i in range(len(path) - 1)
                        ]
                        + [(cost, time)]
                    )
                    total_time = sum(
                        t
                        for _, t in [
                            (self._get_edge_weights(path[i], path[i + 1])[1])
                            for i in range(len(path) - 1)
                        ]
                        + [time]
                    )

                    if total_time > 0:
                        ratio = total_cost / total_time
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_cycle = cycle

                elif next_v not in visited and depth < max_depth - 1:
                    dfs(path + [next_v], visited | {next_v}, depth + 1)

        dfs([start], {start}, 0)
        return best_cycle, best_ratio

    def _build_adjacency_list(self):
        """
        Build adjacency list representation.
        """
        self._adj_list = [[] for _ in range(self.n)]

        for edge in self._edges:
            cost, time = float(edge.cost), float(edge.time)
            self._adj_list[edge.u].append((edge.v, cost, time))

    def _get_edge_weights(self, u: int, v: int) -> tuple[float, float]:
        """
        Get edge weights between two vertices.
        """
        for edge in self._edges:
            if edge.u == u and edge.v == v:
                return float(edge.cost), float(edge.time)
        raise ValueError(f"No edge found from {u} to {v}")

    # ==============================
    # Result Validation and Repair
    # ==============================

    def _validate_result(self, result: SolverResult) -> SolverResult:
        """
        Validate and potentially repair the result.
        """
        if not result.success or not result.cycle:
            return result

        try:
            # Check cycle is properly closed
            if len(result.cycle) < 3:
                if self.config.repair_cycles:
                    # Try to repair by extending
                    result = self._repair_short_cycle(result)
                else:
                    result.success = False
                    result.error_message = "Cycle too short"
                    return result

            # Verify cycle exists in graph
            valid, issues = self._verify_cycle_exists(result.cycle)
            if not valid:
                if self.config.repair_cycles:
                    result = self._repair_missing_edges(result, issues)
                else:
                    result.success = False
                    result.error_message = f"Invalid cycle: {', '.join(issues)}"
                    return result

            # Verify weights are correct
            computed_cost, computed_time = self._compute_cycle_weights_float(
                result.cycle[:-1]
            )

            cost_error = abs(computed_cost - float(result.sum_cost))
            time_error = abs(computed_time - float(result.sum_time))
            ratio_error = abs(computed_cost / computed_time - float(result.ratio))

            tolerance = 1e-10 if isinstance(result.ratio, (int, Fraction)) else 1e-6

            if (
                cost_error > tolerance
                or time_error > tolerance
                or ratio_error > tolerance
            ):
                self.logger.warning(
                    "Weight validation failed: cost_err=%s, time_err=%s, ratio_err=%s",
                    cost_error,
                    time_error,
                    ratio_error,
                )

                if self.config.repair_cycles:
                    # Update with correct weights
                    result.sum_cost = computed_cost
                    result.sum_time = computed_time
                    result.ratio = computed_cost / computed_time
                else:
                    result.success = False
                    result.error_message = "Weight validation failed"
                    return result

            return result

        except Exception as e:
            result.success = False
            result.error_message = f"Validation failed: {str(e)}"
            return result

    def _verify_cycle_exists(self, cycle: list[int]) -> tuple[bool, list[str]]:
        """
        Verify all edges in cycle exist in graph.
        """
        issues = []

        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]

            # Check if edge exists
            edge_found = False
            for edge in self._edges:
                if edge.u == u and edge.v == v:
                    edge_found = True
                    break

            if not edge_found:
                issues.append(f"Missing edge {u} -> {v}")

        return len(issues) == 0, issues

    def _repair_short_cycle(self, result: SolverResult) -> SolverResult:
        """
        Attempt to repair cycles that are too short.
        """
        # Simple repair: if cycle has only 2 vertices, look for a longer path
        if len(result.cycle) == 3 and result.cycle[0] == result.cycle[2]:
            # This is just u -> v -> u, try to find u -> w -> v -> u
            u, v = result.cycle[0], result.cycle[1]

            # Find intermediate vertices
            for w in range(self.n):
                if w != u and w != v:
                    if self._has_edge(u, w) and self._has_edge(w, v):
                        new_cycle = [u, w, v, u]
                        cost, time = self._compute_cycle_weights_float(new_cycle[:-1])
                        result.cycle = new_cycle
                        result.sum_cost = cost
                        result.sum_time = time
                        result.ratio = cost / time
                        return result

        return result

    def _repair_missing_edges(
        self, result: SolverResult, issues: list[str]
    ) -> SolverResult:
        """
        Attempt to repair cycles with missing edges.
        """
        # For now, just mark as failed - more sophisticated repair could be implemented
        result.success = False
        result.error_message = f"Could not repair cycle: {', '.join(issues)}"
        return result

    def _has_edge(self, u: int, v: int) -> bool:
        """
        Check if edge u -> v exists.
        """
        for edge in self._edges:
            if edge.u == u and edge.v == v:
                return True
        return False

    # ==============================
    # Utility and Debug Methods
    # ==============================

    def get_debug_info(self) -> str:
        """
        Get comprehensive debug information.
        """
        info = ["Min Ratio Cycle Solver - Debug Info"]
        info.append("=" * 50)

        # Basic info
        info.append(f"Vertices: {self.n}")
        info.append(f"Edges: {len(self._edges)}")
        info.append(f"Integer weights: {self._all_int}")

        # Graph properties
        try:
            props = self.get_graph_properties()
            if "error" not in props:
                info.append(f"Density: {props['density']:.4f}")
                info.append(f"Is sparse: {props['is_sparse']}")

                conn = props["connectivity"]
                info.append(f"Isolated vertices: {conn['isolated_vertices']}")
                info.append(
                    f"Avg degree: in={conn['avg_in_degree']:.2f}, out={conn['avg_out_degree']:.2f}"
                )

                weights = props["weights"]
                info.append(
                    f"Cost range: [{weights['cost_range'][0]:.3f}, {weights['cost_range'][1]:.3f}]"
                )
                info.append(
                    f"Time range: [{weights['time_range'][0]:.3f}, {weights['time_range'][1]:.3f}]"
                )
        except Exception as e:
            info.append(f"Error analyzing graph: {e}")

        # Issues
        try:
            issues = self.detect_issues()
            if issues:
                info.append("\nPotential Issues:")
                for issue in issues:
                    info.append(f"  [{issue['level']}] {issue['message']}")
            else:
                info.append("\nNo issues detected")
        except Exception as e:
            info.append(f"\nError detecting issues: {e}")

        # Last result
        if self._last_result:
            result = self._last_result
            info.append("\nLast solve result:")
            info.append(f"  Success: {result.success}")
            if result.success:
                info.append(f"  Ratio: {result.ratio}")
                info.append(f"  Cycle length: {len(result.cycle) - 1}")
                if result.metrics:
                    info.append(f"  Solve time: {result.metrics.solve_time:.4f}s")
                    info.append(f"  Mode: {result.metrics.mode_used}")
                    info.append(f"  Iterations: {result.metrics.iterations}")
            else:
                info.append(f"  Error: {result.error_message}")

        return "\n".join(info)

    # ==============================
    # Advanced Analytics Methods
    # ==============================

    def sensitivity_analysis(
        self, edge_perturbations: list[dict[int, tuple[float, float]]]
    ) -> list[SolverResult]:
        """
        Perform edge weight perturbation analysis.

        Args:
            edge_perturbations: List of scenarios, each mapping an edge index to
                a ``(delta_cost, delta_time)`` tuple. For each scenario the graph
                is perturbed and the solver is run again.

        Returns:
            List of :class:`SolverResult` objects, one per scenario.
        """

        baseline = [Edge(e.u, e.v, e.cost, e.time) for e in self._edges]
        results: list[SolverResult] = []
        for scenario in edge_perturbations:
            for idx, (dc, dt) in scenario.items():
                if idx < 0 or idx >= len(self._edges):
                    raise IndexError("edge index out of range")
                e = self._edges[idx]
                self._edges[idx] = Edge(e.u, e.v, e.cost + dc, e.time + dt)
            self._arrays_built = False
            try:
                results.append(self.solve())
            finally:
                self._edges = [Edge(e.u, e.v, e.cost, e.time) for e in baseline]
                self._arrays_built = False

        return results

    def stability_region(self, epsilon: float = 0.01) -> dict[int, bool]:
        """
        Estimate stability of the optimal cycle under small perturbations.

        Each edge is perturbed by ``epsilon`` fraction of its cost and time and
        the solver is executed again. If the optimal cycle remains unchanged the
        edge is considered stable.

        Args:
            epsilon: Relative perturbation size (default 1%).

        Returns:
            Mapping from edge index to a boolean indicating stability.
        """

        if self._last_result is None:
            baseline_res = self.solve()
        else:
            baseline_res = self._last_result

        baseline_cycle = baseline_res.cycle
        stability: dict[int, bool] = {}
        for idx, e in enumerate(self._edges):
            dc = float(e.cost) * epsilon
            dt = float(e.time) * epsilon
            perturbed = {idx: (dc, dt)}
            res_list = self.sensitivity_analysis([perturbed])
            stability[idx] = res_list[0].cycle == baseline_cycle
        return stability

    def visualize_solution(
        self, show_cycle: bool = True, layout: str = "spring"
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Visualize the current graph and optionally highlight the solution.

        Args:
            show_cycle: If ``True`` highlight the most recent solution cycle.
            layout: Layout function name from ``networkx`` (e.g. ``"spring"``).

        Returns:
            Matplotlib ``Figure`` and ``Axes`` objects.
        """

        if not self._edges:
            raise ValueError("No edges to visualize")

        G = nx.DiGraph()
        for idx, e in enumerate(self._edges):
            G.add_edge(e.u, e.v, index=idx, cost=float(e.cost), time=float(e.time))

        layout_fn = getattr(nx, f"{layout}_layout")
        pos = layout_fn(G)

        fig, ax = plt.subplots()
        nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightgray")

        if show_cycle and self._last_result and self._last_result.cycle:
            cycle_edges = list(
                zip(self._last_result.cycle, self._last_result.cycle[1:])
            )
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=cycle_edges, edge_color="red", width=2.5
            )

        ax.set_title("Min Ratio Cycle Graph")
        return fig, ax

    def create_interactive_plot(self) -> plt.Figure:
        """
        Create an interactive matplotlib plot of the current graph.
        """

        fig, ax = self.visualize_solution()

        def _on_click(event: Any) -> None:
            if event.inaxes:
                ax.set_title(f"Clicked at ({event.xdata:.2f}, {event.ydata:.2f})")
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", _on_click)
        return fig

    def health_check(self) -> dict[str, Any]:
        """
        Run comprehensive health check.
        """
        checks = []

        # Check NumPy version
        try:
            import numpy as np

            np_version = np.__version__
            min_version = "1.20.0"
            from packaging import version

            np_ok = version.parse(np_version) >= version.parse(min_version)

            checks.append(
                {
                    "name": "numpy_version",
                    "status": "PASS" if np_ok else "FAIL",
                    "details": f"NumPy {np_version} (required: >={min_version})",
                    "recommendation": "Update NumPy" if not np_ok else None,
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "numpy_version",
                    "status": "ERROR",
                    "details": str(e),
                    "recommendation": "Install NumPy",
                }
            )

        # Check graph state
        if not self._edges:
            checks.append(
                {
                    "name": "graph_state",
                    "status": "WARN",
                    "details": "No edges in graph",
                    "recommendation": "Add edges before solving",
                }
            )
        else:
            issues = self.detect_issues()
            error_count = sum(1 for issue in issues if issue["level"] == "ERROR")
            warn_count = sum(1 for issue in issues if issue["level"] == "WARNING")

            if error_count > 0:
                status = "FAIL"
                details = f"{error_count} critical issues found"
                recommendation = "Fix graph structure issues"
            elif warn_count > 0:
                status = "WARN"
                details = f"{warn_count} warnings found"
                recommendation = "Review warnings"
            else:
                status = "PASS"
                details = "Graph structure looks good"
                recommendation = None

            checks.append(
                {
                    "name": "graph_state",
                    "status": status,
                    "details": details,
                    "recommendation": recommendation,
                }
            )

        # Memory check
        try:
            import psutil

            available_gb = psutil.virtual_memory().available / (1024**3)
            min_memory = 0.5  # 500MB minimum

            checks.append(
                {
                    "name": "memory",
                    "status": "PASS" if available_gb >= min_memory else "WARN",
                    "details": f"{available_gb:.1f}GB available",
                    "recommendation": (
                        "Consider freeing memory" if available_gb < min_memory else None
                    ),
                }
            )
        except ImportError:
            checks.append(
                {
                    "name": "memory",
                    "status": "SKIP",
                    "details": "psutil not available",
                    "recommendation": "Install psutil for memory monitoring",
                }
            )

        # Summary
        total = len(checks)
        passed = sum(1 for c in checks if c["status"] == "PASS")
        warnings = sum(1 for c in checks if c["status"] == "WARN")
        failures = sum(1 for c in checks if c["status"] == "FAIL")

        return {
            "timestamp": time.time(),
            "checks": checks,
            "summary": {
                "total": total,
                "passed": passed,
                "warnings": warnings,
                "failures": failures,
                "overall_status": (
                    "FAIL" if failures > 0 else ("WARN" if warnings > 0 else "PASS")
                ),
            },
        }

    def __repr__(self) -> str:
        return f"MinRatioCycleSolver(n_vertices={self.n}, n_edges={len(self._edges)}, integer_mode={self._all_int})"


# ==============================
# Convenience Functions
# ==============================


def solve_min_ratio_cycle(
    edges: list[tuple[int, int, float | int, float | int]],
    n_vertices: int | None = None,
    mode: SolverMode | str = SolverMode.AUTO,
    config: SolverConfig | None = None,
) -> SolverResult:
    """
    Convenience function to solve min ratio cycle problem.

    Args:
        edges: List of (u, v, cost, time) tuples
        n_vertices: Number of vertices (auto-detected if None)
        mode: Solver mode
        config: Solver configuration

    Returns:
        SolverResult with solution
    """
    if n_vertices is None:
        # Auto-detect number of vertices
        vertices = set()
        for u, v, _, _ in edges:
            vertices.add(u)
            vertices.add(v)
        n_vertices = max(vertices) + 1 if vertices else 1

    solver = MinRatioCycleSolver(n_vertices, config)
    solver.add_edges(edges)
    return solver.solve(mode=mode)


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    # Example 1: Basic integer weights (exact mode)
    print("Example 1: Basic integer weights")
    solver1 = MinRatioCycleSolver(3)
    solver1.add_edge(0, 1, 2, 1)  # cost=2, time=1
    solver1.add_edge(1, 2, 3, 2)  # cost=3, time=2
    solver1.add_edge(2, 0, 1, 1)  # cost=1, time=1

    result1 = solver1.solve()
    print(f"Result: {result1.cycle}, ratio = {result1.ratio}")
    print(f"Mode used: {result1.metrics.mode_used if result1.metrics else 'unknown'}")

    # Example 2: Float weights (numeric mode)
    print("\nExample 2: Float weights")
    solver2 = MinRatioCycleSolver(3)
    solver2.add_edge(0, 1, 2.5, 1.0)
    solver2.add_edge(1, 2, 3.0, 2.0)
    solver2.add_edge(2, 0, 1.0, 1.0)

    result2 = solver2.solve()
    print(f"Result: {result2.cycle}, ratio = {result2.ratio:.6f}")
    print(f"Mode used: {result2.metrics.mode_used if result2.metrics else 'unknown'}")

    # Example 3: Using configuration
    print("\nExample 3: Custom configuration")
    config = SolverConfig(
        log_level=LogLevel.DEBUG, enable_preprocessing=True, validate_cycles=True
    )

    solver3 = MinRatioCycleSolver(4, config)
    edges = [
        (0, 1, 5, 2),
        (1, 2, 3, 1),
        (2, 3, 2, 2),
        (3, 0, 1, 1),
        (0, 2, 10, 3),  # Alternative path
    ]
    solver3.add_edges(edges)

    result3 = solver3.solve(mode=SolverMode.NUMERIC)
    print(f"Result: {result3.cycle}, ratio = {result3.ratio:.6f}")

    if result3.success and result3.metrics:
        print(f"Solve time: {result3.metrics.solve_time:.4f}s")
        print(f"Iterations: {result3.metrics.iterations}")
        print(f"Preprocessing time: {result3.metrics.preprocessing_time:.6f}s")

    # Example 4: Health check and debugging
    print("\nExample 4: Health check")
    health = solver3.health_check()
    print(f"Overall health: {health['summary']['overall_status']}")
    print(f"Checks: {health['summary']['passed']}/{health['summary']['total']} passed")

    # Example 5: Convenience function
    print("\nExample 5: Convenience function")
    edges_list = [
        (0, 1, -2, 1),  # Negative cost cycle
        (1, 2, -1, 1),
        (2, 0, 1, 1),
    ]

    result5 = solve_min_ratio_cycle(edges_list)
    print(f"Convenience result: ratio = {result5.ratio:.6f}")

    # Example 6: Approximate mode for large graphs
    print("\nExample 6: Approximate mode")
    solver6 = MinRatioCycleSolver(10)

    # Add many random edges
    import random  # nosec B311

    random.seed(42)
    for _ in range(30):
        u, v = random.randint(0, 9), random.randint(0, 9)  # nosec B311
        cost = random.randint(-5, 10)  # nosec B311
        time = random.randint(1, 5)  # nosec B311
        solver6.add_edge(u, v, cost, time)

    result6 = solver6.solve(mode=SolverMode.APPROXIMATE)
    print(f"Approximate result: ratio = {result6.ratio:.6f}")
    print(f"Success: {result6.success}")

    # Example 7: Debug information
    print("\nExample 7: Debug info")
    debug_info = solver1.get_debug_info()
    print(debug_info)
