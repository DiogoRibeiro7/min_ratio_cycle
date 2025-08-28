"""
Validation helpers for graphs and cycles.

This module provides reusable validation utilities for the min ratio cycle
solver.  The functions raise :class:`ValidationError` or
:class:`CycleValidationError` on malformed input so callers can surface helpful
error messages.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Union

import networkx as nx

from min_ratio_cycle.exceptions import (
    CycleValidationError,
    GraphStructureError,
    NumericalInstabilityError,
    ValidationError,
)

Weight = Union[int, float, Fraction]


def _is_number(value: object) -> bool:
    """
    Return ``True`` if ``value`` is an accepted numeric type.
    """
    return isinstance(value, (int, float, Fraction))


def validate_graph(
    n_vertices: int, edges: Iterable[tuple[int, int, Weight, Weight]]
) -> None:
    """
    Validate basic structural properties of a directed graph.

    Parameters
    ----------
    n_vertices:
        Total number of vertices in the graph.  Must be a positive integer.
    edges:
        Iterable of ``(u, v, cost, time)`` tuples describing directed edges.

    Raises
    ------
    ValidationError
        If any structural constraint is violated.
    """

    if not isinstance(n_vertices, int) or n_vertices <= 0:
        raise ValidationError(
            "Number of vertices must be a positive integer",
            invalid_value=n_vertices,
            expected_type=int,
            valid_range=(1, None),
        )

    for idx, (u, v, cost, time) in enumerate(edges):
        if not (isinstance(u, int) and isinstance(v, int)):
            raise ValidationError(
                f"Edge {idx} has non-integer vertices",
                invalid_value=(u, v),
                expected_type=int,
            )
        if not (0 <= u < n_vertices and 0 <= v < n_vertices):
            raise ValidationError(
                f"Edge {idx} has vertices outside [0, {n_vertices - 1}]",
                invalid_value=(u, v),
                expected_type=int,
                valid_range=(0, n_vertices - 1),
            )
        if not _is_number(cost) or not _is_number(time):
            raise ValidationError(
                f"Edge {idx} has non-numeric weights",
                invalid_value=(cost, time),
            )
        time_val = float(time) if not isinstance(time, Fraction) else time
        if time_val <= 0:
            raise ValidationError(
                f"Edge {idx} has non-positive transit time",
                invalid_value=time,
                valid_range=(0, None),
            )


def validate_cycle(
    cycle: Sequence[int],
    edge_lookup: dict[tuple[int, int], tuple[Weight, Weight]],
) -> tuple[float, float, float]:
    """
    Validate that ``cycle`` is closed and edges exist in ``edge_lookup``.

    The ``edge_lookup`` mapping should map ``(u, v)`` pairs to ``(cost, time)``
    weights.  On success the function returns the total cost, time, and ratio of
    the cycle.

    Raises
    ------
    CycleValidationError
        If the cycle is malformed or references missing edges.
    """

    if len(cycle) < 2:
        raise CycleValidationError("Cycle must contain at least two vertices", cycle)
    if cycle[0] != cycle[-1]:
        raise CycleValidationError(
            "Cycle must be closed (first vertex equals last)", cycle
        )

    total_cost = 0.0
    total_time = 0.0
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        if (u, v) not in edge_lookup:
            raise CycleValidationError(f"Missing edge {u} -> {v}", (u, v))
        cost, time = edge_lookup[(u, v)]
        if not _is_number(cost) or not _is_number(time):
            raise CycleValidationError("Edge weights must be numeric", (cost, time))
        total_cost += float(cost)
        total_time += float(time)

    if total_time == 0:
        raise CycleValidationError("Cycle has zero total transit time", cycle)

    return total_cost, total_time, total_cost / total_time


@dataclass
class ValidationHelper:
    """
    Namespace wrapper around graph and cycle validation functions.
    """

    @staticmethod
    def validate_graph(
        n_vertices: int, edges: Iterable[tuple[int, int, Weight, Weight]]
    ) -> None:
        validate_graph(n_vertices, edges)

    @staticmethod
    def validate_cycle(
        cycle: Sequence[int],
        edge_lookup: dict[tuple[int, int], tuple[Weight, Weight]],
    ) -> tuple[float, float, float]:
        return validate_cycle(cycle, edge_lookup)

    @staticmethod
    def generate_graph_report(
        n_vertices: int, edges: Iterable[tuple[int, int, Weight, Weight]]
    ) -> ValidationReport:
        return generate_validation_report(n_vertices, edges)


@dataclass
class ValidationIssue:
    message: str
    context: dict[str, Any] | None = None


@dataclass
class ValidationReport:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)


def generate_validation_report(
    n_vertices: int, edges: Iterable[tuple[int, int, Weight, Weight]]
) -> ValidationReport:
    """
    Return a detailed validation report instead of raising errors.
    """

    issues: list[ValidationIssue] = []

    def add(message: str, **ctx: Any) -> None:
        issues.append(ValidationIssue(message, ctx or None))

    if not isinstance(n_vertices, int) or n_vertices <= 0:
        add(
            "Number of vertices must be a positive integer",
            invalid_value=n_vertices,
        )

    for idx, (u, v, cost, time) in enumerate(edges):
        if not (isinstance(u, int) and isinstance(v, int)):
            add("Edge has non-integer vertices", index=idx, value=(u, v))
        if not (0 <= u < n_vertices and 0 <= v < n_vertices):
            add(
                "Edge vertices outside valid range",
                index=idx,
                value=(u, v),
                valid_range=(0, n_vertices - 1),
            )
        if not _is_number(cost) or not _is_number(time):
            add("Edge has non-numeric weights", index=idx, value=(cost, time))
        time_val = float(time) if not isinstance(time, Fraction) else time
        if time_val <= 0:
            add("Edge has non-positive transit time", index=idx, value=time)

    return ValidationReport(is_valid=len(issues) == 0, issues=issues)


@dataclass
class TopologyInfo:
    """
    Simple graph topology summary.
    """

    is_connected: bool
    is_dag: bool
    n_components: int


def analyze_graph_topology(
    n_vertices: int, edges: Iterable[tuple[int, int, Weight, Weight]]
) -> TopologyInfo:
    """
    Return connectivity and acyclicity information for a graph.
    """

    G = nx.DiGraph()
    G.add_nodes_from(range(n_vertices))
    for u, v, *_ in edges:
        G.add_edge(u, v)
    is_connected = nx.is_weakly_connected(G) if n_vertices > 0 else True
    components = list(nx.weakly_connected_components(G))
    return TopologyInfo(
        is_connected=is_connected,
        is_dag=nx.is_directed_acyclic_graph(G),
        n_components=len(components),
    )


def _condition_number(values: Sequence[float]) -> float:
    """
    Return crude condition number ``max/ min`` for ``values``.
    """

    positive = [abs(v) for v in values if v != 0]
    if not positive:
        return math.inf
    max_val = max(positive)
    min_val = min(x for x in positive if x > 0)
    return max_val / min_val if min_val > 0 else math.inf


def pre_solve_validate(
    n_vertices: int,
    edges: Iterable[tuple[int, int, Weight, Weight]],
    *,
    weight_limit: float = 1e9,
    cond_threshold: float = 1e12,
) -> None:
    """
    Run extended checks before attempting to solve the graph.
    """

    edge_list = list(edges)
    validate_graph(n_vertices, edge_list)

    topo = analyze_graph_topology(n_vertices, edge_list)
    if not topo.is_connected:
        raise GraphStructureError(
            "Graph has disconnected components",
            graph_properties={"components": topo.n_components},
            suggested_fix="Ensure all vertices are reachable",
        )
    if topo.is_dag:
        raise GraphStructureError(
            "Graph is acyclic (DAG) and contains no cycles",
            graph_properties={"components": topo.n_components},
            suggested_fix="Add edges to form a cycle",
        )

    weights = [float(c) for _, _, c, _ in edge_list] + [
        float(t) for _, _, _, t in edge_list
    ]
    if any(math.isinf(w) or math.isnan(w) for w in weights):
        raise NumericalInstabilityError(
            "Edge weights must be finite",
            component="validation",
            suggested_fix="Remove inf/nan weights",
        )
    max_abs = max(abs(w) for w in weights) if weights else 0.0
    if max_abs > weight_limit:
        raise ValidationError(
            "Edge weight exceeds allowed range",
            invalid_value=max_abs,
            valid_range=(-weight_limit, weight_limit),
        )
    cond = _condition_number(weights)
    if cond > cond_threshold:
        raise NumericalInstabilityError(
            "Poor numerical conditioning",
            computation_details={"condition_number": cond},
            suggested_fix="Scale weights closer together",
        )
    if cond > cond_threshold / 100:
        warnings.warn(
            f"High condition number {cond:.2e} may lead to precision loss",
            RuntimeWarning,
        )


def post_solve_validate(
    cycle: Sequence[int],
    ratio: float,
    edge_lookup: dict[tuple[int, int], tuple[Weight, Weight]],
    *,
    tol: float = 1e-9,
) -> tuple[float, float, float]:
    """
    Verify solved cycle and ratio consistency.
    """

    cost, time, computed_ratio = validate_cycle(cycle, edge_lookup)
    if not math.isclose(computed_ratio, ratio, rel_tol=tol, abs_tol=tol):
        raise CycleValidationError(
            "Reported ratio inconsistent with edge weights",
            cycle=list(cycle),
            weight_mismatch={
                "reported_ratio": ratio,
                "computed_ratio": computed_ratio,
            },
        )
    return cost, time, computed_ratio
