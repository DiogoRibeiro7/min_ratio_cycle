"""
Debugging utilities for min-ratio-cycle solver.

This module provides comprehensive debugging and diagnostic tools to
help identify and resolve issues with solver performance, graph
structure problems, and algorithm failures.
"""

import time
from typing import TYPE_CHECKING, Any

import numpy as np

# Import exceptions for proper error reporting
from min_ratio_cycle.exceptions import SolverError

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from min_ratio_cycle.solver import MinRatioCycleSolver


class SolverDebugger:
    """
    Comprehensive debugging utilities for solver issues.

    This class provides tools to analyze graph properties, detect
    potential issues, and generate detailed diagnostic reports to help
    troubleshoot solver problems.
    """

    def __init__(self, solver: "MinRatioCycleSolver"):
        """
        Initialize debugger for a solver instance.

        Args:
            solver: The MinRatioCycleSolver instance to debug
        """
        self.solver = solver
        self._analysis_cache: dict[str, Any] = {}
        self._cache_timestamp: float = 0.0

    def analyze_graph_properties(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Analyze comprehensive graph properties for debugging.

        Args:
            use_cache: Whether to use cached analysis results

        Returns:
            Dictionary with detailed graph analysis
        """
        # Check cache validity
        current_time = time.time()
        if (
            use_cache
            and self._analysis_cache
            and current_time - self._cache_timestamp < 60.0
        ):  # Cache for 1 minute
            return self._analysis_cache.copy()

        try:
            # Ensure arrays are built
            self.solver._build_arrays_if_needed()

            if self.solver._U is None or len(self.solver._edges) == 0:
                return {"error": "No edges in graph"}

            n = self.solver.n
            m = len(self.solver._U)

            # Basic graph metrics
            density = m / (n * n) if n > 0 else 0.0

            # Connectivity analysis
            in_degree = np.bincount(self.solver._V, minlength=n)
            out_degree = np.bincount(self.solver._U, minlength=n)

            # Weight analysis
            costs = self.solver._C
            times = self.solver._T

            # Strongly connected components analysis (simplified)
            scc_info = self._analyze_connectivity()

            # Cycle detection hints
            cycle_hints = self._analyze_cycle_potential()

            analysis = {
                "basic_properties": {
                    "vertices": n,
                    "edges": m,
                    "density": density,
                    "is_sparse": density < 0.1,
                    "is_dense": density > 0.5,
                    "complete_graph": m == n * (n - 1),  # Directed complete graph
                },
                "connectivity": {
                    "isolated_vertices": int(
                        np.sum((in_degree == 0) & (out_degree == 0))
                    ),
                    "source_vertices": int(np.sum((in_degree == 0) & (out_degree > 0))),
                    "sink_vertices": int(np.sum((in_degree > 0) & (out_degree == 0))),
                    "avg_in_degree": float(np.mean(in_degree)),
                    "avg_out_degree": float(np.mean(out_degree)),
                    "max_in_degree": int(np.max(in_degree)),
                    "max_out_degree": int(np.max(out_degree)),
                    "min_in_degree": int(np.min(in_degree)),
                    "min_out_degree": int(np.min(out_degree)),
                },
                "weights": {
                    "cost_range": (float(np.min(costs)), float(np.max(costs))),
                    "time_range": (float(np.min(times)), float(np.max(times))),
                    "cost_mean": float(np.mean(costs)),
                    "time_mean": float(np.mean(times)),
                    "cost_std": float(np.std(costs)),
                    "time_std": float(np.std(times)),
                    "integer_weights": self.solver._all_int,
                    "negative_costs": int(np.sum(costs < 0)),
                    "zero_costs": int(np.sum(costs == 0)),
                    "very_small_times": int(np.sum(times < 1e-6)),
                },
                "numerical_properties": {
                    "cost_range_magnitude": float(np.max(costs) - np.min(costs)),
                    "time_ratio_magnitude": (
                        float(np.max(times) / np.min(times))
                        if np.min(times) > 0
                        else float("inf")
                    ),
                    "potential_precision_issues": self._check_numerical_precision(
                        costs, times
                    ),
                    "condition_number_estimate": self._estimate_condition_number(
                        costs, times
                    ),
                },
                "structure_analysis": scc_info,
                "cycle_analysis": cycle_hints,
                "parallel_edges": self._count_parallel_edges(),
                "self_loops": self._count_self_loops(),
            }

            # Cache the results
            self._analysis_cache = analysis
            self._cache_timestamp = current_time

            return analysis

        except Exception as e:
            return {
                "error": f"Failed to analyze graph properties: {str(e)}",
                "exception_type": type(e).__name__,
            }

    def detect_potential_issues(self) -> list[dict[str, str]]:
        """
        Detect potential issues that might cause solve failures.

        Returns:
            List of issue dictionaries with type, message, and severity
        """
        issues = []
        props = self.analyze_graph_properties()

        if "error" in props:
            issues.append(
                {
                    "type": "CRITICAL",
                    "severity": "HIGH",
                    "message": props["error"],
                    "category": "graph_structure",
                }
            )
            return issues

        # Check basic graph structure
        basic = props["basic_properties"]
        connectivity = props["connectivity"]
        weights = props["weights"]
        numerical = props["numerical_properties"]

        # Critical issues
        if connectivity["isolated_vertices"] > 0:
            issues.append(
                {
                    "type": "WARNING",
                    "severity": "MEDIUM",
                    "message": f"{connectivity['isolated_vertices']} isolated vertices (no cycles possible through them)",
                    "category": "connectivity",
                    "suggestion": "Remove isolated vertices or add connecting edges",
                }
            )

        if basic["edges"] == 0:
            issues.append(
                {
                    "type": "CRITICAL",
                    "severity": "HIGH",
                    "message": "Graph has no edges",
                    "category": "graph_structure",
                    "suggestion": "Add edges to create a meaningful graph",
                }
            )

        # Performance issues
        if basic["density"] < 0.005:
            issues.append(
                {
                    "type": "INFO",
                    "severity": "LOW",
                    "message": f"Very sparse graph (density: {basic['density']:.6f})",
                    "category": "performance",
                    "suggestion": "Consider using sparse graph optimizations",
                }
            )

        # Numerical issues
        if numerical["potential_precision_issues"]:
            issues.append(
                {
                    "type": "WARNING",
                    "severity": "HIGH",
                    "message": "Potential floating-point precision issues detected",
                    "category": "numerical",
                    "suggestion": "Consider using exact mode or scaling weights",
                }
            )

        if weights["cost_range"][1] - weights["cost_range"][0] > 1e12:
            issues.append(
                {
                    "type": "WARNING",
                    "severity": "MEDIUM",
                    "message": f"Very large cost range: {weights['cost_range']}",
                    "category": "numerical",
                    "suggestion": "Consider normalizing or scaling cost values",
                }
            )

        if numerical["time_ratio_magnitude"] > 1e12:
            issues.append(
                {
                    "type": "WARNING",
                    "severity": "MEDIUM",
                    "message": f"Very large time ratio: {numerical['time_ratio_magnitude']:.2e}",
                    "category": "numerical",
                    "suggestion": "Consider normalizing or scaling time values",
                }
            )

        if weights["very_small_times"] > 0:
            issues.append(
                {
                    "type": "WARNING",
                    "severity": "MEDIUM",
                    "message": f"{weights['very_small_times']} edges have very small time values (< 1e-6)",
                    "category": "numerical",
                    "suggestion": "Very small time values may cause numerical instability",
                }
            )

        # Structural issues
        if connectivity["source_vertices"] == basic["vertices"]:
            issues.append(
                {
                    "type": "CRITICAL",
                    "severity": "HIGH",
                    "message": "All vertices are sources (no incoming edges) - no cycles possible",
                    "category": "graph_structure",
                    "suggestion": "Add edges to create cycles",
                }
            )

        if connectivity["sink_vertices"] == basic["vertices"]:
            issues.append(
                {
                    "type": "CRITICAL",
                    "severity": "HIGH",
                    "message": "All vertices are sinks (no outgoing edges) - no cycles possible",
                    "category": "graph_structure",
                    "suggestion": "Add edges to create cycles",
                }
            )

        # Algorithm-specific issues
        if (
            not weights["integer_weights"]
            and numerical["condition_number_estimate"] > 1e12
        ):
            issues.append(
                {
                    "type": "WARNING",
                    "severity": "MEDIUM",
                    "message": "High condition number detected - numerical solver may be unstable",
                    "category": "algorithm",
                    "suggestion": "Consider using exact mode or better-conditioned weights",
                }
            )

        return issues

    def generate_debug_report(self, include_detailed_analysis: bool = True) -> str:
        """
        Generate comprehensive debug report.

        Args:
            include_detailed_analysis: Whether to include detailed numerical analysis

        Returns:
            Formatted debug report string
        """
        props = self.analyze_graph_properties()
        issues = self.detect_potential_issues()

        report_lines = [
            "Min Ratio Cycle Solver - Debug Report",
            "=" * 50,
            "",
            f"Analysis timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Graph Properties Section
        report_lines.append("Graph Properties:")
        if "error" not in props:
            basic = props["basic_properties"]
            connectivity = props["connectivity"]
            weights = props["weights"]

            report_lines.extend(
                [
                    f"  Vertices: {basic['vertices']}",
                    f"  Edges: {basic['edges']}",
                    f"  Density: {basic['density']:.6f}",
                    f"  Graph type: {'Complete' if basic['complete_graph'] else 'Sparse' if basic['is_sparse'] else 'Dense' if basic['is_dense'] else 'Medium'}",
                    f"  Integer weights: {weights['integer_weights']}",
                    "",
                    "Connectivity Analysis:",
                    f"  Isolated vertices: {connectivity['isolated_vertices']}",
                    f"  Source vertices: {connectivity['source_vertices']}",
                    f"  Sink vertices: {connectivity['sink_vertices']}",
                    f"  Average degree: in={connectivity['avg_in_degree']:.2f}, out={connectivity['avg_out_degree']:.2f}",
                    "",
                    "Weight Analysis:",
                    f"  Cost range: [{weights['cost_range'][0]:.3f}, {weights['cost_range'][1]:.3f}]",
                    f"  Time range: [{weights['time_range'][0]:.3f}, {weights['time_range'][1]:.3f}]",
                    f"  Cost mean±std: {weights['cost_mean']:.3f}±{weights['cost_std']:.3f}",
                    f"  Time mean±std: {weights['time_mean']:.3f}±{weights['time_std']:.3f}",
                    f"  Negative costs: {weights['negative_costs']}",
                    f"  Zero costs: {weights['zero_costs']}",
                ]
            )

            if include_detailed_analysis:
                numerical = props["numerical_properties"]
                structure = props["structure_analysis"]

                report_lines.extend(
                    [
                        "",
                        "Numerical Analysis:",
                        f"  Cost range magnitude: {numerical['cost_range_magnitude']:.2e}",
                        f"  Time ratio magnitude: {numerical['time_ratio_magnitude']:.2e}",
                        f"  Condition number estimate: {numerical['condition_number_estimate']:.2e}",
                        f"  Precision issues: {numerical['potential_precision_issues']}",
                        "",
                        "Structure Analysis:",
                        f"  Weakly connected: {structure.get('weakly_connected', 'Unknown')}",
                        f"  Parallel edges: {props['parallel_edges']}",
                        f"  Self loops: {props['self_loops']}",
                    ]
                )
        else:
            report_lines.append(f"  Error: {props['error']}")

        # Issues Section
        report_lines.extend(["", "Potential Issues:"])

        if not issues:
            report_lines.append("  None detected")
        else:
            # Group issues by severity
            critical_issues = [i for i in issues if i.get("severity") == "HIGH"]
            warning_issues = [i for i in issues if i.get("severity") == "MEDIUM"]
            info_issues = [i for i in issues if i.get("severity") == "LOW"]

            for severity, issue_list, label in [
                ("HIGH", critical_issues, "Critical Issues"),
                ("MEDIUM", warning_issues, "Warnings"),
                ("LOW", info_issues, "Informational"),
            ]:
                if issue_list:
                    report_lines.append(f"  {label}:")
                    for issue in issue_list:
                        report_lines.append(f"    [{issue['type']}] {issue['message']}")
                        if "suggestion" in issue:
                            report_lines.append(
                                f"      → Suggestion: {issue['suggestion']}"
                            )

        # Recommendations Section
        report_lines.extend(["", "Recommendations:"])

        recommendations = self._generate_recommendations(props, issues)
        if recommendations:
            for rec in recommendations:
                report_lines.append(f"  • {rec}")
        else:
            report_lines.append("  No specific recommendations at this time")

        return "\n".join(report_lines)

    def _analyze_connectivity(self) -> dict[str, Any]:
        """
        Analyze graph connectivity properties.
        """
        try:
            # Simple weakly connected check
            n = self.solver.n

            # Build undirected adjacency for weak connectivity
            visited = np.zeros(n, dtype=bool)

            def dfs(v: int) -> int:
                """
                DFS to count reachable vertices.
                """
                visited[v] = True
                count = 1

                # Check outgoing edges
                if hasattr(self.solver, "_starts") and self.solver._starts is not None:
                    start = self.solver._starts[v]
                    end = start + self.solver._counts[v]
                    for i in range(start, end):
                        neighbor = self.solver._V[i]
                        if not visited[neighbor]:
                            count += dfs(neighbor)

                # Check incoming edges (reverse direction)
                for i in range(len(self.solver._U)):
                    if self.solver._V[i] == v and not visited[self.solver._U[i]]:
                        count += dfs(self.solver._U[i])

                return count

            # Start DFS from vertex 0 if it has edges
            reachable_count = 0
            if len(self.solver._edges) > 0:
                reachable_count = dfs(0)

            return {
                "weakly_connected": reachable_count == n,
                "reachable_vertices": reachable_count,
                "unreachable_vertices": n - reachable_count,
            }

        except Exception:
            return {"weakly_connected": "Unknown", "analysis_error": True}

    def _analyze_cycle_potential(self) -> dict[str, Any]:
        """
        Analyze potential for cycles in the graph.
        """
        try:
            # Simple heuristics for cycle potential
            n = self.solver.n
            m = len(self.solver._edges)

            # Minimum edges needed for a cycle
            min_edges_for_cycle = n
            has_enough_edges = m >= min_edges_for_cycle

            # Check for obvious cycle structures
            self_loops = self._count_self_loops()
            has_self_loops = self_loops > 0

            return {
                "minimum_cycle_possible": has_enough_edges or has_self_loops,
                "edges_needed_for_cycle": (
                    max(0, min_edges_for_cycle - m) if not has_self_loops else 0
                ),
                "self_loops_present": has_self_loops,
                "cycle_likelihood": "High" if has_enough_edges else "Low",
            }

        except Exception:
            return {"analysis_error": True}

    def _count_parallel_edges(self) -> int:
        """
        Count parallel edges (multiple edges between same vertex pairs).
        """
        try:
            edge_counts = {}
            for edge in self.solver._edges:
                key = (edge.u, edge.v)
                edge_counts[key] = edge_counts.get(key, 0) + 1

            return sum(1 for count in edge_counts.values() if count > 1)
        except Exception:
            return 0

    def _count_self_loops(self) -> int:
        """
        Count self-loops (edges from vertex to itself).
        """
        try:
            return sum(1 for edge in self.solver._edges if edge.u == edge.v)
        except Exception:
            return 0

    def _check_numerical_precision(self, costs: np.ndarray, times: np.ndarray) -> bool:
        """
        Check for potential numerical precision issues.
        """
        try:
            # Check for very large or very small values
            cost_range = np.max(costs) - np.min(costs)
            time_range = (
                np.max(times) / np.min(times) if np.min(times) > 0 else float("inf")
            )

            # Check for precision issues
            has_large_range = cost_range > 1e12 or time_range > 1e12
            has_small_values = np.any(np.abs(costs) < 1e-12) or np.any(times < 1e-12)
            has_large_values = np.any(np.abs(costs) > 1e12) or np.any(times > 1e12)

            return has_large_range or has_small_values or has_large_values

        except Exception:
            return False

    def _estimate_condition_number(self, costs: np.ndarray, times: np.ndarray) -> float:
        """
        Estimate condition number for numerical stability assessment.
        """
        try:
            # Simple heuristic based on weight ranges
            if len(costs) == 0 or len(times) == 0:
                return 1.0

            cost_ratio = np.max(np.abs(costs)) / (
                np.min(np.abs(costs[costs != 0])) if np.any(costs != 0) else 1.0
            )
            time_ratio = np.max(times) / np.min(times) if np.min(times) > 0 else 1.0

            return max(cost_ratio, time_ratio)

        except Exception:
            return 1.0

    def _generate_recommendations(
        self, props: dict[str, Any], issues: list[dict[str, str]]
    ) -> list[str]:
        """
        Generate actionable recommendations based on analysis.
        """
        recommendations = []

        if "error" in props:
            return ["Fix graph structure errors before proceeding"]

        # Critical issues first
        critical_issues = [i for i in issues if i.get("severity") == "HIGH"]
        if critical_issues:
            recommendations.append(
                "Address critical issues first - solver may fail without fixes"
            )

        # Performance recommendations
        basic = props.get("basic_properties", {})
        if basic.get("is_sparse", False):
            recommendations.append("Consider enabling sparse graph optimizations")

        if basic.get("is_dense", False):
            recommendations.append(
                "Dense graph detected - consider parallel processing for large graphs"
            )

        # Numerical recommendations
        weights = props.get("weights", {})
        numerical = props.get("numerical_properties", {})

        if weights.get("integer_weights", False):
            recommendations.append(
                "Integer weights detected - exact mode will give precise results"
            )
        elif numerical.get("potential_precision_issues", False):
            recommendations.append(
                "Consider scaling weights or using exact mode for better precision"
            )

        # Algorithm recommendations
        if len(issues) == 0:
            recommendations.append(
                "Graph structure looks good - solver should work well"
            )

        return recommendations

    def clear_cache(self) -> None:
        """
        Clear analysis cache to force fresh analysis.
        """
        self._analysis_cache.clear()
        self._cache_timestamp = 0.0

    def get_solver_state_summary(self) -> dict[str, Any]:
        """
        Get a quick summary of solver state for debugging.
        """
        return {
            "n_vertices": self.solver.n,
            "n_edges": len(self.solver._edges),
            "arrays_built": (
                self.solver._arrays_built
                if hasattr(self.solver, "_arrays_built")
                else False
            ),
            "integer_mode": self.solver._all_int,
            "config_present": hasattr(self.solver, "config")
            and self.solver.config is not None,
            "last_result_available": hasattr(self.solver, "_last_result")
            and self.solver._last_result is not None,
        }


# Convenience functions for quick debugging
def quick_debug(solver: "MinRatioCycleSolver") -> str:
    """
    Quick debug report for a solver instance.

    Args:
        solver: MinRatioCycleSolver instance

    Returns:
        Brief debug report string
    """
    debugger = SolverDebugger(solver)
    issues = debugger.detect_potential_issues()

    critical = len([i for i in issues if i.get("severity") == "HIGH"])
    warnings = len([i for i in issues if i.get("severity") == "MEDIUM"])

    summary = debugger.get_solver_state_summary()

    return (
        f"Quick Debug: {summary['n_vertices']} vertices, {summary['n_edges']} edges, "
        f"{critical} critical issues, {warnings} warnings"
    )


def diagnose_solve_failure(
    solver: "MinRatioCycleSolver", exception: Exception | None = None
) -> str:
    """
    Diagnose why a solve operation failed.

    Args:
        solver: The solver instance that failed
        exception: Optional exception that was raised

    Returns:
        Diagnostic report string
    """
    debugger = SolverDebugger(solver)
    report = debugger.generate_debug_report(include_detailed_analysis=True)

    if exception:
        report += "\n\nException Details:\n"
        report += f"Exception type: {type(exception).__name__}\n"
        report += f"Exception message: {str(exception)}\n"

        if isinstance(exception, SolverError) and hasattr(exception, "details"):
            report += f"Error details: {exception.details}\n"

    return report


# Module exports
__all__ = [
    "SolverDebugger",
    "quick_debug",
    "diagnose_solve_failure",
]
