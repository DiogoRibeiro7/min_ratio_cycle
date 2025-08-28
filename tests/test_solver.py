"""
Comprehensive test suite for MinRatioCycleSolver.

Features:
- Edge case testing (no cycles, self-loops, parallel edges)
- Property-based testing with hypothesis
- Benchmark comparisons with naive implementations
- Correctness validation of returned cycles
"""

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from min_ratio_cycle.exceptions import (
    NumericalInstabilityError,
    ResourceExhaustionError,
)
from min_ratio_cycle.solver import MinRatioCycleSolver, SolverConfig


class TestEdgeCases:
    """
    Test edge cases and degenerate scenarios.
    """

    def test_empty_graph(self):
        """
        Graph with no edges should raise appropriate error.
        """
        solver = MinRatioCycleSolver(3)
        with pytest.raises(ValueError, match="Graph has no edges"):
            solver.solve()

    def test_single_vertex(self):
        """
        Single vertex with no edges.
        """
        solver = MinRatioCycleSolver(1)
        with pytest.raises(ValueError, match="Graph has no edges"):
            solver.solve()

    def test_zero_vertices_invalid(self):
        """
        Zero vertices should be rejected at construction.
        """
        with pytest.raises(ValueError, match="n_vertices must be a positive integer"):
            MinRatioCycleSolver(0)

    def test_no_cycles_tree(self):
        """
        Tree structure with no cycles should fail gracefully.
        """
        solver = MinRatioCycleSolver(4)
        # Create a tree: 0->1, 1->2, 1->3
        solver.add_edge(0, 1, 5, 2)
        solver.add_edge(1, 2, 3, 1)
        solver.add_edge(1, 3, 4, 2)

        from min_ratio_cycle.exceptions import NumericalInstabilityError

        with pytest.raises(NumericalInstabilityError):
            solver.solve()

    def test_self_loop(self):
        """
        Single vertex with self-loop.
        """
        solver = MinRatioCycleSolver(1)
        solver.add_edge(0, 0, 3, 2)

        cycle, sum_cost, sum_time, ratio = solver.solve()
        assert cycle == [0, 0]  # closed form
        assert sum_cost == 3
        assert sum_time == 2
        assert abs(ratio - 1.5) < 1e-10

    def test_parallel_edges(self):
        """
        Multiple edges between same vertices.
        """
        solver = MinRatioCycleSolver(2)
        # Add multiple 0->1 edges with different ratios
        solver.add_edge(0, 1, 10, 2)  # ratio 5
        solver.add_edge(0, 1, 6, 3)  # ratio 2
        solver.add_edge(1, 0, 1, 1)  # ratio 1

        cycle, sum_cost, sum_time, ratio = solver.solve()
        # Should use the better 0->1 edge (ratio 2)
        assert len(cycle) == 3  # includes closing vertex
        assert cycle[0] == cycle[-1]  # proper cycle
        assert abs(ratio - 1.75) < 1e-10  # (6+1)/(3+1) = 1.75

    def test_disconnected_components(self):
        """
        Graph with multiple disconnected components.
        """
        solver = MinRatioCycleSolver(6)
        # Component 1: cycle 0->1->2->0
        solver.add_edge(0, 1, 2, 1)
        solver.add_edge(1, 2, 3, 2)
        solver.add_edge(2, 0, 1, 1)
        # Component 2: cycle 3->4->3 (better ratio)
        solver.add_edge(3, 4, 1, 2)
        solver.add_edge(4, 3, 1, 2)
        # Isolated vertices: 5

        cycle, sum_cost, sum_time, ratio = solver.solve()
        # Should find the better cycle (component 2)
        assert abs(ratio - 0.5) < 1e-10  # (1+1)/(2+2) = 0.5

    def test_zero_time_edge_rejected(self):
        """
        Edges with zero or negative time should be rejected.
        """
        solver = MinRatioCycleSolver(2)

        with pytest.raises(ValueError, match="time must be strictly positive"):
            solver.add_edge(0, 1, 1, 0)

        with pytest.raises(ValueError, match="time must be strictly positive"):
            solver.add_edge(0, 1, 1, -1)

    def test_invalid_vertex_indices(self):
        """
        Invalid vertex indices should be rejected.
        """
        solver = MinRatioCycleSolver(3)

        with pytest.raises(ValueError, match="valid vertex indices"):
            solver.add_edge(-1, 0, 1, 1)

        with pytest.raises(ValueError, match="valid vertex indices"):
            solver.add_edge(0, 3, 1, 1)

    def test_large_weights(self):
        """
        Test with very large integer weights.
        """
        solver = MinRatioCycleSolver(3)
        large_val = 10**15
        solver.add_edge(0, 1, large_val, 1)
        solver.add_edge(1, 2, 1, large_val)
        solver.add_edge(2, 0, 1, 1)

        from min_ratio_cycle.exceptions import NumericalInstabilityError

        with pytest.raises(NumericalInstabilityError):
            solver.solve()


class TestCorrectnessValidation:
    """
    Validate that returned cycles are actually correct.
    """

    def validate_cycle(
        self,
        solver: MinRatioCycleSolver,
        cycle: list[int],
        expected_cost: float,
        expected_time: float,
        expected_ratio: float,
    ):
        """
        Helper to validate a cycle's properties.
        """
        # Check cycle is closed
        assert cycle[0] == cycle[-1], "Cycle should be closed (first == last)"

        # Check cycle length
        assert (
            len(cycle) >= 3
        ), "Cycle should have at least 3 vertices (including closing)"

        # Manually compute cost and time by walking the cycle
        actual_cost = 0.0
        actual_time = 0.0

        # Build edge lookup from solver's internal state
        solver._build_numpy_arrays_once()
        edge_map = {}
        if solver._U is not None:
            for i in range(len(solver._U)):
                u, v = int(solver._U[i]), int(solver._V[i])
                c, t = float(solver._C[i]), float(solver._T[i])
                if (u, v) not in edge_map:
                    edge_map[(u, v)] = []
                edge_map[(u, v)].append((c, t))

        # Walk the cycle (excluding the repeated last vertex)
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            if (u, v) not in edge_map:
                pytest.fail(f"Edge {u} -> {v} in cycle not found in graph")

            # For parallel edges, we need to pick one
            # The solver should have used a consistent choice
            costs_times = edge_map[(u, v)]
            if len(costs_times) == 1:
                c, t = costs_times[0]
            else:
                # For simplicity, use the first one
                # In practice, the solver's choice might be deterministic
                c, t = costs_times[0]

            actual_cost += c
            actual_time += t

        # Validate computed values match expected
        assert (
            abs(actual_cost - expected_cost) < 1e-10
        ), f"Cost mismatch: expected {expected_cost}, got {actual_cost}"
        assert (
            abs(actual_time - expected_time) < 1e-10
        ), f"Time mismatch: expected {expected_time}, got {actual_time}"

        actual_ratio = actual_cost / actual_time
        assert (
            abs(actual_ratio - expected_ratio) < 1e-10
        ), f"Ratio mismatch: expected {expected_ratio}, got {actual_ratio}"

    def test_simple_triangle_validation(self):
        """
        Validate a simple triangle cycle.
        """
        solver = MinRatioCycleSolver(3)
        solver.add_edge(0, 1, 2, 1)  # ratio 2
        solver.add_edge(1, 2, 3, 2)  # ratio 1.5
        solver.add_edge(2, 0, 1, 1)  # ratio 1

        cycle, sum_cost, sum_time, ratio = solver.solve()
        self.validate_cycle(solver, cycle, sum_cost, sum_time, ratio)

    def test_mixed_integer_float_validation(self):
        """
        Validate with mixed integer/float weights.
        """
        solver = MinRatioCycleSolver(3)
        solver.add_edge(0, 1, 2.5, 1.0)
        solver.add_edge(1, 2, 3, 2.0)
        solver.add_edge(2, 0, 1.5, 1.0)

        cycle, sum_cost, sum_time, ratio = solver.solve()
        self.validate_cycle(solver, cycle, sum_cost, sum_time, ratio)


class TestResultPackaging:
    """
    Ensure solver results are easy to consume.
    """

    def test_iter_unpack(self, simple_triangle):
        solver, _ = simple_triangle
        result = solver.solve()
        cycle, cost, time, ratio = result
        assert cycle == result.cycle
        assert cost == result.sum_cost
        assert time == result.sum_time
        assert ratio == result.ratio

    def test_build_numpy_arrays_once(self):
        solver = MinRatioCycleSolver(2)
        solver.add_edge(0, 1, 1, 1)
        solver.add_edge(1, 0, 1, 1)

        calls = 0
        original = solver._build_arrays_if_needed

        def wrapper():
            nonlocal calls
            if not solver._arrays_built:
                calls += 1
            original()

        solver._build_arrays_if_needed = wrapper

        solver._build_numpy_arrays_once()
        solver._build_numpy_arrays_once()

        assert calls == 1


@composite
def random_graph(draw, max_vertices=10, max_edges=20, integer_weights=True):
    """
    Generate random graphs for property-based testing.
    """
    n = draw(st.integers(min_value=2, max_value=max_vertices))
    m = draw(
        st.integers(min_value=n, max_value=max_edges)
    )  # Ensure connectivity possibility

    solver = MinRatioCycleSolver(n)

    # Generate m random edges
    edges_added = set()
    for _ in range(m):
        u = draw(st.integers(min_value=0, max_value=n - 1))
        v = draw(st.integers(min_value=0, max_value=n - 1))

        # Allow self-loops and parallel edges
        if integer_weights:
            cost = draw(st.integers(min_value=-100, max_value=100))
            time = draw(st.integers(min_value=1, max_value=20))
        else:
            cost = draw(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            time = draw(
                st.floats(
                    min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False
                )
            )

        solver.add_edge(u, v, cost, time)
        edges_added.add((u, v, cost, time))

    return solver, list(edges_added)


class TestPropertyBased:
    """
    Property-based testing with hypothesis.
    """

    @given(random_graph(max_vertices=6, max_edges=15, integer_weights=True))
    @settings(max_examples=50, deadline=5000)
    def test_integer_mode_consistency(self, graph_data):
        """
        Test that integer mode produces valid results.
        """
        solver, edges = graph_data

        try:
            cycle, sum_cost, sum_time, ratio = solver.solve()

            # Properties that should always hold:
            # 1. Cycle is closed
            assert cycle[0] == cycle[-1]

            # 2. Ratio matches cost/time
            assert abs(ratio - sum_cost / sum_time) < 1e-10

            # 3. Time is positive
            assert sum_time > 0

            # 4. Cycle has reasonable length
            assert 2 <= len(cycle) <= len(edges) + 1

        except (RuntimeError, ValueError, NumericalInstabilityError):
            # Some graphs may have no cycles or other issues
            pass

    @given(random_graph(max_vertices=6, max_edges=15, integer_weights=False))
    @settings(max_examples=50, deadline=5000)
    def test_float_mode_consistency(self, graph_data):
        """
        Test that float mode produces valid results.
        """
        solver, edges = graph_data

        try:
            cycle, sum_cost, sum_time, ratio = solver.solve()

            # Same properties as integer mode
            assert cycle[0] == cycle[-1]
            assert abs(ratio - sum_cost / sum_time) < 1e-10
            assert sum_time > 0
            assert 2 <= len(cycle) <= len(edges) + 1

        except (RuntimeError, ValueError, NumericalInstabilityError):
            pass

    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=20)
    def test_complete_graph_properties(self, n):
        """
        Test properties on complete graphs with random weights.
        """
        solver = MinRatioCycleSolver(n)

        # Add edges for complete graph with random weights
        for u in range(n):
            for v in range(n):
                if u != v:  # No self-loops for this test
                    cost = np.random.randint(-10, 11)
                    time = np.random.randint(1, 6)
                    solver.add_edge(u, v, cost, time)

        cycle, sum_cost, sum_time, ratio = solver.solve()

        # Complete graph always has cycles
        assert len(cycle) >= 3
        assert cycle[0] == cycle[-1]
        assert sum_time > 0


class NaiveSolver:
    """
    Naive implementation for benchmarking.
    """

    def __init__(self, n: int):
        self.n = n
        self.edges: list[tuple[int, int, float, float]] = []

    def add_edge(self, u: int, v: int, cost: float, time: float):
        self.edges.append((u, v, cost, time))

    def solve(self) -> tuple[list[int], float, float, float]:
        """Brute force: enumerate all simple cycles up to length n."""
        best_ratio = float("inf")
        best_cycle = None
        best_cost = best_time = 0

        # Build adjacency list
        adj = [[] for _ in range(self.n)]
        for u, v, c, t in self.edges:
            adj[u].append((v, c, t))

        def dfs_cycles(path: list[int], visited: set[int], cost: float, time: float):
            nonlocal best_ratio, best_cycle, best_cost, best_time

            if len(path) > self.n:  # Avoid infinite recursion
                return

            u = path[-1]
            for v, c, t in adj[u]:
                new_cost = cost + c
                new_time = time + t

                if v == path[0] and len(path) >= 2:  # Found cycle
                    ratio = new_cost / new_time
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_cycle = path + [v]
                        best_cost = new_cost
                        best_time = new_time
                elif v not in visited:
                    dfs_cycles(path + [v], visited | {v}, new_cost, new_time)

        # Try starting from each vertex
        for start in range(self.n):
            dfs_cycles([start], {start}, 0, 0)

        if best_cycle is None:
            raise RuntimeError("No cycle found")

        return best_cycle, best_cost, best_time, best_ratio


class TestBenchmarks:
    """
    Benchmark comparisons.
    """

    def create_test_graph(self, n: int, density: float = 0.3) -> MinRatioCycleSolver:
        """
        Create a test graph with given density.
        """
        solver = MinRatioCycleSolver(n)

        # Add random edges
        num_edges = int(n * n * density)
        for _ in range(num_edges):
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            cost = np.random.randint(-10, 11)
            time = np.random.randint(1, 6)
            solver.add_edge(u, v, cost, time)

        return solver

    def test_performance_vs_naive_small(self):
        """
        Compare performance against naive implementation on small graphs.
        """
        for n in [3, 4, 5]:
            print(f"\nTesting n={n} vertices:")

            # Create same graph for both solvers
            np.random.seed(42)  # Reproducible
            edges = []
            for _ in range(n * 2):  # Dense enough to ensure cycles
                u = np.random.randint(0, n)
                v = np.random.randint(0, n)
                cost = np.random.randint(-5, 6)
                transit = np.random.randint(1, 4)
                edges.append((u, v, cost, transit))

            # Test our solver
            solver = MinRatioCycleSolver(n)
            for u, v, c, t in edges:
                solver.add_edge(u, v, c, t)

            start_time = time.time()
            try:
                cycle1, cost1, time1, ratio1 = solver.solve()
                our_time = time.time() - start_time
                print(f"  Our solver: {our_time:.6f}s, ratio={ratio1:.6f}")
            except:
                print("  Our solver: failed")
                continue

            # Test naive solver
            naive = NaiveSolver(n)
            for u, v, c, t in edges:
                naive.add_edge(u, v, c, t)

            start_time = time.time()
            try:
                cycle2, cost2, time2, ratio2 = naive.solve()
                naive_time = time.time() - start_time
                print(f"  Naive solver: {naive_time:.6f}s, ratio={ratio2:.6f}")

                # Check if ratios are close (both should find optimal)
                assert (
                    abs(ratio1 - ratio2) < 1e-6
                ), f"Ratio mismatch: {ratio1} vs {ratio2}"

            except:
                print("  Naive solver: failed")

    def test_scaling_performance(self):
        """
        Test how performance scales with graph size.
        """
        print("\nScaling performance test:")

        for n in [10, 20, 50, 100]:
            solver = self.create_test_graph(n, density=0.1)

            start_time = time.time()
            try:
                cycle, cost, time_sum, ratio = solver.solve()
                elapsed = time.time() - start_time
                print(f"  n={n:3d}: {elapsed:.6f}s, ratio={ratio:.6f}")
            except Exception as e:
                print(f"  n={n:3d}: failed ({e})")

    def test_density_impact(self):
        """
        Test how edge density affects performance.
        """
        print("\nDensity impact test (n=20):")

        for density in [0.05, 0.1, 0.2, 0.5]:
            solver = self.create_test_graph(20, density=density)

            start_time = time.time()
            try:
                cycle, cost, time_sum, ratio = solver.solve()
                elapsed = time.time() - start_time
                print(f"  density={density:.2f}: {elapsed:.6f}s, ratio={ratio:.6f}")
            except Exception as e:
                print(f"  density={density:.2f}: failed ({e})")


def test_memory_limit_enforced():
    config = SolverConfig(max_memory_mb=0)
    solver = MinRatioCycleSolver(2, config)
    solver.add_edge(0, 1, 1, 1)
    solver.add_edge(1, 0, 1, 1)
    with pytest.raises(ResourceExhaustionError):
        solver.solve()


if __name__ == "__main__":
    # Run some quick tests if executed directly
    print("Running quick validation tests...")

    # Test basic functionality
    test_edge = TestEdgeCases()
    test_edge.test_self_loop()
    test_edge.test_parallel_edges()
    print("✓ Edge cases passed")

    # Test correctness
    test_correct = TestCorrectnessValidation()
    test_correct.test_simple_triangle_validation()
    print("✓ Correctness validation passed")

    # Run benchmarks
    test_bench = TestBenchmarks()
    test_bench.test_performance_vs_naive_small()
    test_bench.test_scaling_performance()
    print("✓ Benchmarks completed")

    print("\nRun 'pytest test_solver.py -v' for full test suite")
