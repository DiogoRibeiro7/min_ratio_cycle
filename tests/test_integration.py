"""
Integration tests for MinRatioCycleSolver.

These tests focus on end-to-end behavior and real-world scenarios.
"""

import math

import pytest

from min_ratio_cycle.solver import MinRatioCycleSolver


class TestIntegration:
    """
    Integration tests for complete solver functionality.
    """

    def test_simple_known_solution(self, simple_triangle, graph_assertions):
        """
        Test solver on simple graph with known optimal solution.
        """
        solver, expected_ratio = simple_triangle

        cycle, sum_cost, sum_time, ratio = solver.solve()

        # Validate cycle structure
        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Check if we found the optimal or near-optimal solution
        assert (
            ratio <= expected_ratio + 1e-6
        ), f"Found ratio {ratio} is worse than expected {expected_ratio}"

    def test_negative_cost_cycle(self, negative_cycle, graph_assertions):
        """
        Test with negative cost cycles.
        """
        solver, expected_ratio = negative_cycle

        cycle, sum_cost, sum_time, ratio = solver.solve()

        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Should find negative ratio
        assert ratio < 0, "Should find negative ratio cycle"
        assert abs(ratio - expected_ratio) < 1e-6

    def test_mode_selection_integer(self, integer_weights_only):
        """
        Test automatic selection of exact mode for integer weights.
        """
        solver = integer_weights_only

        # Verify it's detected as integer mode
        assert solver._all_int == True, "Should detect integer weights"

        cycle, sum_cost, sum_time, ratio = solver.solve()

        # All returned values should be exact (within floating-point precision)
        assert isinstance(sum_cost, float)
        assert isinstance(sum_time, float)
        assert isinstance(ratio, float)

        # Values should be exactly representable as floats
        assert sum_cost == int(sum_cost), "Cost should be integer value"
        assert sum_time == int(sum_time), "Time should be integer value"

    def test_mode_selection_float(self, float_weights):
        """
        Test automatic selection of numeric mode for float weights.
        """
        solver = float_weights

        # Verify it's detected as float mode
        assert solver._all_int == False, "Should detect float weights"

        cycle, sum_cost, sum_time, ratio = solver.solve()

        # Should return valid results
        assert len(cycle) >= 3
        assert sum_time > 0
        assert math.isfinite(ratio)

    def test_disconnected_components(self, disconnected_graph, graph_assertions):
        """
        Test solver finds optimal cycle across disconnected components.
        """
        solver, expected_ratio = disconnected_graph

        cycle, sum_cost, sum_time, ratio = solver.solve()

        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Should find the best component's cycle
        assert abs(ratio - expected_ratio) < 1e-6

    def test_complete_graphs(self, complete_graph, graph_assertions):
        """
        Test solver on complete graphs of various sizes.
        """
        solver = complete_graph

        cycle, sum_cost, sum_time, ratio = solver.solve()

        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Complete graphs should always have solutions
        assert math.isfinite(ratio)

    def test_parallel_edges_handling(self, parallel_edges_graph, graph_assertions):
        """
        Test correct handling of parallel edges.
        """
        solver = parallel_edges_graph

        cycle, sum_cost, sum_time, ratio = solver.solve()

        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Should find reasonable solution
        assert ratio < 10  # Shouldn't use the worst edge

    @pytest.mark.slow
    def test_large_graph_performance(self, large_graph, graph_assertions):
        """
        Test solver performance on larger graphs.
        """
        import time

        solver = large_graph

        start_time = time.time()
        cycle, sum_cost, sum_time, ratio = solver.solve()
        elapsed = time.time() - start_time

        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Performance expectation
        assert elapsed < 5.0, f"Solver took too long: {elapsed:.2f}s"

        print(f"Large graph (n={solver.n}) solved in {elapsed:.4f}s")

    def test_pathological_case(self, pathological_graph, graph_assertions):
        """
        Test solver on pathologically difficult graph.
        """
        solver = pathological_graph

        cycle, sum_cost, sum_time, ratio = solver.solve()

        graph_assertions.assert_valid_cycle(cycle, solver.n)
        graph_assertions.assert_positive_time(sum_time)
        graph_assertions.assert_ratio_consistency(sum_cost, sum_time, ratio)

        # Should find the optimal negative-ratio cycle at the end
        assert ratio < 0, "Should find the negative ratio cycle"


class TestRealWorldScenarios:
    """
    Tests based on real-world applications.
    """

    def test_arbitrage_detection(self):
        """
        Test currency arbitrage detection scenario.
        """
        # Simplified currency exchange graph
        # Vertices: 0=USD, 1=EUR, 2=GBP, 3=JPY
        solver = MinRatioCycleSolver(4)

        # Exchange rates (costs) and times
        # Looking for negative cost cycles (arbitrage opportunities)
        solver.add_edge(0, 1, -math.log(0.85), 1)  # USD -> EUR
        solver.add_edge(1, 2, -math.log(1.15), 1)  # EUR -> GBP
        solver.add_edge(2, 3, -math.log(150.0), 1)  # GBP -> JPY
        solver.add_edge(3, 0, -math.log(0.0075), 1)  # JPY -> USD

        # Add some reverse edges
        solver.add_edge(1, 0, -math.log(1.18), 1)  # EUR -> USD
        solver.add_edge(2, 1, -math.log(0.87), 1)  # GBP -> EUR

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # In efficient markets, shouldn't find significant arbitrage
        # But test structure is more important than specific values

    def test_resource_scheduling(self):
        """
        Test resource scheduling optimization.
        """
        # Graph where vertices are tasks, edges are dependencies
        # Cost = resource usage, Time = duration
        solver = MinRatioCycleSolver(5)

        # Create a workflow with cyclic dependencies (bad design)
        solver.add_edge(0, 1, 10, 2)  # Task 0 -> Task 1: 10 resources, 2 time
        solver.add_edge(1, 2, 15, 3)  # Task 1 -> Task 2: 15 resources, 3 time
        solver.add_edge(2, 3, 8, 1)  # Task 2 -> Task 3: 8 resources, 1 time
        solver.add_edge(3, 4, 12, 4)  # Task 3 -> Task 4: 12 resources, 4 time
        solver.add_edge(4, 1, 5, 2)  # Task 4 -> Task 1: 5 resources, 2 time (cycle!)

        # Alternative paths
        solver.add_edge(0, 3, 20, 5)  # Direct 0 -> 3
        solver.add_edge(2, 0, 6, 1)  # Task 2 -> Task 0

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # The cycle represents the most efficient resource/time loop
        resource_efficiency = ratio  # resources per unit time
        assert resource_efficiency > 0  # Should be positive for this example

        print(
            f"Most efficient cycle uses {resource_efficiency:.2f} resources per time unit"
        )

    def test_network_routing(self):
        """
        Test network routing with cost and latency.
        """
        # Network nodes with routing costs and latencies
        solver = MinRatioCycleSolver(6)

        # Network topology (simplified)
        # Cost = bandwidth cost, Time = latency
        routes = [
            (0, 1, 5, 10),  # Node 0 -> 1: cost 5, latency 10ms
            (1, 2, 3, 5),  # Node 1 -> 2: cost 3, latency 5ms
            (2, 3, 7, 15),  # Node 2 -> 3: cost 7, latency 15ms
            (3, 4, 2, 8),  # Node 3 -> 4: cost 2, latency 8ms
            (4, 5, 4, 12),  # Node 4 -> 5: cost 4, latency 12ms
            (5, 0, 6, 20),  # Node 5 -> 0: cost 6, latency 20ms (completes cycle)
            # Alternative routes
            (0, 2, 12, 18),  # Direct 0 -> 2
            (1, 4, 8, 25),  # Direct 1 -> 4
            (3, 0, 15, 30),  # Direct 3 -> 0
        ]

        for u, v, cost, latency in routes:
            solver.add_edge(u, v, cost, latency)

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # Ratio represents cost per unit latency
        cost_per_latency = ratio
        print(
            f"Most efficient routing cycle: {cost_per_latency:.3f} cost per latency unit"
        )

    def test_manufacturing_process(self):
        """
        Test manufacturing process optimization.
        """
        # Manufacturing stages with setup costs and processing times
        solver = MinRatioCycleSolver(4)

        # Production line: Raw -> Process1 -> Process2 -> Quality -> (back to Raw)
        solver.add_edge(0, 1, 100, 30)  # Raw -> Process1: $100 setup, 30min
        solver.add_edge(1, 2, 150, 45)  # Process1 -> Process2: $150, 45min
        solver.add_edge(2, 3, 80, 20)  # Process2 -> Quality: $80, 20min
        solver.add_edge(3, 0, 50, 15)  # Quality -> Raw: $50, 15min (rework cycle)

        # Alternative processes
        solver.add_edge(0, 2, 200, 60)  # Raw -> Process2 directly
        solver.add_edge(1, 3, 120, 35)  # Process1 -> Quality directly

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # Ratio represents cost per minute
        cost_per_minute = ratio
        assert cost_per_minute > 0

        print(f"Most efficient manufacturing cycle: ${cost_per_minute:.2f} per minute")


class TestEdgeConfiguration:
    """
    Test various edge configurations and graph properties.
    """

    def test_self_loops_only(self):
        """
        Test graph with only self-loops.
        """
        solver = MinRatioCycleSolver(3)

        # Each vertex has a self-loop
        solver.add_edge(0, 0, 5, 2)  # ratio 2.5
        solver.add_edge(1, 1, 3, 2)  # ratio 1.5
        solver.add_edge(2, 2, 7, 4)  # ratio 1.75

        cycle, sum_cost, sum_time, ratio = solver.solve()

        # Should find the best self-loop
        assert len(cycle) == 2  # Just the vertex and back to itself
        assert cycle[0] == cycle[1]
        assert abs(ratio - 1.5) < 1e-10  # Best ratio is vertex 1

    def test_star_topology(self):
        """
        Test star topology (one central vertex).
        """
        n = 6
        center = 0
        solver = MinRatioCycleSolver(n)

        # Spokes from center to all other vertices
        for i in range(1, n):
            solver.add_edge(center, i, i * 2, i)  # center -> spoke
            solver.add_edge(i, center, i + 1, 1)  # spoke -> center

        # Connect spokes to create cycles through center
        for i in range(1, n - 1):
            solver.add_edge(i, i + 1, 1, 2)  # spoke -> next spoke

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # Should find some cycle through the center
        assert center in cycle[:-1]  # Center should be in the cycle

    def test_bipartite_graph(self):
        """
        Test bipartite graph structure.
        """
        # Vertices 0,1,2 in one set, 3,4,5 in other set
        solver = MinRatioCycleSolver(6)

        # Edges only between the two sets
        for u in [0, 1, 2]:
            for v in [3, 4, 5]:
                cost = abs(u - v) + 1
                time = (u + v) % 3 + 1
                solver.add_edge(u, v, cost, time)
                solver.add_edge(v, u, cost + 1, time)  # Return edge

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # In bipartite graph, cycles must have even length
        assert (len(cycle) - 1) % 2 == 0, "Bipartite graph cycles must have even length"

    def test_layered_graph(self):
        """
        Test layered/hierarchical graph structure.
        """
        layers = 4
        nodes_per_layer = 3
        n = layers * nodes_per_layer
        solver = MinRatioCycleSolver(n)

        def node_id(layer: int, pos: int) -> int:
            return layer * nodes_per_layer + pos

        # Forward edges between consecutive layers
        for layer in range(layers - 1):
            for pos in range(nodes_per_layer):
                for next_pos in range(nodes_per_layer):
                    u = node_id(layer, pos)
                    v = node_id(layer + 1, next_pos)
                    cost = abs(pos - next_pos) + 1
                    time = layer + 1
                    solver.add_edge(u, v, cost, time)

        # Feedback edges from last layer to first layer
        for pos in range(nodes_per_layer):
            for start_pos in range(nodes_per_layer):
                u = node_id(layers - 1, pos)
                v = node_id(0, start_pos)
                cost = pos + start_pos + 1
                time = layers  # Long feedback time
                solver.add_edge(u, v, cost, time)

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0

        # Cycle should span multiple layers
        cycle_layers = set()
        for vertex in cycle[:-1]:  # Exclude repeated last vertex
            layer = vertex // nodes_per_layer
            cycle_layers.add(layer)

        assert len(cycle_layers) >= 2, "Cycle should span multiple layers"


class TestNumericalStability:
    """
    Test numerical stability and precision issues.
    """

    def test_very_small_differences(self):
        """
        Test with very small differences between edge weights.
        """
        solver = MinRatioCycleSolver(3)

        eps = 1e-12
        solver.add_edge(0, 1, 1.0, 1.0)
        solver.add_edge(1, 2, 1.0 + eps, 1.0)
        solver.add_edge(2, 0, 1.0 - eps, 1.0)

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0
        assert math.isfinite(ratio)

        # Should handle small differences without numerical issues
        expected_ratio = (1.0 + (1.0 + eps) + (1.0 - eps)) / 3.0  # = 1.0
        assert abs(ratio - 1.0) < 1e-10

    def test_mixed_scales(self):
        """
        Test with very different weight scales.
        """
        solver = MinRatioCycleSolver(3)

        large = 1e6
        small = 1e-6

        solver.add_edge(0, 1, large, 1.0)
        solver.add_edge(1, 2, small, large)
        solver.add_edge(2, 0, 1.0, 1.0)

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0
        assert math.isfinite(ratio)

        # Should handle mixed scales correctly
        print(f"Mixed scales result: ratio = {ratio:.6e}")

    def test_near_zero_ratio(self):
        """
        Test graphs with near-zero optimal ratio.
        """
        solver = MinRatioCycleSolver(4)

        # Create cycle with costs that nearly cancel out
        solver.add_edge(0, 1, 1000, 1)
        solver.add_edge(1, 2, -999.99, 1)
        solver.add_edge(2, 3, -0.005, 1)
        solver.add_edge(3, 0, -0.005, 1)

        cycle, sum_cost, sum_time, ratio = solver.solve()

        assert len(cycle) >= 3
        assert sum_time > 0
        assert math.isfinite(ratio)

        # Should find near-zero ratio accurately
        assert abs(ratio) < 0.1  # Should be very small
        print(f"Near-zero ratio result: {ratio:.6f}")


class TestEndToEndWorkflows:
    """
    End-to-end tests covering full solver workflow and error propagation.
    """

    def test_dimacs_benchmark_end_to_end(self):
        """
        Load DIMACS data, solve, and validate performance metrics.
        """
        import psutil

        from min_ratio_cycle.benchmarks import benchmark_solver, load_dimacs_graph

        lines = [
            "c simple triangle",
            "p 3 3",
            "a 1 2 1 1",
            "a 2 3 1 1",
            "a 3 1 1 1",
        ]
        solver = load_dimacs_graph(lines)
        proc = psutil.Process()
        before = proc.memory_info().rss
        stats = benchmark_solver(solver)
        after = proc.memory_info().rss
        assert stats["diff"] is None or stats["diff"] < 1e-9
        assert 0 < stats["time"] < 1.0
        assert after - before < 50 * 1024 * 1024

    def test_solver_deterministic_results(self):
        """
        Solver should produce consistent results across runs.
        """
        edges = [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (2, 0, 1.0, 1.0)]
        s1 = MinRatioCycleSolver(3)
        s1.add_edges(edges)
        s2 = MinRatioCycleSolver(3)
        s2.add_edges(edges)
        assert s1.solve().ratio == s2.solve().ratio

    def test_error_propagation_disconnected(self):
        """
        Disconnected graphs surface solver errors with context.
        """
        from min_ratio_cycle.benchmarks import benchmark_solver, load_dimacs_graph
        from min_ratio_cycle.exceptions import SolverError

        lines = [
            "p 4 2",
            "a 1 2 1 1",
            "a 3 4 1 1",
        ]
        solver = load_dimacs_graph(lines)
        with pytest.raises(SolverError) as excinfo:
            benchmark_solver(solver)
        details = excinfo.value.details
        assert "suggested_fix" in details
        assert "recovery_hint" in details


if __name__ == "__main__":
    # Quick integration test runner
    import sys

    sys.path.insert(0, ".")

    # Test basic integration
    print("Running integration tests...")

    try:
        # Simple test
        solver = MinRatioCycleSolver(3)
        solver.add_edge(0, 1, 2, 1)
        solver.add_edge(1, 2, 3, 2)
        solver.add_edge(2, 0, 1, 1)

        cycle, cost, time, ratio = solver.solve()
        print(f"✓ Basic test: cycle={cycle}, ratio={ratio:.4f}")

        # Real-world scenario test
        tester = TestRealWorldScenarios()
        tester.test_arbitrage_detection()
        print("✓ Arbitrage detection test passed")

        tester.test_resource_scheduling()
        print("✓ Resource scheduling test passed")

        print("\nIntegration tests completed successfully!")
        print("Run 'pytest test_integration.py -v' for full test suite")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
