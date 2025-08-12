"""
Shared pytest fixtures for min-ratio-cycle testing.
"""

import pytest
import numpy as np
from typing import List, Tuple
from min_ratio_cycle.solver import MinRatioCycleSolver


@pytest.fixture
def simple_triangle():
    """Simple 3-vertex triangle with known optimal solution."""
    solver = MinRatioCycleSolver(3)
    solver.add_edge(0, 1, 2, 1)  # ratio 2
    solver.add_edge(1, 2, 3, 2)  # ratio 1.5  
    solver.add_edge(2, 0, 1, 1)  # ratio 1
    # Optimal cycle: 1->2->0->1 with ratio (3+1+2)/(2+1+1) = 6/4 = 1.5
    return solver, 1.5


@pytest.fixture
def negative_cycle():
    """Graph with negative cost cycle for minimum ratio."""
    solver = MinRatioCycleSolver(3)
    solver.add_edge(0, 1, -1, 1)  # ratio -1
    solver.add_edge(1, 2, -2, 1)  # ratio -2
    solver.add_edge(2, 0, 1, 1)   # ratio 1
    # Optimal cycle: 0->1->2->0 with ratio (-1-2+1)/(1+1+1) = -2/3
    return solver, -2/3


@pytest.fixture  
def integer_weights_only():
    """Graph with only integer weights (should use exact mode)."""
    solver = MinRatioCycleSolver(4)
    solver.add_edge(0, 1, 5, 2)
    solver.add_edge(1, 2, 3, 1) 
    solver.add_edge(2, 3, 2, 2)
    solver.add_edge(3, 0, 1, 1)
    return solver


@pytest.fixture
def float_weights():
    """Graph with float weights (should use numeric mode)."""
    solver = MinRatioCycleSolver(4)
    solver.add_edge(0, 1, 5.5, 2.0)
    solver.add_edge(1, 2, 3.2, 1.1)
    solver.add_edge(2, 3, 2.7, 2.3) 
    solver.add_edge(3, 0, 1.1, 1.0)
    return solver


@pytest.fixture
def large_graph():
    """Larger graph for performance testing."""
    n = 50
    solver = MinRatioCycleSolver(n)
    
    # Create a graph with guaranteed cycles
    np.random.seed(42)  # Reproducible
    
    # Add a simple cycle through all vertices
    for i in range(n):
        next_v = (i + 1) % n
        cost = np.random.randint(-5, 6)
        time = np.random.randint(1, 4)
        solver.add_edge(i, next_v, cost, time)
    
    # Add random additional edges
    for _ in range(n * 2):
        u = np.random.randint(0, n)
        v = np.random.randint(0, n)
        cost = np.random.randint(-10, 11)
        time = np.random.randint(1, 6)
        solver.add_edge(u, v, cost, time)
    
    return solver


@pytest.fixture
def disconnected_graph():
    """Graph with multiple disconnected components."""
    solver = MinRatioCycleSolver(8)
    
    # Component 1: triangle 0-1-2
    solver.add_edge(0, 1, 3, 2)
    solver.add_edge(1, 2, 2, 1)
    solver.add_edge(2, 0, 1, 1)
    
    # Component 2: triangle 3-4-5 with better ratio
    solver.add_edge(3, 4, 1, 2)
    solver.add_edge(4, 5, 1, 2) 
    solver.add_edge(5, 3, 0, 1)  # completes cycle with ratio 0.4

    # Isolated vertices: 6, 7

    return solver, 0.4  # Best possible ratio


@pytest.fixture(params=[3, 5, 7, 10])
def complete_graph(request):
    """Complete graphs of various sizes."""
    n = request.param
    solver = MinRatioCycleSolver(n)
    
    np.random.seed(42)  # Reproducible
    for u in range(n):
        for v in range(n):
            if u != v:
                cost = np.random.randint(-3, 4)
                time = np.random.randint(1, 3)
                solver.add_edge(u, v, cost, time)
    
    return solver


@pytest.fixture
def pathological_graph():
    """Graph designed to stress-test the algorithm."""
    n = 20
    solver = MinRatioCycleSolver(n)
    
    # Long path with expensive edges
    for i in range(n-1):
        solver.add_edge(i, i+1, 100, 1)
    
    # Cheap cycle at the end
    solver.add_edge(n-3, n-2, 1, 10)
    solver.add_edge(n-2, n-1, 1, 10)
    solver.add_edge(n-1, n-3, -10, 1)  # Makes this cycle optimal
    
    return solver


@pytest.fixture
def parallel_edges_graph():
    """Graph with multiple edges between same vertex pairs."""
    solver = MinRatioCycleSolver(3)
    
    # Multiple 0->1 edges with different costs/times
    solver.add_edge(0, 1, 10, 2)  # ratio 5
    solver.add_edge(0, 1, 6, 3)   # ratio 2
    solver.add_edge(0, 1, 8, 1)   # ratio 8
    
    # Return path
    solver.add_edge(1, 2, 2, 1)   # ratio 2
    solver.add_edge(2, 0, 1, 1)   # ratio 1
    
    return solver


class GraphAssertions:
    """Helper class for graph-specific assertions."""
    
    @staticmethod
    def assert_valid_cycle(cycle: List[int], n_vertices: int):
        """Assert that a cycle is valid."""
        assert isinstance(cycle, list)
        assert len(cycle) >= 3, "Cycle must have at least 3 vertices"
        assert cycle[0] == cycle[-1], "Cycle must be closed"
        
        # All vertices should be valid indices
        for v in cycle:
            assert 0 <= v < n_vertices, f"Invalid vertex {v}"
        
        # No consecutive duplicates (except first/last)
        for i in range(len(cycle) - 1):
            if i == len(cycle) - 2:  # Last edge back to start
                continue
            assert cycle[i] != cycle[i+1], f"Consecutive duplicate at {i}"
    
    @staticmethod
    def assert_positive_time(sum_time: float):
        """Assert that total cycle time is positive."""
        assert sum_time > 0, "Cycle time must be positive"
    
    @staticmethod
    def assert_ratio_consistency(sum_cost: float, sum_time: float, ratio: float):
        """Assert that ratio equals cost/time."""
        expected_ratio = sum_cost / sum_time
        assert abs(ratio - expected_ratio) < 1e-10, \
            f"Ratio inconsistency: {ratio} != {expected_ratio}"


@pytest.fixture
def graph_assertions():
    """Provide graph assertion helper."""
    return GraphAssertions()


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark"
    )
    config.addinivalue_line(
        "markers", "property: mark test as property-based"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark property-based tests
        if "hypothesis" in item.name or "property" in item.name:
            item.add_marker(pytest.mark.property)
        
        # Mark benchmark tests  
        if "benchmark" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.benchmark)
        
        # Mark slow tests
        if "large" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.slow)
