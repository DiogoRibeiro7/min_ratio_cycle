# Testing Framework for Min Ratio Cycle Solver

This comprehensive testing suite provides extensive validation, performance analysis, and correctness verification for the MinRatioCycleSolver.

## 📋 Test Categories

### 1. Edge Cases and Degenerate Scenarios (`TestEdgeCases`)
- **Empty graphs**: Graphs with no edges
- **No cycles**: Tree structures that should fail gracefully
- **Self-loops**: Single vertex cycles
- **Parallel edges**: Multiple edges between same vertices
- **Disconnected components**: Multiple separate graph components
- **Invalid inputs**: Negative times, invalid vertex indices
- **Large weights**: Testing numerical limits

### 2. Correctness Validation (`TestCorrectnessValidation`)
- **Cycle validation**: Ensures returned cycles are properly formed
- **Weight verification**: Manually verifies cost/time sums match reported values
- **Ratio consistency**: Confirms ratio = sum_cost / sum_time
- **Known solutions**: Tests against graphs with predetermined optimal solutions

### 3. Property-Based Testing (`TestPropertyBased`)
- **Hypothesis-driven**: Automatically generates hundreds of random test cases
- **Integer vs Float modes**: Validates both exact and numeric solver paths
- **Invariant checking**: Ensures mathematical properties always hold
- **Complete graphs**: Tests on fully connected graphs with random weights
- **Regression detection**: Catches edge cases that manual tests might miss

### 4. Performance Benchmarks (`TestBenchmarks`, `benchmark_suite.py`)
- **Scaling analysis**: How performance changes with graph size
- **Density impact**: Effect of edge density on solve time
- **Mode comparison**: Exact vs numeric solver performance
- **Memory profiling**: Peak memory usage tracking
- **Naive algorithm comparison**: Validates our solver is actually faster

### 5. Integration Testing (`TestIntegration`)
- **End-to-end workflows**: Complete solver usage scenarios
- **Real-world applications**: Currency arbitrage, resource scheduling, network routing
- **Graph topologies**: Stars, grids, bipartite, layered structures
- **Numerical stability**: Very small differences, mixed scales, near-zero ratios

## 🛠 Test Infrastructure

### Core Files
```
test_solver.py          # Main test suite with edge cases and property tests
test_integration.py     # Integration and real-world scenario tests
conftest.py            # Shared fixtures and test utilities
benchmark_suite.py     # Comprehensive performance analysis
pytest.ini            # Pytest configuration
run_tests.py          # Test runner with multiple execution modes
Makefile              # Development automation commands
```

### Key Features

#### 1. **Shared Fixtures** (`conftest.py`)
Pre-built test graphs for common scenarios:
- `simple_triangle`: Known optimal solution
- `negative_cycle`: Tests negative cost handling
- `large_graph`: Performance testing (50+ vertices)
- `disconnected_graph`: Multiple components
- `complete_graph`: Parameterized complete graphs
- `pathological_graph`: Stress-testing scenarios

#### 2. **Graph Assertions** (`GraphAssertions`)
Specialized validation helpers:
```python
def assert_valid_cycle(cycle, n_vertices)
def assert_positive_time(sum_time)
def assert_ratio_consistency(cost, time, ratio)
```

#### 3. **Property-Based Testing**
Uses Hypothesis to generate random graphs:
```python
@given(random_graph(max_vertices=6, max_edges=15, integer_weights=True))
def test_integer_mode_consistency(self, graph_data):
    # Automatically tests hundreds of random integer graphs
```

#### 4. **Benchmark Framework**
Comprehensive performance analysis:
- **Scaling tests**: Graph sizes from 10 to 100+ vertices
- **Topology comparison**: Random, complete, grid, cycle graphs
- **Memory tracking**: Peak memory usage per test
- **Visualization**: Automatic plot generation of results

## 🚀 Running Tests

### Quick Start
```bash
# Install dependencies
make install-dev

# Run all tests
make test

# Run only fast tests
make test-quick

# Generate coverage report
make coverage
```

### Advanced Usage
```bash
# Property-based tests (extensive random testing)
python run_tests.py --property

# Performance benchmarks
python run_tests.py --bench
python benchmark_suite.py

# Integration tests only
python run_tests.py --integration

# Specific test patterns
python run_tests.py --pattern "negative"
python run_tests.py --file test_integration.py

# Parallel execution with coverage
python run_tests.py --parallel --coverage
```

### Test Markers
Tests are categorized with pytest markers:
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.benchmark`: Performance tests
- `@pytest.mark.property`: Property-based tests
- `@pytest.mark.integration`: End-to-end tests

Exclude slow tests: `pytest -m "not slow"`

## 📊 Benchmark Results

The benchmark suite provides detailed analysis:

### Performance Metrics
- **Solve time**: Wall-clock time for `solver.solve()`
- **Memory usage**: Peak memory consumption
- **Success rate**: Percentage of graphs that solve successfully
- **Mode comparison**: Exact vs numeric algorithm performance

### Graph Topologies Tested
1. **Random sparse** (10% density)
2. **Random dense** (30-50% density)
3. **Complete graphs** (100% density, smaller sizes)
4. **Grid graphs** (2D lattice structure)
5. **Cycle graphs** (Simple ring topology)
6. **Star graphs** (Central hub topology)

### Scaling Analysis
Tests graph sizes from 10 to 100+ vertices and generates plots showing:
- Time vs vertices (log scale)
- Time vs edge count
- Memory vs graph size
- Exact vs numeric mode comparison

## ✅ Correctness Validation

### Manual Verification
For critical test cases, we manually verify results:

```python
def validate_cycle(self, solver, cycle, expected_cost, expected_time, expected_ratio):
    # Walk through cycle edges manually
    actual_cost = sum(edge_costs_in_cycle)
    actual_time = sum(edge_times_in_cycle)

    # Verify against solver output
    assert abs(actual_cost - expected_cost) < 1e-10
    assert abs(actual_ratio - expected_ratio) < 1e-10
```

### Property Invariants
Every valid solution must satisfy:
1. **Cycle closure**: `cycle[0] == cycle[-1]`
2. **Positive time**: `sum_time > 0`
3. **Ratio consistency**: `ratio == sum_cost / sum_time`
4. **Valid vertices**: All vertices in `[0, n_vertices)`
5. **Edge existence**: All cycle edges exist in graph

### Known Solution Tests
Test cases with predetermined correct answers:
- Simple triangles with calculated optimal ratios
- Negative cost cycles
- Self-loops (trivial cycles)
- Disconnected components (should find global optimum)

## 🔧 Stress Testing

### Numerical Edge Cases
- **Large integers**: Values up to 10^12
- **Small differences**: Edge weights differing by 1e-12
- **Mixed scales**: Combining 1e6 and 1e-6 values
- **Near-zero ratios**: Costs that nearly cancel out

### Pathological Graphs
- **Long chains**: 100+ vertex paths with cycle at end
- **Dense complete graphs**: n² edges with random weights
- **Precision stress**: Weights designed to challenge floating-point arithmetic

### Error Handling
- **Invalid inputs**: Negative times, out-of-bound vertices
- **Impossible scenarios**: Graphs guaranteed to have no cycles
- **Resource limits**: Graphs approaching memory/time limits

## 📈 Continuous Integration

### CI Pipeline (`make ci`)
1. **Code quality**: Linting, formatting, type checking
2. **Fast tests**: Core functionality validation
3. **Coverage**: Ensure >90% code coverage
4. **Performance**: Basic scaling verification

### Pre-commit Hooks (`make pre-commit`)
1. **Format code**: Black, isort
2. **Quality checks**: Flake8, mypy
3. **Quick tests**: Fast validation before commit

## 🎯 Test Coverage Goals

Target coverage metrics:
- **Line coverage**: >95%
- **Branch coverage**: >90%
- **Function coverage**: 100%

Critical areas requiring full coverage:
- Core solver algorithms (Bellman-Ford, Stern-Brocot)
- Edge case handling
- Mode selection logic
- Cycle extraction and validation

## 🤝 Contributing Tests

When adding new tests:

1. **Follow naming convention**: `test_descriptive_name`
2. **Use appropriate markers**: `@pytest.mark.slow` for expensive tests
3. **Add fixtures**: Reusable test graphs in `conftest.py`
4. **Document expected behavior**: Clear docstrings explaining test purpose
5. **Include edge cases**: Think about failure modes
6. **Validate correctness**: Don't just check for no exceptions

### Example Test Structure
```python
def test_new_feature(self, graph_fixture, graph_assertions):
    """Test description explaining what we're validating."""
    # Arrange
    solver = setup_test_case()

    # Act
    cycle, cost, time, ratio = solver.solve()

    # Assert
    graph_assertions.assert_valid_cycle(cycle, solver.n)
    assert specific_property_holds(cycle, cost, time, ratio)
    # Explain why this assertion matters
```

## 🐛 Debugging Test Failures

### Common Issues
1. **Numerical precision**: Use appropriate tolerances (1e-10 for exact, 1e-6 for numeric)
2. **Random test failures**: Set `np.random.seed()` for reproducibility
3. **Performance regressions**: Compare against baseline times
4. **Memory leaks**: Check peak memory doesn't grow unexpectedly

### Debugging Tools
```bash
# Run single test with verbose output
pytest test_solver.py::TestEdgeCases::test_self_loop -v -s

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --tb=long

# Profile slow tests
pytest --durations=10
```

This comprehensive testing framework ensures the MinRatioCycleSolver is robust, performant, and mathematically correct across a wide range of scenarios and edge cases.
