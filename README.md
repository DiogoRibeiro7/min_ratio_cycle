# min-ratio-cycle

[![CI](https://github.com/DiogoRibeiro7/min_ratio_cycle/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/min_ratio_cycle/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiogoRibeiro7/min-ratio-cycle/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/min-ratio-cycle)
[![Documentation](https://readthedocs.org/projects/min-ratio-cycle/badge/?version=latest)](https://min-ratio-cycle.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/min-ratio-cycle.svg)](https://badge.fury.io/py/min-ratio-cycle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A Python library for finding minimum cost-to-time ratio cycles in directed graphs.

## Overview

`min_ratio_cycle` implements algorithms for detecting minimum mean cycles where edge weights are represented as (cost, time) pairs. It supports exact rational results for integer inputs and numeric approximations when appropriate.

## Installation

```bash
pip install min-ratio-cycle
```

## Usage

```python
from min_ratio_cycle import MinRatioCycleSolver

solver = MinRatioCycleSolver(n_vertices=3)
solver.add_edge(0, 1, cost=2, time=1)
solver.add_edge(1, 2, cost=3, time=2)
solver.add_edge(2, 0, cost=1, time=1)

cycle, cost, time, ratio = solver.solve()
print(cycle, cost, time, ratio)
```

## Features

- Solve minimum cost/time ratio cycle problems in directed graphs
- Support for integer and floating-point weights
- Configurable precision and timeout settings
- Basic integration with NetworkX for graph input/output

## Configuration

```python
from min_ratio_cycle import SolverConfig, MinRatioCycleSolver

config = SolverConfig(numeric_tolerance=1e-12, max_solve_time=30.0)
solver = MinRatioCycleSolver(n_vertices=100, config=config)
```

## Testing

Run tests with pytest:

```bash
pytest
```

## License

MIT
