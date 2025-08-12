User Guide
==========

Installation
------------
The package is published on PyPI and can be installed with Poetry or pip::

  pip install min-ratio-cycle

For development, clone the repository and install in editable mode::

  git clone https://github.com/DiogoRibeiro7/min_ratio_cycle.git
  cd min_ratio_cycle
  poetry install

Basic Usage
-----------
Construct a solver from edge lists and retrieve the optimal cycle::

  from min_ratio_cycle.solver import MinRatioCycleSolver
  solver = MinRatioCycleSolver([(0, 1, 3, 1), (1, 2, 2, 1), (2, 0, 4, 1)])
  result = solver.solve()
  cycle, cost, time, ratio = result

Advanced Usage
--------------
The solver exposes both numeric and exact modes.  Use ``mode="exact"`` for
integer weights to avoid floating point error or ``tolerance`` to trade
accuracy for speed in numeric mode::

  solver.solve(mode="exact")
  solver.solve(tolerance=1e-6)

Error Handling
--------------
All solver issues derive from :class:`min_ratio_cycle.exceptions.SolverError`.
Typical failures include:

* :class:`~min_ratio_cycle.exceptions.GraphStructureError` – raised for
  disconnected graphs or invalid indices.  Recheck the edge list.
* :class:`~min_ratio_cycle.exceptions.NumericalInstabilityError` – occurs when
  weights span too many orders of magnitude.  Try scaling your inputs.
* :class:`~min_ratio_cycle.exceptions.ResourceExhaustionError` – emitted when
  memory or time limits are exceeded.  Increase limits or simplify the graph.

Each exception provides ``fix_suggestion`` and ``recovery_hint`` fields to help
resolve the problem.

Performance Tuning
------------------
Choose the numeric solver for dense graphs and the exact solver for small
integer-weight graphs.  Tighten ``tolerance`` for more precision, or loosen it
for faster solves.  The ``limit`` parameter caps relaxations to prevent runaway
iterations.

Real-World Examples
-------------------
The utility module ships with DIMACS helpers::

  from min_ratio_cycle.benchmarks import load_dimacs, benchmark_solver
  graph = load_dimacs("tests/data/triangle.dimacs")
  stats = benchmark_solver(graph)
  print(stats.runtime, stats.ratio)

Troubleshooting
---------------
==============  =============================  =================================
Symptom         Diagnosis                       Solution
==============  =============================  =================================
``NaN`` ratio   Graph contains negative times   Ensure all transit times > 0
Timeout         Graph too large                 Increase ``timeout`` or simplify
No cycle found  Graph is acyclic                Check input or allow exact mode
==============  =============================  =================================
