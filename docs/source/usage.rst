Usage Examples
==============

Basic solve
-----------

.. code-block:: python

   from min_ratio_cycle.solver import MinRatioCycleSolver, Edge

   edges = [
       Edge(0, 1, cost=2, time=1),
       Edge(1, 2, cost=3, time=2),
       Edge(2, 0, cost=1, time=1),
   ]

   solver = MinRatioCycleSolver(3)
   solver.add_edges(edges)
   result = solver.solve()
   print(result.cycle, result.ratio)

Advanced configuration
----------------------

.. code-block:: python

   from min_ratio_cycle.solver import MinRatioCycleSolver, SolverConfig, SolverMode

   config = SolverConfig(validate_cycles=True, log_level="INFO")
   solver = MinRatioCycleSolver(4, config)
   solver.add_edges([
       (0, 1, 5, 2),
       (1, 2, 3, 1),
       (2, 3, 2, 2),
       (3, 0, 1, 1),
   ])

   result = solver.solve(mode=SolverMode.NUMERIC)
   if result.metrics:
       print("iterations", result.metrics.iterations)

Statistical analysis
--------------------

.. code-block:: python

   from min_ratio_cycle.analytics import confidence_interval

   ratios = [0.8, 0.82, 0.81, 0.79]
   ci = confidence_interval(ratios)
   print("95% CI:", ci)

Sensitivity analysis
--------------------

.. code-block:: python

   from min_ratio_cycle.analytics import sensitivity_analysis

   solver = MinRatioCycleSolver(3)
   solver.add_edge(0, 1, cost=2, time=1)
   solver.add_edge(1, 2, cost=1, time=1)
   solver.add_edge(2, 0, cost=1, time=1)

   perturb = {(0, 1): {"cost": +1}}
   report = sensitivity_analysis(solver, perturb)
   print(report[(0, 1)])

Visualisation
-------------

.. code-block:: python

   result = solver.solve()
   result.visualize_solution(show_cycle=True)

Benchmarking
------------

.. code-block:: python

   from min_ratio_cycle.benchmarks import benchmark_solver

   runtime, ratio_diff = benchmark_solver(solver, ground_truth=-2.5)
   print("runtime", runtime)
