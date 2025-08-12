Algorithm Theory
================

Problem Statement
-----------------
Given a directed graph :math:`G=(V,E)` where each edge :math:`e\in E` has a
**cost** :math:`c_e` and a **time** :math:`t_e`, the goal is to find a cycle
:math:`C` that minimises the ratio

.. math::

   r(C) = \frac{\sum_{e\in C} c_e}{\sum_{e\in C} t_e}.

The optimal value :math:`r^*` is the minimum of :math:`r(C)` over all directed
cycles in the graph.

Lawler's Parametric Search
--------------------------
The solver follows Lawler's approach by repeatedly asking whether a candidate
ratio :math:`\lambda` is **too high**.  Each edge weight is shifted according to

.. math::

   w_\lambda(e) = c_e - \lambda t_e.

If the transformed graph contains a negative cycle, then a feasible solution
with ratio less than :math:`\lambda` exists; otherwise the optimum is at least
as large as :math:`\lambda`.  By binary searching on :math:`\lambda` we obtain a
ratio within a specified tolerance.

Binary Search Strategy
^^^^^^^^^^^^^^^^^^^^^^
The search maintains lower and upper bounds :math:`\lambda_{\text{lo}}` and
:math:`\lambda_{\text{hi}}` on the optimal ratio.  At each step the midpoint
:math:`\lambda=(\lambda_{\text{lo}}+\lambda_{\text{hi}})/2` is tested.  If a
negative cycle is found, the optimum lies below and the upper bound is
updated; otherwise the lower bound increases.  Iteration continues until
``hi - lo`` is below a user supplied tolerance, ensuring convergence in
:math:`O(\log((\lambda_{\text{hi}}-\lambda_{\text{lo}})/\text{tol}))`
iterations.

Worked Example
^^^^^^^^^^^^^^
Consider a triangle with edges :math:`0\to1\to2\to0` and edge weights
:math:`(c,t) = (2,1),(3,2),(1,1)`.  Testing :math:`\lambda=1` produces shifted
weights :math:`w_1 = (1,1,0)`.  The resulting negative cycle indicates the
ratio is less than one, so the binary search narrows the interval accordingly.

Stern–Brocot Exact Search
-------------------------
When all weights are integers the optimum ratio is rational.  The solver can
switch to an exact mode that performs a Stern–Brocot search on the tree of
positive rationals.  The algorithm maintains bounding fractions
:math:`\frac{a}{b} < r^* < \frac{c}{d}` and refines them using mediants
:math:`\frac{a+c}{b+d}` until the exact ratio is located, eliminating floating
point error.

Vectorised Bellman–Ford
-----------------------
Negative cycle queries are answered using a Bellman–Ford relaxation scheme
implemented with NumPy arrays.  For :math:`n` vertices and :math:`m` edges the
method performs :math:`O(nm)` relaxations but operates on whole arrays at a
time, yielding significant speedups over a pure Python implementation.

Complexity
----------
Combining the binary search with the relaxation phase yields a total running
time of :math:`O(nm \log((\lambda_{\text{hi}}-\lambda_{\text{lo}})/\text{tol}))`.
When the exact Stern–Brocot mode is used the search depth is bounded by the
size of the optimal numerator and denominator, providing a polynomial guarantee
for integer-weighted inputs.
