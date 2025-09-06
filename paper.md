---
title: >-
  min-ratio-cycle: A high-performance Python library for minimum cost-to-time
  ratio cycle detection in directed graphs
tags:
  - Python
  - graph algorithms
  - optimization
  - operations research
  - computational geometry
  - parametric search
authors:
  - name: Diogo de Bastos Ribeiro
    orcid: 0009-0001-2022-7072
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: 'ESMAD – Instituto Politécnico do Porto, Portugal'
    index: 1
date: 17 January 2025
bibliography: paper.bib
---

# Summary

The minimum cost-to-time ratio cycle problem seeks to find a directed cycle in a graph that minimizes the ratio of total edge costs to total edge times. This fundamental optimization problem appears across diverse domains including financial arbitrage detection, resource scheduling, network routing, and supply chain optimization. While conceptually straightforward, efficient algorithms for this problem require sophisticated techniques from parametric optimization and negative cycle detection.

`min-ratio-cycle` is a high-performance Python library that implements state-of-the-art algorithms for solving the minimum cost-to-time ratio cycle problem. The library features NumPy-accelerated implementations of Lawler's parametric search [@lawler1976minimum] combined with vectorized Bellman-Ford negative cycle detection, achieving significant performance improvements over naive implementations. For integer-weighted graphs, the library provides an exact arithmetic mode using Stern-Brocot tree search [@graham1994concrete] that eliminates floating-point precision errors entirely.

# Statement of need

The minimum cost-to-time ratio cycle problem, while fundamental in combinatorial optimization, lacks accessible, high-performance implementations in popular programming languages. Existing solutions are typically either toy implementations with exponential complexity, or embedded within larger optimization frameworks that are difficult to use for this specific problem.

Current approaches suffer from several limitations:

1. **Performance**: Naive implementations using cycle enumeration have factorial time complexity, making them impractical for graphs with more than a few dozen vertices.

2. **Numerical precision**: Floating-point implementations suffer from precision loss, particularly problematic in financial applications where exact arithmetic is crucial.

3. **Usability**: Existing implementations often lack proper error handling, validation, and debugging tools necessary for production use.

4. **Extensibility**: Most implementations are monolithic and difficult to extend for domain-specific requirements.

`min-ratio-cycle` addresses these limitations by providing a production-ready library with optimal algorithmic complexity O(nm log(1/ε)), exact arithmetic capabilities, comprehensive validation, and extensive analytics tools.

# Mathematical Background

The minimum cost-to-time ratio cycle problem can be formally stated as follows: Given a directed graph G = (V, E) where each edge e ∈ E has an associated cost c(e) and positive time t(e), find a cycle C that minimizes:

$$r(C) = \frac{\sum_{e \in C} c(e)}{\sum_{e \in C} t(e)}$$

The optimal solution employs Lawler's parametric search approach [@lawler1976minimum], which transforms the problem into a sequence of negative cycle detection queries. For a candidate ratio λ, we reweight each edge e with:

$$w_\lambda(e) = c(e) - \lambda \cdot t(e)$$

The minimum feasible λ for which the reweighted graph contains no negative cycles equals the optimal cost-to-time ratio. This reduction enables the use of efficient negative cycle detection algorithms such as Bellman-Ford, yielding an overall complexity of O(nm log(1/ε)) where n = |V|, m = |E|, and ε is the desired precision.

For integer-weighted graphs, the optimal ratio is rational and can be found exactly using Stern-Brocot tree search [@graham1994concrete], eliminating floating-point errors entirely.

# Software Architecture

The library is organized into several key components:

- **Core Solver** (`MinRatioCycleSolver`): Main interface providing multiple solving modes (automatic, exact, numeric, approximate)
- **Algorithm Implementations**: Vectorized Bellman-Ford with NumPy acceleration and exact Stern-Brocot search
- **Validation Framework**: Comprehensive input validation and result verification
- **Analytics Module**: Sensitivity analysis, performance benchmarking, and visualization tools
- **Configuration System**: Flexible parameter tuning for different use cases
- **Monitoring Infrastructure**: Performance metrics collection and debugging utilities

The design emphasizes both performance and usability, with automatic algorithm selection based on input characteristics, extensive error handling with actionable error messages, and rich debugging capabilities for troubleshooting complex scenarios.

# Performance and Validation

`min-ratio-cycle` has been extensively benchmarked against theoretical complexity bounds and alternative implementations. Performance testing shows near-linear scaling with graph size for sparse graphs and quadratic scaling for dense graphs, matching theoretical expectations. Memory usage remains bounded at O(V + E) with typical overhead under 100MB for graphs with thousands of vertices.

The library includes a comprehensive test suite with over 500 test cases covering:

- **Edge cases**: Empty graphs, disconnected components, self-loops, parallel edges
- **Correctness validation**: Manual verification against known optimal solutions
- **Property-based testing**: Automated generation of random test cases using Hypothesis
- **Performance regression testing**: Automated benchmarking to prevent performance degradation
- **Real-world scenarios**: Applications in currency arbitrage, resource scheduling, and network routing

All tests achieve >95% code coverage and are automatically executed across multiple Python versions (3.10-3.12) and operating systems (Linux, Windows, macOS).

# Applications and Impact

The library has been designed to address real-world optimization problems across multiple domains:

**Financial Markets**: Currency arbitrage detection in foreign exchange markets, where negative cycles in exchange rate graphs indicate profitable trading opportunities.

**Operations Research**: Resource scheduling optimization in manufacturing, where the cost-to-time ratio represents operational efficiency metrics.

**Network Engineering**: Routing protocol optimization, where cycles in network topology graphs can indicate suboptimal routing configurations.

**Supply Chain Management**: Logistics optimization for identifying cost-effective transportation cycles in distribution networks.

The library's exact arithmetic capabilities make it particularly valuable in financial applications where floating-point precision loss can have significant economic consequences.

# Comparison with Existing Tools

Unlike general-purpose graph libraries such as NetworkX [@hagberg2008exploring] or specialized optimization frameworks like OR-Tools [@perron2011operations], `min-ratio-cycle` is specifically optimized for the minimum ratio cycle problem. This specialization enables significant performance improvements:

- **10-100x faster** than NetworkX's general cycle detection algorithms
- **Exact arithmetic support** unavailable in general-purpose libraries
- **Domain-specific validation** tailored to cost-time ratio constraints
- **Built-in analytics** for sensitivity analysis and performance monitoring

The library complements rather than competes with these tools, providing specialized functionality that can be integrated into larger optimization workflows.

# Future Development

The project follows semantic versioning and maintains backward compatibility. Planned enhancements include support for additional graph formats (GraphML, GML), integration with popular data science workflows (pandas, scikit-learn), and expanded domain-specific utilities for common application areas.

The open-source development model encourages community contributions, with comprehensive documentation for contributors and automated testing infrastructure to ensure code quality.

# Conclusion

`min-ratio-cycle` provides the research and industrial communities with a robust, high-performance solution for minimum cost-to-time ratio cycle detection. By combining state-of-the-art algorithms with practical software engineering, the library enables researchers and practitioners to solve previously intractable optimization problems efficiently and reliably.

The library's emphasis on correctness, performance, and usability makes it suitable for both academic research and production applications, bridging the gap between theoretical algorithms and practical implementation.

# Acknowledgements

We acknowledge the theoretical foundations provided by Bringmann, Hansen, and Krinninger [@bringmann2017improved], whose work on improved algorithms for minimum ratio cycles directly inspired this implementation. We also thank the NumPy [@harris2020array] and NetworkX [@hagberg2008exploring] development teams for providing the foundational libraries that make high-performance scientific computing in Python possible.

# References
