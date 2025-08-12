"""
Comprehensive benchmark suite for MinRatioCycleSolver.

This script provides detailed performance analysis including:
- Scaling behavior analysis
- Comparison with theoretical complexity
- Memory usage profiling
- Different graph topology impacts
"""

import time
import sys
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from min_ratio_cycle.solver import MinRatioCycleSolver


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run."""
    n_vertices: int
    n_edges: int
    density: float
    solve_time: float
    memory_peak: int  # bytes
    ratio: float
    cycle_length: int
    mode: str  # 'exact' or 'numeric'
    success: bool
    error_msg: Optional[str] = None


class GraphGenerator:
    """Generate various types of test graphs."""
    
    @staticmethod
    def random_graph(n: int, density: float, seed: int = None) -> MinRatioCycleSolver:
        """Generate random graph with given density."""
        if seed is not None:
            np.random.seed(seed)
        
        solver = MinRatioCycleSolver(n)
        num_edges = int(n * n * density)
        
        for _ in range(num_edges):
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            cost = np.random.randint(-10, 11)
            time = np.random.randint(1, 6)
            solver.add_edge(u, v, cost, time)
        
        return solver
    
    @staticmethod
    def complete_graph(n: int, weight_range: Tuple[int, int] = (-5, 5)) -> MinRatioCycleSolver:
        """Generate complete graph with random weights."""
        solver = MinRatioCycleSolver(n)
        
        for u in range(n):
            for v in range(n):
                if u != v:
                    cost = np.random.randint(weight_range[0], weight_range[1] + 1)
                    time = np.random.randint(1, 6)
                    solver.add_edge(u, v, cost, time)
        
        return solver
    
    @staticmethod
    def cycle_graph(n: int) -> MinRatioCycleSolver:
        """Generate simple cycle graph 0->1->...->n-1->0."""
        solver = MinRatioCycleSolver(n)
        
        for i in range(n):
            next_vertex = (i + 1) % n
            cost = np.random.randint(-5, 6)
            time = np.random.randint(1, 4)
            solver.add_edge(i, next_vertex, cost, time)
        
        return solver
    
    @staticmethod
    def grid_graph(rows: int, cols: int) -> MinRatioCycleSolver:
        """Generate grid graph with random weights."""
        n = rows * cols
        solver = MinRatioCycleSolver(n)
        
        def vertex_id(r: int, c: int) -> int:
            return r * cols + c
        
        for r in range(rows):
            for c in range(cols):
                u = vertex_id(r, c)
                
                # Add edges to neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        v = vertex_id(nr, nc)
                        cost = np.random.randint(-3, 4)
                        time = np.random.randint(1, 3)
                        solver.add_edge(u, v, cost, time)
        
        return solver


class BenchmarkRunner:
    """Run and collect benchmark results."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_single_benchmark(self, solver: MinRatioCycleSolver, 
                           graph_type: str, **kwargs) -> BenchmarkResult:
        """Run a single benchmark and collect metrics."""
        # Count edges and vertices
        solver._build_numpy_arrays_once()
        n_vertices = solver.n
        n_edges = len(solver._edges)
        density = n_edges / (n_vertices * n_vertices) if n_vertices > 0 else 0
        
        # Determine mode (integer vs float)
        mode = 'exact' if solver._all_int else 'numeric'
        
        # Run with memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            cycle, sum_cost, sum_time, ratio = solver.solve()
            solve_time = time.perf_counter() - start_time
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result = BenchmarkResult(
                n_vertices=n_vertices,
                n_edges=n_edges,
                density=density,
                solve_time=solve_time,
                memory_peak=peak,
                ratio=ratio,
                cycle_length=len(cycle) - 1,  # exclude closing vertex
                mode=mode,
                success=True
            )
            
        except Exception as e:
            solve_time = time.perf_counter() - start_time
            tracemalloc.stop()
            
            result = BenchmarkResult(
                n_vertices=n_vertices,
                n_edges=n_edges,
                density=density,
                solve_time=solve_time,
                memory_peak=0,
                ratio=float('inf'),
                cycle_length=0,
                mode=mode,
                success=False,
                error_msg=str(e)
            )
        
        self.results.append(result)
        return result
    
    def scaling_benchmark(self, max_vertices: int = 100, step: int = 10):
        """Test how performance scales with graph size."""
        print("Running scaling benchmark...")
        
        sizes = list(range(10, max_vertices + 1, step))
        
        for n in sizes:
            print(f"  Testing n={n}...")
            
            # Test different graph types
            for graph_type, generator in [
                ('random_sparse', lambda: GraphGenerator.random_graph(n, 0.1)),
                ('random_dense', lambda: GraphGenerator.random_graph(n, 0.3)),
                ('complete', lambda: GraphGenerator.complete_graph(min(n, 15))),  # Cap complete graphs
            ]:
                if graph_type == 'complete' and n > 15:
                    continue  # Skip large complete graphs
                
                try:
                    solver = generator()
                    result = self.run_single_benchmark(solver, graph_type)
                    
                    if result.success:
                        print(f"    {graph_type}: {result.solve_time:.4f}s, "
                              f"ratio={result.ratio:.4f}")
                    else:
                        print(f"    {graph_type}: FAILED - {result.error_msg}")
                        
                except Exception as e:
                    print(f"    {graph_type}: ERROR - {e}")
    
    def density_benchmark(self, n_vertices: int = 50):
        """Test how edge density affects performance."""
        print(f"\nRunning density benchmark (n={n_vertices})...")
        
        densities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        
        for density in densities:
            print(f"  Testing density={density:.2f}...")
            
            solver = GraphGenerator.random_graph(n_vertices, density)
            result = self.run_single_benchmark(solver, 'random')
            
            if result.success:
                print(f"    Time: {result.solve_time:.4f}s, "
                      f"Memory: {result.memory_peak/1024:.1f}KB")
            else:
                print(f"    FAILED - {result.error_msg}")
    
    def mode_comparison_benchmark(self):
        """Compare exact vs numeric modes."""
        print("\nRunning mode comparison benchmark...")
        
        sizes = [10, 20, 30, 40, 50]
        
        for n in sizes:
            print(f"  Testing n={n}...")
            
            # Create identical graph structure for both modes
            np.random.seed(42)  # Ensure reproducibility
            edges = []
            for _ in range(n * 2):  # Moderate density
                u = np.random.randint(0, n)
                v = np.random.randint(0, n)
                cost = np.random.randint(-5, 6)
                time = np.random.randint(1, 4)
                edges.append((u, v, cost, time))
            
            # Test exact mode (integer weights)
            solver_exact = MinRatioCycleSolver(n)
            for u, v, c, t in edges:
                solver_exact.add_edge(u, v, c, t)
            
            result_exact = self.run_single_benchmark(solver_exact, 'exact')
            
            # Test numeric mode (float weights)
            solver_numeric = MinRatioCycleSolver(n)
            for u, v, c, t in edges:
                solver_numeric.add_edge(u, v, float(c) + 0.1, float(t))  # Add small float
            
            result_numeric = self.run_single_benchmark(solver_numeric, 'numeric')
            
            if result_exact.success and result_numeric.success:
                ratio_diff = abs(result_exact.ratio - result_numeric.ratio)
                print(f"    Exact:   {result_exact.solve_time:.4f}s, ratio={result_exact.ratio:.6f}")
                print(f"    Numeric: {result_numeric.solve_time:.4f}s, ratio={result_numeric.ratio:.6f}")
                print(f"    Ratio difference: {ratio_diff:.8f}")
            else:
                exact_status = "OK" if result_exact.success else "FAILED"
                numeric_status = "OK" if result_numeric.success else "FAILED"
                print(f"    Exact: {exact_status}, Numeric: {numeric_status}")
    
    def topology_benchmark(self):
        """Test different graph topologies."""
        print("\nRunning topology benchmark...")
        
        n = 30  # Fixed size for comparison
        
        topologies = [
            ('cycle', lambda: GraphGenerator.cycle_graph(n)),
            ('grid_5x6', lambda: GraphGenerator.grid_graph(5, 6)),
            ('random_sparse', lambda: GraphGenerator.random_graph(n, 0.1)),
            ('random_dense', lambda: GraphGenerator.random_graph(n, 0.4)),
        ]
        
        for topo_name, generator in topologies:
            print(f"  Testing {topo_name}...")
            
            try:
                solver = generator()
                result = self.run_single_benchmark(solver, topo_name)
                
                if result.success:
                    print(f"    Time: {result.solve_time:.4f}s, "
                          f"Edges: {result.n_edges}, "
                          f"Ratio: {result.ratio:.4f}")
                else:
                    print(f"    FAILED - {result.error_msg}")
                    
            except Exception as e:
                print(f"    ERROR - {e}")
    
    def generate_report(self, save_plots: bool = True):
        """Generate comprehensive benchmark report."""
        if not self.results:
            print("No benchmark results to report!")
            return
        
        print(f"\n{'='*60}")
        print("BENCHMARK REPORT")
        print(f"{'='*60}")
        
        # Success rate
        successful = [r for r in self.results if r.success]
        success_rate = len(successful) / len(self.results) * 100
        print(f"Success rate: {success_rate:.1f}% ({len(successful)}/{len(self.results)})")
        
        if not successful:
            print("No successful runs to analyze!")
            return
        
        # Performance statistics
        times = [r.solve_time for r in successful]
        print(f"\nSolve times:")
        print(f"  Mean: {np.mean(times):.4f}s")
        print(f"  Median: {np.median(times):.4f}s")
        print(f"  Min: {np.min(times):.4f}s")
        print(f"  Max: {np.max(times):.4f}s")
        print(f"  Std: {np.std(times):.4f}s")
        
        # Memory usage
        memories = [r.memory_peak for r in successful]
        print(f"\nMemory usage:")
        print(f"  Mean: {np.mean(memories)/1024:.1f}KB")
        print(f"  Max: {np.max(memories)/1024:.1f}KB")
        
        # Mode comparison
        exact_results = [r for r in successful if r.mode == 'exact']
        numeric_results = [r for r in successful if r.mode == 'numeric']
        
        if exact_results and numeric_results:
            exact_times = [r.solve_time for r in exact_results]
            numeric_times = [r.solve_time for r in numeric_results]
            
            print(f"\nMode comparison:")
            print(f"  Exact mode:   {len(exact_results)} runs, mean time: {np.mean(exact_times):.4f}s")
            print(f"  Numeric mode: {len(numeric_results)} runs, mean time: {np.mean(numeric_times):.4f}s")
        
        # Generate plots if requested
        if save_plots:
            self._generate_plots()
    
    def _generate_plots(self):
        """Generate performance visualization plots."""
        successful = [r for r in self.results if r.success]
        if len(successful) < 5:
            print("Not enough data points for meaningful plots")
            return
        
        # Scaling plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Time vs vertices
        plt.subplot(2, 2, 1)
        vertices = [r.n_vertices for r in successful]
        times = [r.solve_time for r in successful]
        plt.scatter(vertices, times, alpha=0.6)
        plt.xlabel('Number of vertices')
        plt.ylabel('Solve time (s)')
        plt.title('Scaling: Time vs Graph Size')
        plt.yscale('log')
        
        # Plot 2: Time vs edges
        plt.subplot(2, 2, 2)
        edges = [r.n_edges for r in successful]
        plt.scatter(edges, times, alpha=0.6, color='orange')
        plt.xlabel('Number of edges')
        plt.ylabel('Solve time (s)')
        plt.title('Scaling: Time vs Edge Count')
        plt.yscale('log')
        
        # Plot 3: Memory vs vertices
        plt.subplot(2, 2, 3)
        memories = [r.memory_peak/1024 for r in successful]  # Convert to KB
        plt.scatter(vertices, memories, alpha=0.6, color='green')
        plt.xlabel('Number of vertices')
        plt.ylabel('Peak memory (KB)')
        plt.title('Memory Usage vs Graph Size')
        
        # Plot 4: Mode comparison
        plt.subplot(2, 2, 4)
        exact_results = [r for r in successful if r.mode == 'exact']
        numeric_results = [r for r in successful if r.mode == 'numeric']
        
        if exact_results and numeric_results:
            exact_times = [r.solve_time for r in exact_results]
            numeric_times = [r.solve_time for r in numeric_results]
            
            plt.boxplot([exact_times, numeric_times], labels=['Exact', 'Numeric'])
            plt.ylabel('Solve time (s)')
            plt.title('Mode Comparison')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
        print("Plots saved to 'benchmark_results.png'")
        plt.show()


class StressTest:
    """Stress testing for edge cases and robustness."""
    
    def __init__(self):
        self.failed_cases = []
    
    def test_large_weights(self):
        """Test with very large integer weights."""
        print("Testing large weights...")
        
        solver = MinRatioCycleSolver(5)
        large_val = 10**12
        
        # Create cycle with extreme weights
        solver.add_edge(0, 1, large_val, 1)
        solver.add_edge(1, 2, 1, large_val)
        solver.add_edge(2, 3, -large_val//2, 1)
        solver.add_edge(3, 4, 1, 1)
        solver.add_edge(4, 0, 1, 1)
        
        try:
            cycle, cost, time, ratio = solver.solve()
            print(f"  Large weights test: SUCCESS, ratio={ratio:.6e}")
        except Exception as e:
            print(f"  Large weights test: FAILED - {e}")
            self.failed_cases.append(("large_weights", e))
    
    def test_precision_edge_cases(self):
        """Test numerical precision edge cases."""
        print("Testing precision edge cases...")
        
        # Very small differences
        solver = MinRatioCycleSolver(3)
        solver.add_edge(0, 1, 1.0000001, 1.0)
        solver.add_edge(1, 2, 1.0, 1.0000001)
        solver.add_edge(2, 0, 1.0, 1.0)
        
        try:
            cycle, cost, time, ratio = solver.solve()
            print(f"  Precision test: SUCCESS, ratio={ratio:.10f}")
        except Exception as e:
            print(f"  Precision test: FAILED - {e}")
            self.failed_cases.append(("precision", e))
    
    def test_pathological_graphs(self):
        """Test graphs designed to stress the algorithm."""
        print("Testing pathological graphs...")
        
        # Long chain with cycle at the end
        n = 100
        solver = MinRatioCycleSolver(n)
        
        # Chain: 0->1->2->...->97
        for i in range(n-3):
            solver.add_edge(i, i+1, 1, 1)
        
        # Cycle at end: 97->98->99->97
        solver.add_edge(n-3, n-2, 1, 2)
        solver.add_edge(n-2, n-1, 1, 2)
        solver.add_edge(n-1, n-3, -1, 1)  # Negative cost for better ratio
        
        try:
            start_time = time.time()
            cycle, cost, time_sum, ratio = solver.solve()
            elapsed = time.time() - start_time
            print(f"  Pathological graph: SUCCESS in {elapsed:.4f}s, ratio={ratio:.6f}")
        except Exception as e:
            print(f"  Pathological graph: FAILED - {e}")
            self.failed_cases.append(("pathological", e))
    
    def run_all_stress_tests(self):
        """Run all stress tests."""
        print("Running stress tests...")
        self.test_large_weights()
        self.test_precision_edge_cases()
        self.test_pathological_graphs()
        
        if self.failed_cases:
            print(f"\nStress test failures: {len(self.failed_cases)}")
            for name, error in self.failed_cases:
                print(f"  {name}: {error}")
        else:
            print("\nAll stress tests passed!")


def main():
    """Run the complete benchmark suite."""
    print("Min Ratio Cycle Solver - Comprehensive Benchmark Suite")
    print("=" * 60)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Run different benchmark categories
    try:
        runner.scaling_benchmark(max_vertices=80, step=10)
        runner.density_benchmark(n_vertices=40)
        runner.mode_comparison_benchmark()
        runner.topology_benchmark()
        
        # Generate comprehensive report
        runner.generate_report(save_plots=True)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Run stress tests
    print("\n" + "=" * 60)
    stress_tester = StressTest()
    stress_tester.run_all_stress_tests()
    
    print("\nBenchmark suite completed!")


if __name__ == "__main__":
    main()