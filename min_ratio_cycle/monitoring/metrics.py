"""
Performance metrics collection and analysis for min-ratio-cycle solver.

This module provides comprehensive metrics collection including:
- Solve time tracking
- Memory usage monitoring
- Success/failure rates
- Algorithm performance analysis
- Comparative benchmarking
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class SolveMetrics:
    """Metrics for a single solve operation."""

    # Basic metrics
    solve_time: float
    success: bool
    mode_used: str

    # Graph properties
    n_vertices: int
    n_edges: int
    graph_density: float

    # Algorithm metrics
    iterations: int = 0
    preprocessing_time: float = 0.0

    # Result metrics
    ratio_found: Optional[float] = None
    cycle_length: int = 0

    # Resource usage
    memory_peak: Optional[int] = None
    cpu_time: Optional[float] = None

    # Error information
    error_message: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    solver_version: str = "0.2.0"


class SolverMetrics:
    """
    Comprehensive metrics collection for solver performance analysis.

    This class collects and analyzes performance metrics across multiple
    solver runs, providing insights into performance patterns, bottlenecks,
    and optimization opportunities.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of solve records to keep in memory
        """
        self.max_history = max_history
        self._solve_records: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

        # Aggregated statistics
        self._stats_cache: Dict[str, Any] = {}
        self._cache_valid = False

    def record_solve(
        self,
        solver: Any,
        solve_time: float,
        success: bool,
        mode: str,
        iterations: int = 0,
        preprocessing_time: float = 0.0,
        ratio_found: Optional[float] = None,
        cycle_length: int = 0,
        memory_peak: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Record metrics from a solve operation.

        Args:
            solver: The solver instance
            solve_time: Total time taken to solve
            success: Whether solve was successful
            mode: Solver mode used ('exact', 'numeric', 'approximate')
            iterations: Number of algorithm iterations
            preprocessing_time: Time spent in preprocessing
            ratio_found: The optimal ratio found (if successful)
            cycle_length: Length of the optimal cycle found
            memory_peak: Peak memory usage in bytes
            error_message: Error message if solve failed
        """
        with self._lock:
            # Get graph properties
            n_vertices = solver.n
            n_edges = len(getattr(solver, "_edges", []))
            graph_density = (
                n_edges / (n_vertices * n_vertices) if n_vertices > 0 else 0.0
            )

            # Get CPU time if available
            cpu_time = None
            if HAS_PSUTIL:
                try:
                    process = psutil.Process()
                    cpu_time = process.cpu_times().user + process.cpu_times().system
                except Exception:
                    cpu_time = None

            # Create solve record
            record = SolveMetrics(
                solve_time=solve_time,
                success=success,
                mode_used=mode,
                n_vertices=n_vertices,
                n_edges=n_edges,
                graph_density=graph_density,
                iterations=iterations,
                preprocessing_time=preprocessing_time,
                ratio_found=ratio_found,
                cycle_length=cycle_length,
                memory_peak=memory_peak,
                cpu_time=cpu_time,
                error_message=error_message,
            )

            self._solve_records.append(record)
            self._cache_valid = False  # Invalidate statistics cache

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all recorded solves.

        Returns:
            Dictionary with comprehensive summary statistics
        """
        if self._cache_valid and self._stats_cache:
            return self._stats_cache.copy()

        with self._lock:
            if not self._solve_records:
                return {"error": "No solve records available"}

            records = list(self._solve_records)
            successful_records = [r for r in records if r.success]
            failed_records = [r for r in records if not r.success]

        if not HAS_NUMPY:
            # Fallback calculations without numpy
            return self._calculate_basic_stats(
                records, successful_records, failed_records
            )

        # Advanced statistics with numpy
        stats = self._calculate_advanced_stats(
            records, successful_records, failed_records
        )

        # Cache the results
        self._stats_cache = stats
        self._cache_valid = True

        return stats.copy()

    def _calculate_basic_stats(
        self,
        records: List[SolveMetrics],
        successful: List[SolveMetrics],
        failed: List[SolveMetrics],
    ) -> Dict[str, Any]:
        """Calculate basic statistics without numpy dependency."""
        total_records = len(records)

        if not records:
            return {"error": "No records to analyze"}

        # Basic counts
        success_count = len(successful)
        failure_count = len(failed)
        success_rate = success_count / total_records if total_records > 0 else 0.0

        # Mode distribution
        mode_counts = defaultdict(int)
        for record in records:
            mode_counts[record.mode_used] += 1

        # Time statistics (successful solves only)
        if successful:
            solve_times = [r.solve_time for r in successful]
            min_time = min(solve_times)
            max_time = max(solve_times)
            avg_time = sum(solve_times) / len(solve_times)

            # Simple median calculation
            sorted_times = sorted(solve_times)
            n = len(sorted_times)
            median_time = (sorted_times[n // 2] + sorted_times[(n - 1) // 2]) / 2
        else:
            min_time = max_time = avg_time = median_time = 0.0

        return {
            "total_solves": total_records,
            "successful_solves": success_count,
            "failed_solves": failure_count,
            "success_rate": success_rate,
            "mode_distribution": dict(mode_counts),
            "timing": {
                "avg_solve_time": avg_time,
                "min_solve_time": min_time,
                "max_solve_time": max_time,
                "median_solve_time": median_time,
            },
            "graph_sizes": {
                "avg_vertices": sum(r.n_vertices for r in records) / total_records,
                "avg_edges": sum(r.n_edges for r in records) / total_records,
                "avg_density": sum(r.graph_density for r in records) / total_records,
            },
        }

    def _calculate_advanced_stats(
        self,
        records: List[SolveMetrics],
        successful: List[SolveMetrics],
        failed: List[SolveMetrics],
    ) -> Dict[str, Any]:
        """Calculate advanced statistics with numpy."""
        import numpy as np

        total_records = len(records)
        success_count = len(successful)
        failure_count = len(failed)

        # Basic stats
        success_rate = success_count / total_records if total_records > 0 else 0.0

        # Mode distribution
        mode_counts = defaultdict(int)
        for record in records:
            mode_counts[record.mode_used] += 1

        # Timing statistics
        timing_stats = {}
        if successful:
            solve_times = np.array([r.solve_time for r in successful])
            timing_stats = {
                "avg_solve_time": float(np.mean(solve_times)),
                "median_solve_time": float(np.median(solve_times)),
                "std_solve_time": float(np.std(solve_times)),
                "min_solve_time": float(np.min(solve_times)),
                "max_solve_time": float(np.max(solve_times)),
                "percentile_95": float(np.percentile(solve_times, 95)),
                "percentile_99": float(np.percentile(solve_times, 99)),
            }

        # Graph size statistics
        vertices = np.array([r.n_vertices for r in records])
        edges = np.array([r.n_edges for r in records])
        densities = np.array([r.graph_density for r in records])

        graph_stats = {
            "vertices": {
                "avg": float(np.mean(vertices)),
                "median": float(np.median(vertices)),
                "min": int(np.min(vertices)),
                "max": int(np.max(vertices)),
                "std": float(np.std(vertices)),
            },
            "edges": {
                "avg": float(np.mean(edges)),
                "median": float(np.median(edges)),
                "min": int(np.min(edges)),
                "max": int(np.max(edges)),
                "std": float(np.std(edges)),
            },
            "density": {
                "avg": float(np.mean(densities)),
                "median": float(np.median(densities)),
                "min": float(np.min(densities)),
                "max": float(np.max(densities)),
                "std": float(np.std(densities)),
            },
        }

        # Performance by mode
        performance_by_mode = {}
        for mode in mode_counts:
            mode_records = [r for r in successful if r.mode_used == mode]
            if mode_records:
                mode_times = np.array([r.solve_time for r in mode_records])
                performance_by_mode[mode] = {
                    "count": len(mode_records),
                    "avg_time": float(np.mean(mode_times)),
                    "median_time": float(np.median(mode_times)),
                    "success_rate": len(mode_records) / mode_counts[mode],
                }

        # Iterations analysis
        iterations_data = [r.iterations for r in successful if r.iterations > 0]
        iterations_stats = {}
        if iterations_data:
            iterations_array = np.array(iterations_data)
            iterations_stats = {
                "avg_iterations": float(np.mean(iterations_array)),
                "median_iterations": float(np.median(iterations_array)),
                "max_iterations": int(np.max(iterations_array)),
                "min_iterations": int(np.min(iterations_array)),
            }

        # Memory usage analysis
        memory_stats = {}
        memory_data = [r.memory_peak for r in records if r.memory_peak is not None]
        if memory_data:
            memory_array = np.array(memory_data)
            memory_stats = {
                "avg_memory_mb": float(np.mean(memory_array)) / (1024 * 1024),
                "max_memory_mb": float(np.max(memory_array)) / (1024 * 1024),
                "min_memory_mb": float(np.min(memory_array)) / (1024 * 1024),
            }

        # Error analysis
        error_stats = {}
        if failed:
            error_types = defaultdict(int)
            for record in failed:
                if record.error_message:
                    # Extract error type from message
                    error_type = (
                        record.error_message.split(":")[0]
                        if ":" in record.error_message
                        else "Unknown"
                    )
                    error_types[error_type] += 1
            error_stats = dict(error_types)

        return {
            "total_solves": total_records,
            "successful_solves": success_count,
            "failed_solves": failure_count,
            "success_rate": success_rate,
            "mode_distribution": dict(mode_counts),
            "timing": timing_stats,
            "graph_statistics": graph_stats,
            "performance_by_mode": performance_by_mode,
            "iterations": iterations_stats,
            "memory_usage": memory_stats,
            "error_analysis": error_stats,
            "last_updated": time.time(),
        }

    def get_performance_trends(self, window_size: int = 100) -> Dict[str, List[float]]:
        """
        Get performance trends over time using moving averages.

        Args:
            window_size: Size of the moving average window

        Returns:
            Dictionary with trending metrics
        """
        with self._lock:
            if len(self._solve_records) < window_size:
                return {
                    "error": f"Insufficient data (need at least {window_size} records)"
                }

            records = list(self._solve_records)

        trends = {
            "solve_times": [],
            "success_rates": [],
            "iterations": [],
            "timestamps": [],
        }

        for i in range(window_size, len(records) + 1):
            window = records[i - window_size : i]

            # Calculate metrics for this window
            successful = [r for r in window if r.success]

            if successful:
                avg_time = sum(r.solve_time for r in successful) / len(successful)
                trends["solve_times"].append(avg_time)
            else:
                trends["solve_times"].append(0.0)

            success_rate = len(successful) / len(window)
            trends["success_rates"].append(success_rate)

            iterations_data = [r.iterations for r in successful if r.iterations > 0]
            avg_iterations = (
                sum(iterations_data) / len(iterations_data) if iterations_data else 0.0
            )
            trends["iterations"].append(avg_iterations)

            trends["timestamps"].append(window[-1].timestamp)

        return trends

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Export all metrics to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Convert records to serializable format
            export_data = {
                "metadata": {
                    "export_timestamp": time.time(),
                    "total_records": len(self._solve_records),
                    "max_history": self.max_history,
                },
                "summary": self.get_summary_stats(),
                "records": [],
            }

            for record in self._solve_records:
                record_dict = {
                    "solve_time": record.solve_time,
                    "success": record.success,
                    "mode_used": record.mode_used,
                    "n_vertices": record.n_vertices,
                    "n_edges": record.n_edges,
                    "graph_density": record.graph_density,
                    "iterations": record.iterations,
                    "preprocessing_time": record.preprocessing_time,
                    "ratio_found": record.ratio_found,
                    "cycle_length": record.cycle_length,
                    "memory_peak": record.memory_peak,
                    "cpu_time": record.cpu_time,
                    "error_message": record.error_message,
                    "timestamp": record.timestamp,
                    "solver_version": record.solver_version,
                }
                export_data["records"].append(record_dict)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, sort_keys=True)

    def import_from_json(self, filepath: Union[str, Path]) -> None:
        """
        Import metrics from JSON file.

        Args:
            filepath: Path to the JSON file to import
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        with self._lock:
            # Import records
            for record_dict in data.get("records", []):
                record = SolveMetrics(
                    solve_time=record_dict["solve_time"],
                    success=record_dict["success"],
                    mode_used=record_dict["mode_used"],
                    n_vertices=record_dict["n_vertices"],
                    n_edges=record_dict["n_edges"],
                    graph_density=record_dict["graph_density"],
                    iterations=record_dict.get("iterations", 0),
                    preprocessing_time=record_dict.get("preprocessing_time", 0.0),
                    ratio_found=record_dict.get("ratio_found"),
                    cycle_length=record_dict.get("cycle_length", 0),
                    memory_peak=record_dict.get("memory_peak"),
                    cpu_time=record_dict.get("cpu_time"),
                    error_message=record_dict.get("error_message"),
                    timestamp=record_dict.get("timestamp", time.time()),
                    solver_version=record_dict.get("solver_version", "0.2.0"),
                )
                self._solve_records.append(record)

            self._cache_valid = False  # Invalidate cache

    def clear_history(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._solve_records.clear()
            self._stats_cache.clear()
            self._cache_valid = False

    def get_record_count(self) -> int:
        """Get the number of recorded solve operations."""
        return len(self._solve_records)

    def get_recent_records(self, n: int = 10) -> List[SolveMetrics]:
        """
        Get the n most recent solve records.

        Args:
            n: Number of recent records to return

        Returns:
            List of recent SolveMetrics records
        """
        with self._lock:
            return list(self._solve_records)[-n:]


class MetricsCollector:
    """
    Global metrics collector singleton for easy access across the application.

    This class provides a global instance for collecting metrics without
    needing to pass around metrics objects.
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            self.metrics = SolverMetrics()
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get the global metrics collector instance."""
        return cls()

    def record(self, *args, **kwargs) -> None:
        """Record a solve operation (delegates to SolverMetrics.record_solve)."""
        self.metrics.record_solve(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics (delegates to SolverMetrics.get_summary_stats)."""
        return self.metrics.get_summary_stats()

    def export(self, filepath: Union[str, Path]) -> None:
        """Export metrics to file (delegates to SolverMetrics.export_to_json)."""
        self.metrics.export_to_json(filepath)


class PerformanceAnalyzer:
    """
    Advanced performance analysis utilities for solver metrics.

    Provides sophisticated analysis capabilities including regression
    detection, performance comparison, and bottleneck identification.
    """

    def __init__(self, metrics: SolverMetrics):
        self.metrics = metrics

    def detect_performance_regression(
        self,
        recent_window: int = 50,
        baseline_window: int = 100,
        threshold: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Detect performance regressions by comparing recent vs baseline performance.

        Args:
            recent_window: Size of recent performance window
            baseline_window: Size of baseline performance window
            threshold: Performance degradation threshold (1.2 = 20% worse)

        Returns:
            Regression analysis results
        """
        with self.metrics._lock:
            records = list(self.metrics._solve_records)

        if len(records) < recent_window + baseline_window:
            return {"error": "Insufficient data for regression analysis"}

        # Split into baseline and recent periods
        baseline_records = records[-(baseline_window + recent_window) : -recent_window]
        recent_records = records[-recent_window:]

        # Filter successful solves
        baseline_successful = [r for r in baseline_records if r.success]
        recent_successful = [r for r in recent_records if r.success]

        if not baseline_successful or not recent_successful:
            return {"error": "Insufficient successful solves for comparison"}

        # Calculate average performance metrics
        baseline_avg_time = sum(r.solve_time for r in baseline_successful) / len(
            baseline_successful
        )
        recent_avg_time = sum(r.solve_time for r in recent_successful) / len(
            recent_successful
        )

        baseline_success_rate = len(baseline_successful) / len(baseline_records)
        recent_success_rate = len(recent_successful) / len(recent_records)

        # Check for regressions
        time_regression = recent_avg_time > baseline_avg_time * threshold
        success_regression = recent_success_rate < baseline_success_rate * (
            2 - threshold
        )  # Inverse threshold

        return {
            "time_regression_detected": time_regression,
            "success_regression_detected": success_regression,
            "baseline_avg_time": baseline_avg_time,
            "recent_avg_time": recent_avg_time,
            "time_degradation_factor": recent_avg_time / baseline_avg_time
            if baseline_avg_time > 0
            else 0,
            "baseline_success_rate": baseline_success_rate,
            "recent_success_rate": recent_success_rate,
            "success_degradation_factor": recent_success_rate / baseline_success_rate
            if baseline_success_rate > 0
            else 0,
            "analysis_timestamp": time.time(),
        }

    def compare_modes(self) -> Dict[str, Any]:
        """
        Compare performance across different solver modes.

        Returns:
            Comparative analysis of solver modes
        """
        stats = self.metrics.get_summary_stats()

        if "performance_by_mode" not in stats:
            return {"error": "No mode performance data available"}

        mode_performance = stats["performance_by_mode"]

        if len(mode_performance) < 2:
            return {"error": "Need at least 2 modes for comparison"}

        # Find best and worst performing modes
        modes_by_time = sorted(mode_performance.items(), key=lambda x: x[1]["avg_time"])
        modes_by_success = sorted(
            mode_performance.items(), key=lambda x: x[1]["success_rate"], reverse=True
        )

        return {
            "fastest_mode": modes_by_time[0][0] if modes_by_time else None,
            "slowest_mode": modes_by_time[-1][0] if modes_by_time else None,
            "most_reliable_mode": modes_by_success[0][0] if modes_by_success else None,
            "least_reliable_mode": modes_by_success[-1][0]
            if modes_by_success
            else None,
            "mode_performance": mode_performance,
            "speed_improvement_factor": (
                modes_by_time[-1][1]["avg_time"] / modes_by_time[0][1]["avg_time"]
            )
            if len(modes_by_time) >= 2
            else 1.0,
        }

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify performance bottlenecks based on recorded metrics.

        Returns:
            Bottleneck analysis results
        """
        with self.metrics._lock:
            records = [r for r in self.metrics._solve_records if r.success]

        if not records:
            return {"error": "No successful solves to analyze"}

        bottlenecks = []

        # Check for preprocessing bottlenecks
        preprocessing_times = [
            r.preprocessing_time for r in records if r.preprocessing_time > 0
        ]
        if preprocessing_times:
            avg_preprocessing = sum(preprocessing_times) / len(preprocessing_times)
            avg_total = sum(r.solve_time for r in records) / len(records)

            if avg_preprocessing > avg_total * 0.3:  # More than 30% of total time
                bottlenecks.append(
                    {
                        "type": "preprocessing",
                        "description": "Preprocessing takes significant portion of solve time",
                        "avg_time": avg_preprocessing,
                        "percentage_of_total": (avg_preprocessing / avg_total) * 100,
                    }
                )

        # Check for iteration count issues
        iteration_data = [r.iterations for r in records if r.iterations > 0]
        if iteration_data and HAS_NUMPY:
            import numpy as np

            iterations_array = np.array(iteration_data)
            high_iteration_threshold = np.percentile(iterations_array, 95)

            high_iteration_records = [
                r for r in records if r.iterations > high_iteration_threshold
            ]
            if (
                len(high_iteration_records) > len(records) * 0.1
            ):  # More than 10% of solves
                bottlenecks.append(
                    {
                        "type": "high_iterations",
                        "description": "Some solves require excessive iterations",
                        "threshold": int(high_iteration_threshold),
                        "affected_solves": len(high_iteration_records),
                        "percentage_affected": (
                            len(high_iteration_records) / len(records)
                        )
                        * 100,
                    }
                )

        # Check for memory usage issues
        memory_data = [r.memory_peak for r in records if r.memory_peak is not None]
        if memory_data and HAS_NUMPY:
            import numpy as np

            memory_array = np.array(memory_data)
            high_memory_threshold = np.percentile(memory_array, 90)

            if high_memory_threshold > 1024 * 1024 * 1024:  # > 1GB
                bottlenecks.append(
                    {
                        "type": "high_memory",
                        "description": "High memory usage detected",
                        "threshold_mb": high_memory_threshold / (1024 * 1024),
                        "max_memory_mb": np.max(memory_array) / (1024 * 1024),
                    }
                )

        return {
            "bottlenecks_found": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "analysis_timestamp": time.time(),
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Min Ratio Cycle Solver - Metrics Module")
    print("=" * 40)

    # Create metrics collector
    metrics = SolverMetrics()

    # Mock solver object for testing
    class MockSolver:
        def __init__(self, n, edges):
            self.n = n
            self._edges = [None] * edges

    # Record some sample metrics
    for i in range(50):
        solver = MockSolver(10 + i % 20, 50 + i % 100)
        success = i % 7 != 0  # ~85% success rate
        mode = ["exact", "numeric", "approximate"][i % 3]

        metrics.record_solve(
            solver=solver,
            solve_time=0.1 + (i % 10) * 0.05,
            success=success,
            mode=mode,
            iterations=10 + i % 50,
            memory_peak=1024 * 1024 * (1 + i % 5) if i % 4 == 0 else None,
            error_message="Test error" if not success else None,
        )

    # Get summary statistics
    stats = metrics.get_summary_stats()
    print(f"Total solves: {stats['total_solves']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average solve time: {stats['timing']['avg_solve_time']:.4f}s")

    # Test performance analysis
    analyzer = PerformanceAnalyzer(metrics)

    # Test mode comparison
    mode_comparison = analyzer.compare_modes()
    if "fastest_mode" in mode_comparison:
        print(f"Fastest mode: {mode_comparison['fastest_mode']}")
        print(f"Most reliable mode: {mode_comparison['most_reliable_mode']}")

    # Test bottleneck detection
    bottlenecks = analyzer.identify_bottlenecks()
    print(f"Bottlenecks found: {bottlenecks['bottlenecks_found']}")

    print("Metrics module test completed successfully!")
