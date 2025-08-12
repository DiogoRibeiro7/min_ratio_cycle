"""
Monitoring and observability components for min-ratio-cycle solver.

This module provides comprehensive monitoring capabilities including:
- Performance metrics collection and analysis
- Structured logging with configurable levels
- Performance profiling and bottleneck detection
- System health checks and diagnostics
- Progress tracking for long-running operations

Components:
- SolverMetrics: Performance metrics collection
- SolverLogger: Structured logging
- SolverProfiler: Performance profiling
- SolverHealthCheck: System health diagnostics
- progress_tracker: Progress tracking context manager
"""

from .health import SolverHealthCheck, run_health_check
from .logging import SolverLogger, setup_logging
from .metrics import MetricsCollector, PerformanceAnalyzer, SolverMetrics
from .profiler import SolverProfiler, profile_operation
from .progress import ProgressTracker, progress_tracker

__all__ = [
    # Core monitoring classes
    "SolverMetrics",
    "SolverLogger",
    "SolverProfiler",
    "SolverHealthCheck",
    # Utility classes
    "MetricsCollector",
    "PerformanceAnalyzer",
    "ProgressTracker",
    # Convenience functions
    "setup_logging",
    "profile_operation",
    "run_health_check",
    "progress_tracker",
]

# Version info
__version__ = "0.2.0"
__author__ = "Diogo Ribeiro"
