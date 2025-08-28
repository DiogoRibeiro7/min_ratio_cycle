"""
Custom exception classes for min-ratio-cycle solver.

This module defines a hierarchy of custom exceptions to provide clear
error handling and debugging information for different types of failures
in the solver.
"""

from typing import Any


class SolverError(Exception):
    """
    Base exception class for all solver-related errors.

    This is the parent class for all custom exceptions in the min-ratio-
    cycle solver package.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        component: str | None = None,
        suggested_fix: str | None = None,
        recovery_hint: str | None = None,
    ):
        """
        Initialize solver error with rich context.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error details.
            component: Name of the affected component or subsystem.
            suggested_fix: Short hint describing how to resolve the issue.
            recovery_hint: Guidance on how the solver might recover.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if component:
            self.details.setdefault("component", component)
        if suggested_fix:
            self.details.setdefault("suggested_fix", suggested_fix)
        if recovery_hint:
            self.details.setdefault("recovery_hint", recovery_hint)

    def __str__(self) -> str:
        """
        String representation of the error.
        """
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """
        Convert error to dictionary representation.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(SolverError):
    """
    Exception raised for input validation failures.

    This includes invalid graph parameters, edge weights, vertex
    indices, and other input validation issues.
    """

    def __init__(
        self,
        message: str,
        invalid_value: Any = None,
        expected_type: type | None = None,
        valid_range: tuple | None = None,
    ):
        """
        Initialize validation error.

        Args:
            message: Error description
            invalid_value: The value that failed validation
            expected_type: Expected type for the value
            valid_range: Valid range for numeric values
        """
        details = {}
        if invalid_value is not None:
            details["invalid_value"] = invalid_value
            details["invalid_type"] = type(invalid_value).__name__
        if expected_type is not None:
            details["expected_type"] = expected_type.__name__
        if valid_range is not None:
            details["valid_range"] = valid_range

        super().__init__(message, details)
        self.invalid_value = invalid_value
        self.expected_type = expected_type
        self.valid_range = valid_range


class GraphError(SolverError):
    """
    Exception raised for graph structure related errors.

    This includes issues like disconnected graphs, missing edges,
    invalid graph topology, and other structural problems.
    """

    def __init__(
        self,
        message: str,
        graph_properties: dict[str, Any] | None = None,
        suggested_fix: str | None = None,
    ):
        """
        Initialize graph error.

        Args:
            message: Error description
            graph_properties: Properties of the problematic graph
            suggested_fix: Suggested solution for the user
        """
        details = {}
        if graph_properties:
            details.update(graph_properties)
        if suggested_fix:
            details["suggested_fix"] = suggested_fix

        super().__init__(message, details)
        self.graph_properties = graph_properties
        self.suggested_fix = suggested_fix


class GraphStructureError(GraphError):
    """
    Graph is structurally invalid for the requested operation.
    """

    def __init__(
        self,
        message: str,
        graph_properties: dict[str, Any] | None = None,
        suggested_fix: str | None = None,
        recovery_hint: str | None = None,
    ):
        super().__init__(
            message,
            graph_properties=graph_properties,
            suggested_fix=suggested_fix,
        )
        if recovery_hint:
            self.details.setdefault("recovery_hint", recovery_hint)


class AlgorithmError(SolverError):
    """
    Exception raised for algorithm execution failures.

    This includes convergence failures, numerical instability, iteration
    limits exceeded, and other algorithmic issues.
    """

    def __init__(
        self,
        message: str,
        algorithm_name: str | None = None,
        iterations: int | None = None,
        convergence_info: dict[str, float] | None = None,
    ):
        """
        Initialize algorithm error.

        Args:
            message: Error description
            algorithm_name: Name of the algorithm that failed
            iterations: Number of iterations completed
            convergence_info: Information about convergence state
        """
        details = {}
        if algorithm_name:
            details["algorithm"] = algorithm_name
        if iterations is not None:
            details["iterations_completed"] = iterations
        if convergence_info:
            details.update(convergence_info)

        super().__init__(message, details)
        self.algorithm_name = algorithm_name
        self.iterations = iterations
        self.convergence_info = convergence_info


class ConfigurationError(SolverError):
    """
    Exception raised for configuration-related errors.

    This includes invalid configuration parameters, conflicting
    settings, missing required configuration, and file I/O errors.
    """

    def __init__(
        self,
        message: str,
        config_section: str | None = None,
        invalid_parameters: list[str] | None = None,
        config_file: str | None = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Error description
            config_section: Configuration section with the error
            invalid_parameters: List of invalid parameter names
            config_file: Path to configuration file if applicable
        """
        details = {}
        if config_section:
            details["config_section"] = config_section
        if invalid_parameters:
            details["invalid_parameters"] = invalid_parameters
        if config_file:
            details["config_file"] = config_file

        super().__init__(message, details)
        self.config_section = config_section
        self.invalid_parameters = invalid_parameters
        self.config_file = config_file


class MemoryError(SolverError):
    """
    Exception raised for memory-related issues.

    This includes out-of-memory conditions, memory limit exceeded, and
    other resource exhaustion problems.
    """

    def __init__(
        self,
        message: str,
        memory_required: int | None = None,
        memory_available: int | None = None,
        memory_limit: int | None = None,
    ):
        """
        Initialize memory error.

        Args:
            message: Error description
            memory_required: Required memory in bytes
            memory_available: Available memory in bytes
            memory_limit: Configured memory limit in bytes
        """
        details = {}
        if memory_required is not None:
            details["memory_required_mb"] = memory_required / (1024 * 1024)
        if memory_available is not None:
            details["memory_available_mb"] = memory_available / (1024 * 1024)
        if memory_limit is not None:
            details["memory_limit_mb"] = memory_limit / (1024 * 1024)

        super().__init__(message, details)
        self.memory_required = memory_required
        self.memory_available = memory_available
        self.memory_limit = memory_limit


class ResourceExhaustionError(SolverError):
    """
    Raised when computation exceeds configured resource limits.
    """

    def __init__(
        self,
        message: str,
        resource: str,
        limit: float | None = None,
        usage: float | None = None,
        suggested_fix: str | None = None,
        recovery_hint: str | None = None,
    ):
        details = {"resource": resource}
        if limit is not None:
            details["limit"] = limit
        if usage is not None:
            details["usage"] = usage
        super().__init__(
            message,
            details=details,
            suggested_fix=suggested_fix,
            recovery_hint=recovery_hint,
        )
        self.resource = resource
        self.limit = limit
        self.usage = usage


class TimeoutError(SolverError):
    """
    Exception raised when solver operations exceed time limits.

    This includes solve timeout, preprocessing timeout, and other time-
    related constraints.
    """

    def __init__(
        self,
        message: str,
        time_elapsed: float | None = None,
        time_limit: float | None = None,
        operation: str | None = None,
    ):
        """
        Initialize timeout error.

        Args:
            message: Error description
            time_elapsed: Time elapsed before timeout in seconds
            time_limit: Configured time limit in seconds
            operation: Name of the operation that timed out
        """
        details = {}
        if time_elapsed is not None:
            details["time_elapsed_s"] = time_elapsed
        if time_limit is not None:
            details["time_limit_s"] = time_limit
        if operation:
            details["operation"] = operation

        super().__init__(message, details)
        self.time_elapsed = time_elapsed
        self.time_limit = time_limit
        self.operation = operation


class NumericalError(SolverError):
    """
    Exception raised for numerical computation issues.

    This includes floating-point overflow/underflow, loss of precision,
    and other numerical stability problems.
    """

    def __init__(
        self,
        message: str,
        computation_details: dict[str, Any] | None = None,
        suggested_mode: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize numerical error.
        """
        details = {}
        if computation_details:
            details.update(computation_details)
        if suggested_mode:
            details["suggested_mode"] = suggested_mode

        super().__init__(message, details, **kwargs)
        self.computation_details = computation_details
        self.suggested_mode = suggested_mode


class NumericalInstabilityError(NumericalError):
    """
    Raised when numerical computations become unstable.
    """

    def __init__(
        self,
        message: str,
        computation_details: dict[str, Any] | None = None,
        component: str | None = None,
        suggested_fix: str | None = None,
        recovery_hint: str | None = None,
    ):
        super().__init__(
            message,
            computation_details=computation_details,
            suggested_mode=None,
            component=component,
        )
        if suggested_fix:
            self.details.setdefault("suggested_fix", suggested_fix)
        if recovery_hint:
            self.details.setdefault("recovery_hint", recovery_hint)


class ConvergenceError(AlgorithmError):
    """
    Exception raised when algorithms fail to converge.

    This is a specialized AlgorithmError for convergence-related
    failures in iterative algorithms like binary search or Bellman-Ford.
    """

    def __init__(
        self,
        message: str,
        algorithm_name: str,
        max_iterations: int,
        tolerance: float | None = None,
        final_error: float | None = None,
    ):
        """
        Initialize convergence error.

        Args:
            message: Error description
            algorithm_name: Name of the algorithm that failed to converge
            max_iterations: Maximum iterations allowed
            tolerance: Required tolerance for convergence
            final_error: Final error when convergence failed
        """
        convergence_info = {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "final_error": final_error,
        }

        super().__init__(
            message=message,
            algorithm_name=algorithm_name,
            iterations=max_iterations,
            convergence_info={
                k: v for k, v in convergence_info.items() if v is not None
            },
        )


class CycleValidationError(ValidationError):
    """
    Exception raised when cycle validation fails.

    This includes invalid cycle structure, missing edges in cycles,
    incorrect weight calculations, and other cycle-related validation
    issues.
    """

    def __init__(
        self,
        message: str,
        cycle: list[int] | None = None,
        missing_edges: list[tuple] | None = None,
        weight_mismatch: dict[str, float] | None = None,
    ):
        """
        Initialize cycle validation error.

        Args:
            message: Error description
            cycle: The invalid cycle
            missing_edges: List of missing edges as (u, v) tuples
            weight_mismatch: Dictionary with expected vs actual weights
        """
        details = {}
        if cycle is not None:
            details["cycle"] = cycle
            details["cycle_length"] = len(cycle)
        if missing_edges:
            details["missing_edges"] = missing_edges
            details["missing_edge_count"] = len(missing_edges)
        if weight_mismatch:
            details.update(weight_mismatch)

        super().__init__(message, details)
        self.cycle = cycle
        self.missing_edges = missing_edges
        self.weight_mismatch = weight_mismatch


# Exception handling utilities
class ErrorHandler:
    """
    Utility class for consistent error handling and reporting.
    """

    @staticmethod
    def handle_validation_error(
        invalid_value: Any,
        parameter_name: str,
        expected_type: type | None = None,
        valid_range: tuple | None = None,
        custom_message: str | None = None,
    ) -> None:
        """
        Raise a properly formatted validation error.

        Args:
            invalid_value: The invalid value
            parameter_name: Name of the parameter
            expected_type: Expected type
            valid_range: Valid range for the parameter
            custom_message: Custom error message

        Raises:
            ValidationError: Always raises this exception
        """
        if custom_message:
            message = custom_message
        else:
            message = f"Invalid value for parameter '{parameter_name}': {invalid_value}"

            if expected_type:
                message += f", expected {expected_type.__name__}"
            if valid_range:
                message += f", valid range: {valid_range}"

        raise ValidationError(
            message=message,
            invalid_value=invalid_value,
            expected_type=expected_type,
            valid_range=valid_range,
        )

    @staticmethod
    def handle_graph_error(
        message: str,
        n_vertices: int | None = None,
        n_edges: int | None = None,
        density: float | None = None,
        suggested_fix: str | None = None,
    ) -> None:
        """
        Raise a properly formatted graph error.

        Args:
            message: Error message
            n_vertices: Number of vertices
            n_edges: Number of edges
            density: Graph density
            suggested_fix: Suggested fix for the issue

        Raises:
            GraphError: Always raises this exception
        """
        graph_properties = {}
        if n_vertices is not None:
            graph_properties["n_vertices"] = n_vertices
        if n_edges is not None:
            graph_properties["n_edges"] = n_edges
        if density is not None:
            graph_properties["density"] = density

        raise GraphError(
            message=message,
            graph_properties=graph_properties if graph_properties else None,
            suggested_fix=suggested_fix,
        )

    @staticmethod
    def handle_algorithm_error(
        message: str,
        algorithm_name: str,
        iterations: int | None = None,
        tolerance: float | None = None,
        final_error: float | None = None,
        is_convergence_error: bool = False,
    ) -> None:
        """
        Raise a properly formatted algorithm error.

        Args:
            message: Error message
            algorithm_name: Name of the failing algorithm
            iterations: Number of iterations completed
            tolerance: Required tolerance (for convergence errors)
            final_error: Final error value (for convergence errors)
            is_convergence_error: Whether this is a convergence failure

        Raises:
            ConvergenceError or AlgorithmError: Depending on error type
        """
        if is_convergence_error and tolerance is not None:
            raise ConvergenceError(
                message=message,
                algorithm_name=algorithm_name,
                max_iterations=iterations or 0,
                tolerance=tolerance,
                final_error=final_error,
            )
        else:
            convergence_info = {}
            if tolerance is not None:
                convergence_info["tolerance"] = tolerance
            if final_error is not None:
                convergence_info["final_error"] = final_error

            raise AlgorithmError(
                message=message,
                algorithm_name=algorithm_name,
                iterations=iterations,
                convergence_info=convergence_info if convergence_info else None,
            )


def format_exception_chain(exc: Exception) -> str:
    """
    Format an exception chain for better error reporting.

    Args:
        exc: The exception to format

    Returns:
        Formatted string representation of the exception chain
    """
    lines = []
    current = exc

    while current is not None:
        if isinstance(current, SolverError):
            lines.append(f"{current.__class__.__name__}: {current}")
            if current.details:
                for key, value in current.details.items():
                    lines.append(f"  {key}: {value}")
        else:
            lines.append(f"{current.__class__.__name__}: {str(current)}")

        current = current.__cause__ or current.__context__
        if current:
            lines.append("Caused by:")

    return "\n".join(lines)


# Export common error handling functions
__all__ = [
    # Exception classes
    "SolverError",
    "ValidationError",
    "GraphError",
    "GraphStructureError",
    "AlgorithmError",
    "ConfigurationError",
    "MemoryError",
    "ResourceExhaustionError",
    "TimeoutError",
    "NumericalError",
    "NumericalInstabilityError",
    "ConvergenceError",
    "CycleValidationError",
    # Utility classes
    "ErrorHandler",
    # Utility functions
    "format_exception_chain",
]


if __name__ == "__main__":
    # Test the exception classes
    print("Testing Min Ratio Cycle Solver exceptions...")

    try:
        # Test validation error
        ErrorHandler.handle_validation_error(
            invalid_value=-1,
            parameter_name="n_vertices",
            expected_type=int,
            valid_range=(1, float("inf")),
        )
    except ValidationError as e:
        print(f"ValidationError: {e}")
        print(f"Error details: {e.details}")

    try:
        # Test graph error
        ErrorHandler.handle_graph_error(
            message="Graph has no cycles",
            n_vertices=5,
            n_edges=4,
            density=0.16,
            suggested_fix="Add more edges to create cycles",
        )
    except GraphError as e:
        print(f"GraphError: {e}")
        print(f"Suggested fix: {e.suggested_fix}")

    try:
        # Test convergence error
        ErrorHandler.handle_algorithm_error(
            message="Binary search failed to converge",
            algorithm_name="Lawler",
            iterations=100,
            tolerance=1e-12,
            final_error=1e-6,
            is_convergence_error=True,
        )
    except ConvergenceError as e:
        print(f"ConvergenceError: {e}")
        print(f"Algorithm: {e.algorithm_name}")

    # Test exception formatting
    try:
        try:
            raise ValidationError("Invalid parameter")
        except ValidationError as inner:
            raise AlgorithmError("Algorithm failed", algorithm_name="test") from inner
    except Exception as e:
        formatted = format_exception_chain(e)
        print(f"Exception chain:\n{formatted}")

    print("Exception testing completed!")
