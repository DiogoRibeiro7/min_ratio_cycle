"""
Configuration management for min-ratio-cycle solver.

This module provides comprehensive configuration management including:
- Solver parameters for different modes
- Performance and monitoring settings
- Validation and debugging options
- File-based configuration loading/saving
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SolverMode(Enum):
    """
    Solver mode enumeration.
    """

    AUTO = "auto"  # Automatic selection based on weight types
    EXACT = "exact"  # Force exact rational arithmetic
    NUMERIC = "numeric"  # Force floating-point arithmetic
    APPROXIMATE = "approximate"  # Fast approximation


class LogLevel(Enum):
    """
    Logging level enumeration.
    """

    NONE = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4


@dataclass
class SolverConfig:
    """
    Comprehensive configuration for the MinRatioCycleSolver.

    This class manages all configurable parameters for the solver,
    including algorithmic parameters, performance settings, and
    monitoring options.
    """

    # Numeric mode parameters
    numeric_max_iter: int = 60
    numeric_tolerance: float = 1e-12
    numeric_cycle_slack: float = 1e-15

    # Exact mode parameters
    exact_max_denominator: int | None = None
    exact_max_steps: int | None = None

    # Performance parameters
    sparse_threshold: float = 0.1  # Use sparse optimizations below this density
    enable_preprocessing: bool = True
    enable_early_termination: bool = True
    max_solve_time: float | None = None  # Maximum time in seconds

    # Validation parameters
    validate_cycles: bool = True
    repair_cycles: bool = True
    strict_validation: bool = False

    # Logging and monitoring
    log_level: LogLevel = LogLevel.INFO
    collect_metrics: bool = True
    enable_profiling: bool = False
    log_to_file: bool = False
    log_file_path: str | None = None

    # Advanced options
    use_kahan_summation: bool = True  # For numerical stability
    parallel_threshold: int = 1000  # Use parallel processing above this size
    memory_limit_gb: float | None = None  # Memory usage limit

    # Debugging options
    debug_mode: bool = False
    save_intermediate_results: bool = False
    intermediate_results_dir: str | None = None

    # Default configuration dictionary for file-based config
    _default_dict: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        """
        Initialize default configuration dictionary.
        """
        self._default_dict = {
            "numeric_mode": {
                "max_iter": self.numeric_max_iter,
                "tolerance": self.numeric_tolerance,
                "cycle_slack": self.numeric_cycle_slack,
            },
            "exact_mode": {
                "max_denominator": self.exact_max_denominator,
                "max_steps": self.exact_max_steps,
            },
            "performance": {
                "sparse_threshold": self.sparse_threshold,
                "enable_preprocessing": self.enable_preprocessing,
                "enable_early_termination": self.enable_early_termination,
                "max_solve_time": self.max_solve_time,
                "use_kahan_summation": self.use_kahan_summation,
                "parallel_threshold": self.parallel_threshold,
                "memory_limit_gb": self.memory_limit_gb,
            },
            "validation": {
                "validate_cycles": self.validate_cycles,
                "repair_cycles": self.repair_cycles,
                "strict_validation": self.strict_validation,
            },
            "monitoring": {
                "log_level": (
                    self.log_level.value
                    if isinstance(self.log_level, LogLevel)
                    else self.log_level
                ),
                "collect_metrics": self.collect_metrics,
                "enable_profiling": self.enable_profiling,
                "log_to_file": self.log_to_file,
                "log_file_path": self.log_file_path,
            },
            "debugging": {
                "debug_mode": self.debug_mode,
                "save_intermediate_results": self.save_intermediate_results,
                "intermediate_results_dir": self.intermediate_results_dir,
            },
        }

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SolverConfig":
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            SolverConfig instance with loaded settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid JSON
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SolverConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            SolverConfig instance
        """
        # Start with defaults
        instance = cls()

        # Update with provided values
        def update_from_nested_dict(obj, nested_dict):
            """
            Update object attributes from nested dictionary.
            """
            for section, values in nested_dict.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        attr_name = f"{section}_{key}" if section != "root" else key
                        if hasattr(obj, attr_name):
                            # Handle enum conversions
                            if attr_name == "log_level" and isinstance(value, str):
                                try:
                                    value = LogLevel(value)
                                except ValueError:
                                    value = getattr(
                                        LogLevel, value.upper(), LogLevel.INFO
                                    )
                            setattr(obj, attr_name, value)
                else:
                    # Direct attribute
                    if hasattr(obj, section):
                        setattr(obj, section, values)

        update_from_nested_dict(instance, config_dict)
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary representation of configuration
        """
        self.__post_init__()  # Ensure _default_dict is updated
        return self._default_dict.copy()

    def save_to_file(self, config_path: str | Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

    def get_logging_config(self) -> dict[str, Any]:
        """
        Get logging-specific configuration.
        """
        return {
            "level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file_path": self.log_file_path,
            "debug_mode": self.debug_mode,
        }

    def get_performance_config(self) -> dict[str, Any]:
        """
        Get performance-specific configuration.
        """
        return {
            "sparse_threshold": self.sparse_threshold,
            "enable_preprocessing": self.enable_preprocessing,
            "enable_early_termination": self.enable_early_termination,
            "max_solve_time": self.max_solve_time,
            "use_kahan_summation": self.use_kahan_summation,
            "parallel_threshold": self.parallel_threshold,
            "memory_limit_gb": self.memory_limit_gb,
        }

    def get_validation_config(self) -> dict[str, Any]:
        """
        Get validation-specific configuration.
        """
        return {
            "validate_cycles": self.validate_cycles,
            "repair_cycles": self.repair_cycles,
            "strict_validation": self.strict_validation,
        }

    def is_high_performance_mode(self) -> bool:
        """
        Check if high-performance optimizations should be enabled.
        """
        return (
            not self.validate_cycles
            and not self.collect_metrics
            and not self.enable_profiling
            and self.log_level in (LogLevel.NONE, LogLevel.ERROR)
        )

    def validate(self) -> List[str]:
        """
        Validate configuration parameters.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Validate numeric parameters
        if self.numeric_max_iter <= 0:
            issues.append("numeric_max_iter must be positive")

        if self.numeric_tolerance <= 0:
            issues.append("numeric_tolerance must be positive")

        if not (0.0 < self.sparse_threshold <= 1.0):
            issues.append("sparse_threshold must be between 0 and 1")

        # Validate file paths
        if self.log_to_file and not self.log_file_path:
            issues.append("log_file_path required when log_to_file is True")

        if self.save_intermediate_results and not self.intermediate_results_dir:
            issues.append(
                "intermediate_results_dir required when save_intermediate_results is True"
            )

        # Validate memory limit
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            issues.append("memory_limit_gb must be positive if specified")

        return issues

    def __str__(self) -> str:
        """
        String representation of configuration.
        """
        return f"SolverConfig(mode=auto, logging={self.log_level.name}, validation={self.validate_cycles})"

    def __repr__(self) -> str:
        """
        Detailed string representation.
        """
        return (
            f"SolverConfig(numeric_max_iter={self.numeric_max_iter}, "
            f"log_level={self.log_level.name}, "
            f"validate_cycles={self.validate_cycles}, "
            f"collect_metrics={self.collect_metrics})"
        )


# Predefined configurations for common use cases
class ConfigPresets:
    """
    Predefined configuration presets for common scenarios.
    """

    @staticmethod
    def fast_mode() -> SolverConfig:
        """
        Configuration optimized for speed over accuracy.
        """
        return SolverConfig(
            validate_cycles=False,
            repair_cycles=False,
            collect_metrics=False,
            enable_profiling=False,
            log_level=LogLevel.ERROR,
            enable_preprocessing=True,
            enable_early_termination=True,
            use_kahan_summation=False,  # Slightly faster but less precise
        )

    @staticmethod
    def accurate_mode() -> SolverConfig:
        """
        Configuration optimized for accuracy over speed.
        """
        return SolverConfig(
            validate_cycles=True,
            repair_cycles=True,
            strict_validation=True,
            collect_metrics=True,
            log_level=LogLevel.INFO,
            use_kahan_summation=True,
            numeric_tolerance=1e-15,  # Higher precision
            numeric_cycle_slack=1e-18,
        )

    @staticmethod
    def debug_mode() -> SolverConfig:
        """
        Configuration for debugging and development.
        """
        return SolverConfig(
            debug_mode=True,
            log_level=LogLevel.DEBUG,
            validate_cycles=True,
            repair_cycles=True,
            collect_metrics=True,
            enable_profiling=True,
            save_intermediate_results=True,
            log_to_file=True,
            log_file_path="solver_debug.log",
        )

    @staticmethod
    def production_mode() -> SolverConfig:
        """
        Configuration for production deployment.
        """
        return SolverConfig(
            validate_cycles=True,
            repair_cycles=True,
            collect_metrics=True,
            log_level=LogLevel.INFO,
            log_to_file=True,
            log_file_path="/var/log/min_ratio_cycle.log",
            memory_limit_gb=4.0,
            max_solve_time=300.0,  # 5 minutes max
        )

    @staticmethod
    def research_mode() -> SolverConfig:
        """
        Configuration for research and experimentation.
        """
        return SolverConfig(
            collect_metrics=True,
            enable_profiling=True,
            log_level=LogLevel.DEBUG,
            save_intermediate_results=True,
            validate_cycles=True,
            strict_validation=True,
            debug_mode=True,
        )


# Default configuration instance
DEFAULT_CONFIG = SolverConfig()


if __name__ == "__main__":
    # Example usage
    print("Min Ratio Cycle Solver - Configuration Management")
    print("=" * 50)

    # Test basic configuration
    config = SolverConfig()
    print(f"Default config: {config}")

    # Test preset configurations
    fast_config = ConfigPresets.fast_mode()
    print(f"Fast mode: {fast_config}")

    debug_config = ConfigPresets.debug_mode()
    print(f"Debug mode: {debug_config}")

    # Test file operations
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_path = f.name

    try:
        # Save and load configuration
        debug_config.save_to_file(config_path)
        loaded_config = SolverConfig.from_file(config_path)
        print(f"Loaded config: {loaded_config}")

        # Validate configuration
        issues = loaded_config.validate()
        if issues:
            print(f"Validation issues: {issues}")
        else:
            print("Configuration validation passed")

    finally:
        # Cleanup
        import os

        if os.path.exists(config_path):
            os.unlink(config_path)
