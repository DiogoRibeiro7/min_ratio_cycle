# Makefile for min-ratio-cycle development

.PHONY: help install test test-quick test-slow test-bench test-property test-integration
.PHONY: coverage lint format type-check clean benchmark
.PHONY: test-all test-ci

# Default target
help:
	@echo "Min Ratio Cycle Solver - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  install         Install dependencies"
	@echo "  install-dev     Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo "  test-quick      Run fast tests only"
	@echo "  test-slow       Run slow tests"
	@echo "  test-bench      Run benchmark tests"
	@echo "  test-property   Run property-based tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-ci         Run CI test suite"
	@echo "  coverage        Generate coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run linting"
	@echo "  format          Format code"
	@echo "  type-check      Run type checking"
	@echo "  quality         Run all quality checks"
	@echo ""
	@echo "Performance:"
	@echo "  benchmark       Run performance benchmarks"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean           Clean build artifacts"
	@echo "  clean-all       Clean everything including caches"

# Installation
install:
	poetry install

install-dev:
	poetry install --with dev,test

# Testing targets
test:
	python run_tests.py

test-quick:
	python run_tests.py --quick

test-slow:
	python run_tests.py --slow

test-bench:
	python run_tests.py --bench

test-property:
	python run_tests.py --property

test-integration:
	python run_tests.py --integration

test-ci:
	python run_tests.py --quick --coverage --parallel

test-all:
	python run_tests.py --slow --bench --property --coverage

# Coverage
coverage:
	python run_tests.py --coverage
	@echo "Coverage report: htmlcov/index.html"

# Code quality
lint:
	poetry run flake8 min_ratio_cycle/ test_*.py --max-line-length=88
	@echo "✓ Linting completed"

format:
	poetry run black min_ratio_cycle/ test_*.py *.py
	poetry run isort min_ratio_cycle/ test_*.py *.py
	@echo "✓ Code formatting completed"

type-check:
	poetry run mypy min_ratio_cycle/ --strict
	@echo "✓ Type checking completed"

quality: lint type-check
	@echo "✓ All quality checks passed"

# Performance
benchmark:
	python benchmark_suite.py
	@echo "✓ Benchmarks completed"

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Build artifacts cleaned"

clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .hypothesis/
	rm -f benchmark_results.png
	@echo "✓ All artifacts cleaned"

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test-quick' to verify installation"

# Continuous Integration simulation
ci: install-dev quality test-ci
	@echo "✓ CI pipeline completed successfully"

# Pre-commit hook simulation
pre-commit: format quality test-quick
	@echo "✓ Pre-commit checks passed"

# Release preparation
release-check: clean quality test-all benchmark
	@echo "✓ Release checks completed"
	@echo "Ready for release!"

# Show test status
test-status:
	@echo "Test Status Summary:"
	@echo "==================="
	@python -c "
import subprocess
import sys

try:
    result = subprocess.run(['pytest', '--collect-only', '-q'], 
                          capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        if 'collected' in line:
            print(f'📊 {line}')
except:
    print('❌ Could not collect test information')
"

# Performance monitoring
perf-monitor:
	@echo "Running performance monitoring..."
	python -c "
import time
import psutil
import subprocess

print('System Resources:')
print(f'  CPU cores: {psutil.cpu_count()}')
print(f'  Memory: {psutil.virtual_memory().total // (1024**3)}GB')
print()

start_time = time.time()
result = subprocess.run(['python', 'run_tests.py', '--quick'], 
                       capture_output=True)
elapsed = time.time() - start_time

print(f'Quick test suite completed in {elapsed:.2f}s')
if result.returncode == 0:
    print('✓ Tests passed')
else:
    print('❌ Tests failed')
"
