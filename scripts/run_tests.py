#!/usr/bin/env python3
"""
Comprehensive test runner for min-ratio-cycle solver.

Usage:
    python run_tests.py [options]

Options:
    --quick     Run only fast tests
    --slow      Include slow tests  
    --bench     Run benchmarks
    --property  Run property-based tests
    --coverage  Generate coverage report
    --parallel  Run tests in parallel
    --verbose   Verbose output
    --help      Show this help
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str, verbose: bool = False):
    """Run a command and handle errors."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    print(f"{description}...")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        if not verbose and result.stdout:
            print(result.stdout)
        
        print(f"✓ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for min-ratio-cycle solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test selection options
    parser.add_argument('--quick', action='store_true', 
                       help='Run only fast tests (exclude slow/benchmark)')
    parser.add_argument('--slow', action='store_true',
                       help='Include slow tests')
    parser.add_argument('--bench', action='store_true',
                       help='Run benchmark tests')
    parser.add_argument('--property', action='store_true',
                       help='Run property-based tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    
    # Output options
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    # Specific test patterns
    parser.add_argument('--pattern', '-k', type=str,
                       help='Run tests matching pattern')
    parser.add_argument('--file', type=str,
                       help='Run specific test file')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ['pytest']
    
    # Base options
    if args.verbose:
        cmd.extend(['-v', '-s'])
    else:
        cmd.append('-q')
    
    # Parallel execution
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    # Coverage
    if args.coverage:
        cmd.extend(['--cov=min_ratio_cycle', '--cov-report=html', '--cov-report=term'])
    
    # Test selection
    markers = []
    
    if args.quick:
        markers.append('not slow and not benchmark')
    elif args.slow:
        markers.append('slow')
    elif args.bench:
        markers.append('benchmark')
    elif args.property:
        markers.append('property')
    elif args.integration:
        markers.append('integration')
    
    if markers:
        cmd.extend(['-m', ' or '.join(markers)])
    
    # Pattern matching
    if args.pattern:
        cmd.extend(['-k', args.pattern])
    
    # Specific file
    if args.file:
        cmd.append(args.file)
    else:
        # Default test directories
        test_files = []
        if Path('test_solver.py').exists():
            test_files.append('test_solver.py')
        if Path('test_integration.py').exists():
            test_files.append('test_integration.py')
        if Path('tests').exists():
            test_files.append('tests/')
        
        if test_files:
            cmd.extend(test_files)
    
    # Run the tests
    print("Min Ratio Cycle Solver - Test Suite")
    print("=" * 50)
    
    success = run_command(cmd, "Running tests", args.verbose)
    
    if not success:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    
    # Additional operations
    if args.coverage:
        print(f"\n📊 Coverage report generated in htmlcov/")
    
    # Run benchmarks if requested
    if args.bench:
        print("\n" + "=" * 50)
        bench_cmd = ['python', 'benchmark_suite.py']
        run_command(bench_cmd, "Running benchmark suite", args.verbose)
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()
