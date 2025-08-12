import time
from contextlib import contextmanager


# Missing: Performance profiler integration
class SolverProfiler:
    """Performance profiling utilities."""
    
    def __init__(self, enable_profiling=False):
        self.enable_profiling = enable_profiling
        self.profiles = []
    
    @contextmanager
    def profile_operation(self, operation_name):
        """Profile a specific operation."""
        if not self.enable_profiling:
            yield
            return
        
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            profiler.disable()
            
            # Capture profile stats
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
            
            profile_data = {
                'operation': operation_name,
                'wall_time': elapsed,
                'profile_stats': stats_stream.getvalue()
            }
            
            self.profiles.append(profile_data)
    
    def get_profile_report(self):
        """Get formatted profile report."""
        if not self.profiles:
            return "No profiling data collected"
        
        report = "Performance Profile Report\n" + "="*40 + "\n"
        for profile in self.profiles:
            report += f"\nOperation: {profile['operation']}\n"
            report += f"Wall time: {profile['wall_time']:.4f}s\n"
            report += f"Profile details:\n{profile['profile_stats']}\n"
        
        return report
