import time
from contextlib import contextmanager


# Missing: Progress tracking for long operations
@contextmanager
def progress_tracker(description, total_steps=None):
    """
    Context manager for tracking progress of long operations.
    """

    print(f"\n{description}...", end="", flush=True)
    start_time = time.time()

    class ProgressState:
        def __init__(self):
            self.step = 0
            self.total = total_steps

        def update(self, step=None, message=""):
            if step is not None:
                self.step = step
            else:
                self.step += 1

            if self.total:
                percent = (self.step / self.total) * 100
                print(
                    f"\r{description}... {percent:.1f}% {message}", end="", flush=True
                )
            else:
                print(".", end="", flush=True)

    try:
        yield ProgressState()
    finally:
        elapsed = time.time() - start_time
        print(f" Done! ({elapsed:.2f}s)")
