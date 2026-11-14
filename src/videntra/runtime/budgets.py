"""Deadline and execution budget enforcement preventing SLA violations."""
import time


class DeadlineBudget:
    """Tracks elapsed wall-clock time against a configured maximum duration budget."""

    def __init__(self, max_duration_seconds: float = 2.0):
        self.max_duration_seconds = max_duration_seconds
        self.start_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_seconds * 1000.0

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_duration_seconds - self.elapsed_seconds)

    def is_expired(self) -> bool:
        return self.elapsed_seconds >= self.max_duration_seconds

    def check(self) -> None:
        """Raise TimeoutError if budget has expired."""
        if self.is_expired():
            raise TimeoutError(
                f"Execution budget exceeded: {self.elapsed_seconds:.3f}s >= {self.max_duration_seconds:.3f}s"
            )
