"""Workload scheduling and resource admission controller preventing VRAM OOM."""
import contextlib
import threading
from typing import Iterator


class AdmissionController:
    """Controls concurrent access to memory-heavy resources (e.g. 6GB VRAM on RTX 3060).

    Enforces that heavy model invocations (SigLIP 2, PhoWhisper, LLM) do not exceed available concurrency.
    """

    def __init__(self, max_concurrent_heavy: int = 1):
        self.semaphore = threading.Semaphore(max_concurrent_heavy)
        self.max_concurrent_heavy = max_concurrent_heavy
        self._active_count = 0
        self._lock = threading.Lock()

    @property
    def active_jobs(self) -> int:
        with self._lock:
            return self._active_count

    @contextlib.contextmanager
    def acquire(self, timeout_seconds: float | None = 10.0) -> Iterator[None]:
        """Acquire execution ticket for heavy inference task."""
        acquired = self.semaphore.acquire(timeout=timeout_seconds if timeout_seconds is not None else -1)
        if not acquired:
            raise RuntimeError(
                f"Admission controller timed out waiting for GPU ticket ({timeout_seconds}s)"
            )
        with self._lock:
            self._active_count += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_count = max(0, self._active_count - 1)
            self.semaphore.release()
