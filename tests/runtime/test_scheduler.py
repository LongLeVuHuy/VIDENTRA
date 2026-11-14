"""Tests for runtime admission controller."""
import threading
import time
import pytest

from videntra.runtime.scheduler import AdmissionController


def test_admission_controller_concurrency_limit():
    controller = AdmissionController(max_concurrent_heavy=1)
    execution_order = []

    def task(task_id: int):
        with controller.acquire():
            execution_order.append(f"start_{task_id}")
            time.sleep(0.05)
            execution_order.append(f"end_{task_id}")

    t1 = threading.Thread(target=task, args=(1,))
    t2 = threading.Thread(target=task, args=(2,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # The tasks must run sequentially because max_concurrent_heavy=1
    assert execution_order in (
        ["start_1", "end_1", "start_2", "end_2"],
        ["start_2", "end_2", "start_1", "end_1"],
    )


def test_admission_controller_timeout():
    controller = AdmissionController(max_concurrent_heavy=1)

    with controller.acquire():
        # Second acquire should timeout quickly
        with pytest.raises(RuntimeError, match="Admission controller timed out"):
            with controller.acquire(timeout_seconds=0.01):
                pass
