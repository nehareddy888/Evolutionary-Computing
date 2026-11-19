"""Deterministic synthetic workload generation."""
import numpy as np
from .models import Task, VM


def default_vms() -> list[VM]:
    return [VM("vm-0", 900, 4096, 0.030), VM("vm-1", 1500, 6144, 0.052), VM("vm-2", 2400, 8192, 0.081), VM("vm-3", 1200, 3072, 0.038)]


def generate_workload(n_tasks: int, seed: int = 7, vms: list[VM] | None = None) -> list[Task]:
    if n_tasks < 0:
        raise ValueError("Task count cannot be negative")
    vms = vms or default_vms()
    rng = np.random.default_rng(seed)
    max_mem = max(vm.memory_mb for vm in vms)
    # Keep synthetic aggregate demand comfortably below aggregate capacity so
    # the comparison matrix measures scheduler quality rather than overload.
    mem_upper = max(129, int(max_mem * 0.055))
    return [Task(f"task-{i:03d}", float(rng.integers(20_000, 300_001)),
                 float(rng.integers(128, mem_upper)), int(rng.integers(1, 6))) for i in range(n_tasks)]
