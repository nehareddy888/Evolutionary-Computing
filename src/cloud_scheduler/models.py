"""Core immutable data models."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    length_mi: float
    memory_mb: float
    priority: int = 1

    def __post_init__(self):
        if not self.task_id or self.length_mi < 0 or self.memory_mb < 0 or self.priority < 1:
            raise ValueError("Task ID must be nonempty; length/memory nonnegative and priority >= 1")


@dataclass(frozen=True)
class VM:
    vm_id: str
    mips: float
    memory_mb: float
    cost_per_time: float

    def __post_init__(self):
        if not self.vm_id or self.mips <= 0 or self.memory_mb < 0 or self.cost_per_time < 0:
            raise ValueError("VM ID must be nonempty, MIPS positive, memory/cost nonnegative")


@dataclass
class ScheduleMetrics:
    algorithm: str
    assignment: list[int]
    makespan: float
    total_cost: float
    vm_loads: list[float]
    load_imbalance: float
    feasible: bool
    runtime_seconds: float
    objective: float | None = None
