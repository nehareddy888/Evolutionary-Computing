"""Shared feasibility and metric calculations."""
import math
from .models import Task, VM, ScheduleMetrics


def execution_time(task: Task, vm: VM) -> float:
    return task.length_mi / vm.mips


def feasible_assignment(tasks: list[Task], vms: list[VM], assignment: list[int]) -> bool:
    if len(tasks) != len(assignment) or not vms:
        return False
    usage = [0.0] * len(vms)
    for task, index in zip(tasks, assignment):
        if not isinstance(index, int) or index < 0 or index >= len(vms):
            return False
        usage[index] += task.memory_mb
    return all(x <= vm.memory_mb + 1e-9 for x, vm in zip(usage, vms))


def evaluate(tasks: list[Task], vms: list[VM], assignment: list[int], algorithm: str,
             runtime_seconds: float = 0.0, objective: float | None = None) -> ScheduleMetrics:
    if not vms:
        raise ValueError("At least one VM is required")
    if len(tasks) != len(assignment):
        raise ValueError("Assignment length must equal task count")
    times = [0.0] * len(vms)
    cost = 0.0
    memory = [0.0] * len(vms)
    valid = True
    for task, ix in zip(tasks, assignment):
        if not isinstance(ix, int) or not 0 <= ix < len(vms):
            valid = False
            continue
        vm = vms[ix]
        times[ix] += execution_time(task, vm)
        memory[ix] += task.memory_mb
        cost += execution_time(task, vm) * vm.cost_per_time
    valid = valid and all(x <= vm.memory_mb + 1e-9 for x, vm in zip(memory, vms))
    total = sum(times)
    loads = [x / total if total else 0.0 for x in times]
    imbalance = math.sqrt(sum((x - (sum(times) / len(times))) ** 2 for x in times) / len(times)) / max(sum(times) / len(times), 1e-12) if times else 0.0
    return ScheduleMetrics(algorithm, list(assignment), max(times, default=0.0), cost, loads, imbalance, valid, runtime_seconds, objective)
