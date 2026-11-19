"""Random, Min-Min, and hybrid genetic scheduling algorithms."""
from __future__ import annotations
import random
import time
from .models import Task, VM, ScheduleMetrics
from .simulation import execution_time, feasible_assignment, evaluate


def _check(tasks, vms):
    if not vms: raise ValueError("At least one VM is required")
    impossible = [t.task_id for t in tasks if not any(t.memory_mb <= v.memory_mb for v in vms)]
    if impossible: raise ValueError("Tasks exceed every VM memory capacity: " + ", ".join(impossible))


def random_schedule(tasks: list[Task], vms: list[VM], seed: int = 7) -> ScheduleMetrics:
    _check(tasks, vms); start = time.perf_counter(); rng = random.Random(seed)
    assignment = [-1] * len(tasks); used = [0.0] * len(vms)
    order = sorted(range(len(tasks)), key=lambda i: tasks[i].memory_mb, reverse=True)
    for i in order:
        choices = [j for j, vm in enumerate(vms) if used[j] + tasks[i].memory_mb <= vm.memory_mb + 1e-9]
        if not choices: raise ValueError(f"No feasible placement for {tasks[i].task_id}")
        j = rng.choice(choices); assignment[i] = j; used[j] += tasks[i].memory_mb
    return evaluate(tasks, vms, assignment, "Random", time.perf_counter()-start)


def min_min_schedule(tasks: list[Task], vms: list[VM]) -> ScheduleMetrics:
    _check(tasks, vms); start = time.perf_counter(); avail = [0.0]*len(vms); used = [0.0]*len(vms)
    assignment = [-1]*len(tasks); remaining = set(range(len(tasks)))
    while remaining:
        candidates = []
        for i in remaining:
            for j, vm in enumerate(vms):
                if used[j] + tasks[i].memory_mb <= vm.memory_mb + 1e-9:
                    completion = avail[j] + execution_time(tasks[i], vm)
                    candidates.append((completion, i, j))
        if not candidates:
            raise ValueError("Workload is infeasible under VM memory capacities")
        _, i, j = min(candidates)
        assignment[i] = j; avail[j] += execution_time(tasks[i], vms[j]); used[j] += tasks[i].memory_mb; remaining.remove(i)
    return evaluate(tasks, vms, assignment, "Min-Min", time.perf_counter()-start)


def _repair(chrom, tasks, vms):
    used = [0.0]*len(vms); result = [-1]*len(tasks)
    for i in sorted(range(len(tasks)), key=lambda i: tasks[i].memory_mb, reverse=True):
        preferred = chrom[i] if isinstance(chrom[i], int) and 0 <= chrom[i] < len(vms) else 0
        options = sorted(range(len(vms)), key=lambda j: (j != preferred, vms[j].mips))
        found = next((j for j in options if used[j] + tasks[i].memory_mb <= vms[j].memory_mb + 1e-9), None)
        if found is None: return None
        result[i] = found; used[found] += tasks[i].memory_mb
    return result


def _objective(tasks, vms, chrom, weights):
    m = evaluate(tasks, vms, chrom, "candidate")
    max_time = sum(t.length_mi/min(vm.mips for vm in vms) for t in tasks) or 1
    max_cost = sum(t.length_mi/min(vm.mips for vm in vms)*max(vm.cost_per_time for vm in vms) for t in tasks) or 1
    # Imbalance is normalized relative load deviation and is already dimensionless.
    return weights[0]*m.makespan/max_time + weights[1]*m.total_cost/max_cost + weights[2]*m.load_imbalance, m


def hybrid_ga_schedule(tasks: list[Task], vms: list[VM], seed: int = 7, population_size: int = 40,
                       generations: int = 80, mutation_rate: float = 0.08, weights=(0.55, 0.25, 0.20),
                       elite_count: int = 2, local_search_iterations: int = 2, convergence: list[float] | None = None) -> ScheduleMetrics:
    _check(tasks, vms)
    if population_size < 2 or generations < 0 or not 0 <= mutation_rate <= 1: raise ValueError("Invalid GA configuration")
    if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0: raise ValueError("Weights must be three nonnegative values with positive sum")
    weights = tuple(w/sum(weights) for w in weights); start = time.perf_counter(); rng = random.Random(seed); nvm=len(vms)
    def make(): return _repair([rng.randrange(nvm) for _ in tasks], tasks, vms)
    pop=[]
    while len(pop)<population_size:
        item=make()
        if item is not None: pop.append(item)
        if len(pop)==0 and len(pop)==0 and len(tasks)==0: pop.append([])
    def score(c): return _objective(tasks,vms,c,weights)[0]
    def local(c):
        c=list(c); best=score(c)
        for _ in range(local_search_iterations):
            improved=False
            for i in range(len(tasks)):
                for j in range(nvm):
                    if j==c[i]: continue
                    cand=c.copy(); cand[i]=j; cand=_repair(cand,tasks,vms)
                    if cand is not None and score(cand) < best-1e-12: c,best,improved=cand,score(cand),True
            if not improved: break
        return c
    for _ in range(generations):
        pop.sort(key=score); nextpop=[list(x) for x in pop[:max(1,min(elite_count,population_size-1))]]
        def tournament(): return min((rng.choice(pop) for _ in range(3)), key=score)
        while len(nextpop)<population_size:
            a,b=tournament(),tournament(); child=[a[i] if rng.random()<0.5 else b[i] for i in range(len(tasks))]
            for i in range(len(child)):
                if rng.random()<mutation_rate: child[i]=rng.randrange(nvm)
            repaired=_repair(child,tasks,vms)
            nextpop.append(repaired if repaired is not None else list(a))
        pop=nextpop
        # Refine a small elite subset each generation.
        pop.sort(key=score)
        for k in range(min(2,len(pop))): pop[k]=local(pop[k])
        if convergence is not None:
            convergence.append(min(map(score, pop)))
    pop.sort(key=score); best=pop[0]; obj, _ = _objective(tasks,vms,best,weights)
    return evaluate(tasks,vms,best,"Hybrid GA",time.perf_counter()-start,obj)
