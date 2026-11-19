# Cloud Computing Task Scheduling Using Hybrid Genetic Algorithm

An explainable Python simulation that assigns independent tasks to heterogeneous virtual machines (VMs). It compares reproducible Random Scheduling, Min-Min, and a Hybrid Genetic Algorithm (GA) with memory-capacity repair and local search. It is an academic simulator; it does not connect to a real cloud provider.

## Problem and objectives

For each task, choose one VM while respecting each VM's aggregate memory capacity. Tasks on a VM run sequentially. The optimizer balances makespan, estimated execution cost, and load imbalance. Since these measures have different units, the GA scales time and cost by workload-dependent upper bounds and combines them with nonnegative configurable weights.

## Project structure

* `src/cloud_scheduler/models.py`: task, VM, and result records.
* `simulation.py`: common execution time, memory feasibility, and metrics.
* `workloads.py`: seeded workloads and default heterogeneous VMs.
* `algorithms.py`: Random, Min-Min, chromosome repair, and Hybrid GA.
* `experiment.py`: matched workload runs, CSV/JSON exports, and plots.
* `cli.py`: command line interface.
* `tests/`: automated checks.
* `results/`: generated experiment files and visualizations.

## Mathematical model

For task *i* with instruction length *Lᵢ* (MI) assigned to VM *j* with capacity *MIPSⱼ*, execution time is `tᵢⱼ = Lᵢ / MIPSⱼ`. The sequential completion time on VM *j* is `Cⱼ = Σ(i assigned to j) tᵢⱼ`, and makespan is `maxⱼ Cⱼ`. Estimated cost is `Σᵢ tᵢⱼ × rateⱼ` using the rate of the assigned VM. A VM is memory-feasible when the sum of memory requirements of all tasks assigned to it does not exceed its capacity.

Load imbalance is the population standard deviation of VM completion times divided by their mean. It is dimensionless; an empty or zero-time workload has zero imbalance. The GA minimizes `α × makespan / Tbound + β × cost / Kbound + γ × imbalance`, where `Tbound` is the sum of task lengths divided by the slowest VM's MIPS and `Kbound` is `Tbound` times the largest VM price. The three weights are normalized to sum to one. Lower objective is better. These are practical normalization bounds, not claims of mathematically tight optima.

### Algorithms

* **Random:** seeded random placement among VMs with remaining memory. Tasks are placed largest-memory first to avoid avoidable dead ends.
* **Min-Min:** repeatedly finds each remaining task's earliest feasible completion time, then schedules the task with the smallest such time and updates the selected VM.
* **Hybrid GA:** a chromosome stores one VM index per task. Tournament selection, uniform crossover, mutation, and elitism explore assignments. A repair pass remaps tasks to fit memory. Best-improvement local search tests feasible single-task moves for leading candidates. The GA can be better than a basic GA or heuristic on some instances because it combines broad search with local refinement, but outcomes depend on workload and weights; no universal advantage is assumed.

## Assumptions and limitations

Tasks are independent and static; each VM runs its assigned tasks sequentially with no overlap. Memory is modeled as a total reservation per VM across its assigned tasks. There are no dependencies, communication or network delays, VM startup delays, failures, deadlines, dynamic arrivals, or cloud API calls. Execution cost uses a simple per-time price and is not a real provider bill. The synthetic workload generator and default VM profiles are illustrative. Results depend on seed, objective weights, and GA parameters. The simulator is not a production scheduler.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this project directory. The `src` layout is supported through `PYTHONPATH` below; alternatively install in editable mode with a packaging configuration of your choice.

```powershell
# PowerShell
$env:PYTHONPATH = "src"
python -m cloud_scheduler.cli --tasks 40 --seed 7 --compare
```

```bash
# macOS/Linux
PYTHONPATH=src python -m cloud_scheduler.cli --tasks 40 --seed 7 --compare
```

The single-run CLI writes `single_run.json` under the selected output folder. Useful options:

```text
--tasks N --seed N --compare
--population N --generations N --mutation-rate RATE
--weights MAKESPAN COST IMBALANCE --output DIRECTORY
--full-experiments
```

To run the full requested experiment matrix (20, 40, 60 tasks; seeds 7, 21, 42):

```powershell
python -m cloud_scheduler.cli --full-experiments --output results
```

## Tests and experiments

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

The experiment runner gives all algorithms the same generated task list and VM profiles for each `(task count, seed)` pair. CSV and JSON retain each run's assignment, metrics, feasibility, and measured algorithm runtime. Plots show mean metric values by task count, one VM load distribution, and the GA best objective by generation averaged across seeds. Runtime and results may vary slightly with machine load; algorithm choices are deterministic for a fixed seed and configuration.

## Interpreting results

Compare algorithms within the same workload and seed. Makespan and cost should be read alongside imbalance and runtime. A lower weighted GA objective does not imply that every individual metric is lower. The GA has an explicit runtime budget controlled by population size, generations, and local-search iterations. The included plots and generated tables are empirical observations from this simulator, not generalized cloud performance claims.

## Future work

Possible extensions include task dependencies, deadlines, network transfer, VM startup and failure models, richer resource vectors (CPU, memory, storage), Pareto optimization, real workload traces, and statistically stronger repeated-run analysis.
