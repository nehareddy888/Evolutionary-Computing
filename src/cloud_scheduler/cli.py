"""Command line entry point."""
import argparse, json
from dataclasses import asdict
from pathlib import Path
from .workloads import default_vms, generate_workload
from .algorithms import random_schedule, min_min_schedule, hybrid_ga_schedule
from .experiment import run_experiments


def main(argv=None):
    p=argparse.ArgumentParser(description="Cloud task scheduling simulator")
    p.add_argument("--tasks",type=int,default=40); p.add_argument("--seed",type=int,default=7); p.add_argument("--compare",action="store_true")
    p.add_argument("--population",type=int,default=40); p.add_argument("--generations",type=int,default=80); p.add_argument("--mutation-rate",type=float,default=.08)
    p.add_argument("--weights",type=float,nargs=3,default=(.55,.25,.20),metavar=("MAKESPAN","COST","IMBALANCE"))
    p.add_argument("--output",default="results"); p.add_argument("--full-experiments",action="store_true")
    a=p.parse_args(argv); vms=default_vms(); tasks=generate_workload(a.tasks,a.seed,vms)
    options=dict(population_size=a.population,generations=a.generations,mutation_rate=a.mutation_rate,weights=tuple(a.weights))
    if a.full_experiments:
        run_experiments(a.output,ga_options=options); print(f"Experiments written to {Path(a.output).resolve()}"); return
    metrics=[random_schedule(tasks,vms,a.seed),min_min_schedule(tasks,vms),hybrid_ga_schedule(tasks,vms,a.seed,**options)] if a.compare else [hybrid_ga_schedule(tasks,vms,a.seed,**options)]
    dest=Path(a.output); dest.mkdir(parents=True,exist_ok=True)
    data=[asdict(m) for m in metrics]
    (dest/"single_run.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    for m in metrics: print(f"{m.algorithm:10s} makespan={m.makespan:.3f}s cost={m.total_cost:.3f} imbalance={m.load_imbalance:.3f} feasible={m.feasible} runtime={m.runtime_seconds:.4f}s")

if __name__=="__main__": main()
