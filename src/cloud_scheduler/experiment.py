"""Experiment runner, serialization, and plotting."""
import json, csv
from pathlib import Path
from dataclasses import asdict
import matplotlib.pyplot as plt
import pandas as pd
from .algorithms import random_schedule, min_min_schedule, hybrid_ga_schedule
from .workloads import default_vms, generate_workload


def run_experiments(output_dir="results", sizes=(20,40,60), seeds=(7,21,42), ga_options=None):
    out=Path(output_dir); (out/"csv").mkdir(parents=True,exist_ok=True); (out/"json").mkdir(parents=True,exist_ok=True); (out/"plots").mkdir(parents=True,exist_ok=True)
    records=[]; traces=[]; ga_options=ga_options or {}
    for n in sizes:
        for seed in seeds:
            vms=default_vms(); tasks=generate_workload(n,seed,vms)
            trace=[]
            metrics=[random_schedule(tasks,vms,seed),min_min_schedule(tasks,vms),hybrid_ga_schedule(tasks,vms,seed=seed,convergence=trace,**ga_options)]
            traces.append({"task_count":n,"seed":seed,"values":trace})
            for m in metrics:
                d=asdict(m); d["task_count"]=n; d["seed"]=seed; d["vm_loads"]=json.dumps(m.vm_loads); d["assignment"]=json.dumps(m.assignment); records.append(d)
    df=pd.DataFrame(records); df.to_csv(out/"csv"/"experiment_results.csv",index=False)
    with open(out/"json"/"experiment_results.json","w",encoding="utf-8") as f: json.dump(records,f,indent=2)
    # Matched-instance means with individual runs retained in CSV/JSON.
    for metric,title,ylabel,filename in [("makespan","Makespan by workload size","Time (seconds)","makespan.png"),("total_cost","Estimated execution cost","Cost units","cost.png"),("load_imbalance","Load imbalance","Relative standard deviation","imbalance.png"),("runtime_seconds","Algorithm runtime","Time (seconds)","runtime.png")]:
        fig,ax=plt.subplots(figsize=(8,5)); df.groupby(["task_count","algorithm"])[metric].mean().unstack().plot(kind="bar",ax=ax)
        ax.set_title(title); ax.set_xlabel("Number of tasks"); ax.set_ylabel(ylabel); ax.legend(title="Algorithm"); fig.tight_layout(); fig.savefig(out/"plots"/filename,dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    grouped={}
    for tr in traces:
        if tr["values"]: grouped.setdefault(tr["task_count"],[]).append(tr["values"])
    for n, runs in grouped.items():
        length=min(map(len,runs)); mean=[sum(run[i] for run in runs)/len(runs) for i in range(length)]
        ax.plot(range(1,length+1),mean,label=f"{n} tasks")
    ax.set_title("Hybrid GA best objective by generation"); ax.set_xlabel("Generation"); ax.set_ylabel("Normalized weighted objective (lower is better)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"plots"/"ga_convergence.png",dpi=150); plt.close(fig)
    sample=json.loads(df.iloc[-1].vm_loads); fig,ax=plt.subplots(figsize=(8,5)); ax.bar([v.vm_id for v in default_vms()],sample); ax.set_title("VM load distribution (last recorded run)"); ax.set_xlabel("Virtual machine"); ax.set_ylabel("Share of busy execution time"); fig.tight_layout(); fig.savefig(out/"plots"/"vm_load_distribution.png",dpi=150); plt.close(fig)
    return records
