import pytest
from cloud_scheduler.models import Task, VM
from cloud_scheduler.simulation import execution_time, feasible_assignment
from cloud_scheduler.algorithms import random_schedule, min_min_schedule, hybrid_ga_schedule, _repair
from cloud_scheduler.experiment import run_experiments

def fixture(): return [Task("a",100,2),Task("b",200,2)], [VM("x",100,3,.1),VM("y",200,3,.2)]

def test_models_and_execution():
    t,v=fixture(); assert execution_time(t[0],v[0])==1
    with pytest.raises(ValueError): VM("bad",0,1,1)

def test_feasibility_and_algorithms():
    t,v=fixture(); assert feasible_assignment(t,v,[0,1]); assert not feasible_assignment(t,v,[0,0])
    assert random_schedule(t,v,2).feasible; assert min_min_schedule(t,v).feasible

def test_ga_and_repair():
    t,v=fixture(); assert _repair([0,0],t,v)==[0,1]
    m=hybrid_ga_schedule(t,v,seed=1,population_size=6,generations=3)
    assert m.feasible and m.objective is not None

def test_infeasible_empty_single():
    with pytest.raises(ValueError): random_schedule([Task("huge",1,10)],[VM("x",1,2,1)])
    vm=[VM("x",10,4,1)]; assert min_min_schedule([],vm).feasible
    assert hybrid_ga_schedule([Task("one",10,1)],vm,population_size=2,generations=1).feasible

def test_experiment_outputs(tmp_path):
    rows=run_experiments(tmp_path,sizes=(2,),seeds=(7,),ga_options={"population_size":4,"generations":1})
    assert len(rows)==3 and (tmp_path/"csv"/"experiment_results.csv").exists()
