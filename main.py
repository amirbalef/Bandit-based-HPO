from __future__ import annotations
from asyncio import Future
from typing import Any
from pathlib import Path
import argparse

from amltk.optimization import Metric
from amltk.store import PathBucket

from experiment.experiment import Experiment
from utils.config_space_analysis import make_initial_config

from optimizers.SMAC import SMACOptimizer
from optimizers.random_search import RandomSearch
from optimizers.bandit_optimizer import BanditOptimizer

import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs='?',default="synth", help="dataset name")
parser.add_argument("--instance", nargs='?',default="All", help="instance of the dataset names")
parser.add_argument("--optimizer", nargs='?',default='All', help="Random, SMAC, ...")
parser.add_argument("--output_root_dir", nargs='?',default='results/')
parser.add_argument("--iterations", nargs='?',default=200, type=int, help='number_of_iterations')
parser.add_argument("--trial_number", nargs='?',default=1, type=int, help='starting seed')
parser.add_argument("--trials", nargs='?',default=32, type=int, help='number_of_runs')
parser.add_argument("--n_worker", nargs='?',default=32, type=int, help="timeout for each run")
parser.add_argument("--n_worker_scheduler", nargs='?',default=1, type=int, help="timeout for each run")
parser.add_argument("--multi_processing", nargs='?',default="None", help="Pool")
parser.add_argument("--trial_timeout", nargs='?',default=0, help="timeout for each run")
args = parser.parse_args()

if(args.dataset == "TabRepo"):
    from hpo_datasets.TabRepo import TabRepo 
    dataset = TabRepo(context_name = "D244_F3_C1416_30")
elif(args.dataset == "yahpo_gym"): 
    from hpo_datasets.yahpo_gym_dataset import yahpo_gym_dataset
    dataset = yahpo_gym_dataset()
elif(args.dataset == "synth"): 
    from hpo_datasets.synthetic_dataset import Synthetic_dataset
    dataset = Synthetic_dataset()
else:
    exit()

if( args.instance =='All'):
    instance_names = dataset.get_instances_list()
else:
    instance_names = [args.instance]

if(args.optimizer=='All'):
    optimizer_methods = [ 'Bandit_SMAC_ER_UCB_S', 'Bandit_SMAC_ER_UCB_N',
                          'Bandit_SMAC_QoMax_ETC', 'Bandit_SMAC_QoMax_SDA', 
                          'Bandit_SMAC_Q_BayesUCB', 'Bandit_SMAC_QuantileUCB', 
                          'Bandit_SMAC_BayesUCB', 'Bandit_SMAC_UCB', 
                          'Bandit_SMAC_Max_Median', 'Bandit_SMAC_Rising_Bandit', 
                          'Bandit_SMAC_Exp3', 'Bandit_SMAC_Successive_Halving', 
                          "SMAC","RandomSearch"]
else:
    optimizer_methods = [args.optimizer]

metrics=Metric("model_error", minimize=True)



def run_Expriment(pipeline, optimizer_method, metrics, dataset_name, i):
    trial_name = str(i)
    seed = i
    output_path = Path(args.output_root_dir + '/' +dataset_name+ '/' +optimizer_method+ '/' +trial_name)
    bucket = PathBucket(output_path, clean=True, create=True)
    limit_to_configs = None
    if isinstance(pipeline, tuple):
        pipeline, limit_to_configs = pipeline
    space= pipeline.search_space("configspace")
    initial_configs = make_initial_config(space)

    if(optimizer_method=='RandomSearch'):
        optimizer = RandomSearch(space= space ,metrics=metrics, bucket=bucket, seed=seed, initial_configs = initial_configs, limit_to_configs = limit_to_configs)
    elif(optimizer_method=="SMAC"):
        optimizer = SMACOptimizer.create(space=space, metrics=metrics, bucket=bucket, seed=seed, n_trials = args.iterations, initial_configs = initial_configs, n_configs=(args.iterations//4) -len(initial_configs), limit_to_configs = limit_to_configs) #
    
    elif("Bandit" in optimizer_method): #Bandit_suboptimzerName.suboptimzerParams_policyName.policyParams
        subsapce_optimizer_name = optimizer_method.split("_")[1].split(".")[0]
        subsapce_optimizer_parameters = {"metrics":metrics,"bucket":bucket, "seed":seed}
        if(subsapce_optimizer_name=="SMAC"):
            subsapce_optimizer = SMACOptimizer.create
            subsapce_optimizer_parameters = {"metrics":metrics,"bucket":bucket, "seed":seed, "n_trials":args.iterations, "n_configs":None}
        elif(subsapce_optimizer_name=="RandomSearch"):
            subsapce_optimizer = RandomSearch
            subsapce_optimizer_parameters = {"metrics":metrics,"bucket":bucket, "seed":seed}

        policy_name = "_".join(optimizer_method.split("_")[2:]).split(".")[0]
        policy = getattr(importlib.import_module("optimizers.bandit_policies."+policy_name),policy_name)
        policy_parameters = {}
        optimizer = BanditOptimizer(space= space,subsapce_optimizer=subsapce_optimizer,subsapce_optimizer_parameters=subsapce_optimizer_parameters, policy = policy ,policy_parameters=policy_parameters , seed=seed, initial_configs = initial_configs, limit_to_configs = limit_to_configs)
    else:
        print("error")
        exit()

    Experiment(pipeline = pipeline,
                optimizer = optimizer,
                iterations = args.iterations,
                output_path = output_path,
                bucket = bucket,
                n_worker_scheduler = args.n_worker_scheduler).run()
    

for instance in instance_names:
    pipeline  = dataset.get_pipeline(instance=instance )
    dataset_name = args.dataset + "/" + instance

    if(args.multi_processing == "joblib"):
        from joblib import Parallel, delayed
        Parallel(backend="threading", n_jobs= int(args.n_worker) )( delayed(run_Expriment)
                                                                    (pipeline, optimizer_method,metrics, dataset_name, i)
                                                                    for i in range(args.trial_number, args.trial_number + args.trials)
                                                                    for optimizer_method in optimizer_methods)
    elif(args.multi_processing == "Pool"):
        from concurrent.futures import ProcessPoolExecutor as Pool
        with Pool(max_workers=int(args.n_worker)) as outer_pool:
            for optimizer_method in optimizer_methods:
                print(optimizer_method, dataset_name)
                from functools import partial
                for expriment in outer_pool.map(partial(run_Expriment, pipeline, optimizer_method,metrics, dataset_name), range(args.trial_number, args.trial_number + args.trials)):
                    pass
    else:
        for optimizer_method in optimizer_methods:
            for i in range(args.trial_number, args.trial_number + args.trials):
                print(i, optimizer_method, dataset_name)
                run_Expriment(pipeline, optimizer_method, metrics, dataset_name, i)

print("Done")