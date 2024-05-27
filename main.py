from __future__ import annotations
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="synth", help="dataset name")
    parser.add_argument(
        "--instance", nargs="?", default="All", help="instance of the dataset names"
    )
    parser.add_argument(
        "--optimizer", nargs="?", default="All", help="Random, SMAC, ..."
    )
    parser.add_argument("--output_root_dir", nargs="?", default="results/")
    parser.add_argument(
        "--iterations", nargs="?", default=200, type=int, help="number_of_iterations"
    )
    parser.add_argument("--trial_number", nargs="?", default=1, type=int, help="seed")
    parser.add_argument(
        "--n_worker", nargs="?", default=32, type=int, help="timeout for each run"
    )
    parser.add_argument(
        "--n_worker_scheduler",
        nargs="?",
        default=1,
        type=int,
        help="timeout for each run",
    )
    parser.add_argument(
        "--trial_timeout", nargs="?", default=0, help="timeout for each run"
    )
    args = parser.parse_args()

    if args.dataset == "TabRepo":
        from hpo_datasets.TabRepo import TabRepo

        dataset = TabRepo(context_name="D244_F3_C1416_30")
    elif args.dataset == "yahpo_gym":
        from hpo_datasets.yahpo_gym_dataset import yahpo_gym_dataset

        dataset = yahpo_gym_dataset()
    elif args.dataset == "synth":
        from hpo_datasets.synthetic_dataset import Synthetic_dataset

        dataset = Synthetic_dataset()
    else:
        exit()

    metrics = Metric("model_error", minimize=True)
    optimizer_method = args.optimizer

    instance_names = dataset.get_instances_list()
    if args.instance in instance_names:
        instance = args.instance
    else:
        print("The instance name is wrong and not in", instance_names)
        exit()

    dataset_name = args.dataset + "/" + instance

    trial_name = str(args.trial_number)
    seed = args.trial_number
    output_path = Path(
        args.output_root_dir
        + "/"
        + dataset_name
        + "/"
        + optimizer_method
        + "/"
        + trial_name
    )
    bucket = PathBucket(output_path, clean=True, create=True)
    limit_to_configs = None

    pipeline = dataset.get_pipeline(instance=instance)
    if isinstance(pipeline, tuple):
        pipeline, limit_to_configs = pipeline
    space = pipeline.search_space("configspace")
    initial_configs = make_initial_config(space)

    if optimizer_method == "RandomSearch":
        optimizer = RandomSearch(
            space=space,
            metrics=metrics,
            bucket=bucket,
            seed=seed,
            initial_configs=initial_configs,
            limit_to_configs=limit_to_configs,
        )
    elif optimizer_method == "SMAC":
        optimizer = SMACOptimizer.create(
            space=space,
            metrics=metrics,
            bucket=bucket,
            seed=seed,
            n_trials=args.iterations,
            initial_configs=initial_configs,
            n_configs=(args.iterations // 4) - len(initial_configs),
            limit_to_configs=limit_to_configs,
        )  #

    elif (
        "Bandit" in optimizer_method
    ):  # Bandit_suboptimzerName.suboptimzerParams_policyName.policyParams
        subsapce_optimizer_name = optimizer_method.split("_")[1].split(".")[0]
        subsapce_optimizer_parameters = {
            "metrics": metrics,
            "bucket": bucket,
            "seed": seed,
        }
        if subsapce_optimizer_name == "SMAC":
            subsapce_optimizer = SMACOptimizer.create
            subsapce_optimizer_parameters = {
                "metrics": metrics,
                "bucket": bucket,
                "seed": seed,
                "n_trials": args.iterations,
                "n_configs": None,
            }
        elif subsapce_optimizer_name == "RandomSearch":
            subsapce_optimizer = RandomSearch
            subsapce_optimizer_parameters = {
                "metrics": metrics,
                "bucket": bucket,
                "seed": seed,
            }

        policy_name = "_".join(optimizer_method.split("_")[2:]).split(".")[0]
        policy = getattr(
            importlib.import_module("optimizers.bandit_policies." + policy_name),
            policy_name,
        )
        policy_parameters = {}
        optimizer = BanditOptimizer(
            space=space,
            subsapce_optimizer=subsapce_optimizer,
            subsapce_optimizer_parameters=subsapce_optimizer_parameters,
            policy=policy,
            policy_parameters=policy_parameters,
            seed=seed,
            initial_configs=initial_configs,
            limit_to_configs=limit_to_configs,
        )
    else:
        print("error")
        exit()

    Experiment(
        pipeline=pipeline,
        optimizer=optimizer,
        iterations=args.iterations,
        output_path=output_path,
        bucket=bucket,
        n_worker_scheduler=args.n_worker_scheduler,
    ).run()

    print("Done, results are avaible in", output_path)
