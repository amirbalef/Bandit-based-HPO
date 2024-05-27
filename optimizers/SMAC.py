from __future__ import annotations
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload
from typing_extensions import override

import numpy as np
from more_itertools import first_true
from pynisher import MemoryLimitException, TimeoutException
from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario
from smac.runhistory import (
    StatusType,
    TrialInfo as SMACTrialInfo,
    TrialValue as SMACTrialValue,
)

from amltk.optimization import Metric, Optimizer, Trial
from amltk.pipeline import Node
from amltk.randomness import as_int
from amltk.store import PathBucket

if TYPE_CHECKING:
    from typing_extensions import Self

    from ConfigSpace import ConfigurationSpace
    from smac.facade import AbstractFacade

    from amltk.types import FidT, Seed

from smac.model.random_forest.random_forest import RandomForest

logger = logging.getLogger(__name__)
from smac.initial_design.sobol_design import SobolInitialDesign

import logging
import numpy as np

from ConfigSpace import ConfigurationSpace
import types

from optimizers.SMAC_utils.SMAC_utils import (
    FixedSet,
    select_configurations,
    FixedSetRandomInitialDesign,
)
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.main.config_selector import ConfigSelector


class SMACOptimizer(Optimizer[SMACTrialInfo]):
    """An optimizer that uses SMAC to optimize a config space."""

    def __init__(
        self,
        *,
        facade: AbstractFacade,
        bucket: PathBucket | None = None,
        metrics: Metric | Sequence[Metric],
        fidelities: Mapping[str, FidT] | None = None,
    ) -> None:
        assert self.crash_cost(metrics) == facade.scenario.crash_cost
        metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        super().__init__(metrics=metrics, bucket=bucket)
        self.facade = facade
        self.metrics = metrics
        self.fidelities = fidelities

    @classmethod
    def create(
        cls,
        *,
        space: ConfigurationSpace | Node,
        metrics: Metric | Sequence[Metric],
        bucket: PathBucket | str | Path | None = None,
        deterministic: bool = True,
        seed: Seed | None = None,
        fidelities: Mapping[str, FidT] | None = None,
        logging_level: int | Path | Literal[False] | None = False,
        n_configs=None,
        n_trials=100,
        initial_configs=None,
        limit_to_configs=None,
    ) -> Self:
        seed = as_int(seed)
        match bucket:
            case None:
                bucket = PathBucket(
                    f"{cls.__name__}-{datetime.now().isoformat()}",
                )
            case str() | Path():
                bucket = PathBucket(bucket)
            case bucket:
                bucket = bucket  # noqa: PLW0127

        # NOTE SMAC always minimizes! Hence we make it a minimization problem
        metric_names: str | list[str]
        if isinstance(metrics, Sequence):
            metric_names = [metric.name for metric in metrics]
        else:
            metric_names = metrics.name

        if isinstance(space, Node):
            space = space.search_space(parser=cls.preferred_parser())

        scenario = Scenario(
            configspace=space,
            seed=seed,
            output_directory=bucket.path / "smac3_output",
            deterministic=deterministic,
            objectives=metric_names,
            n_trials=n_trials,
            crash_cost=cls.crash_cost(metrics),
        )
        facade_cls = HyperparameterOptimizationFacade
        if limit_to_configs != None:
            model = RandomForest(
                log_y=True,
                n_trees=10,
                bootstrapping=True,
                ratio_features=1.0,
                min_samples_split=2,
                min_samples_leaf=1,
                max_depth=2**20,
                configspace=scenario.configspace,
                instance_features=scenario.instance_features,
                seed=scenario.seed,
            )

            from smac.acquisition.function.expected_improvement import EI

            acquisition_function = EI(log=True)
            acquisition_function._model = model
            acquisition_function.model = model

            initial_design = FixedSetRandomInitialDesign(
                limit_to_configs,
                scenario=scenario,
                n_configs=n_configs,
                n_configs_per_hyperparameter=10,
                max_ratio=0.25,
                additional_configs=initial_configs,
            )
            initial_design.select_configurations = types.MethodType(
                select_configurations, initial_design
            )

            acquisition_maximizer = FixedSet(
                configspace=scenario.configspace,
                configurations=limit_to_configs,
                acquisition_function=acquisition_function,
                seed=seed,
            )
            random_design = ProbabilityRandomDesign(probability=0.2)

            config_selector = ConfigSelector(
                scenario, retrain_after=1, retries=16 + len(limit_to_configs)
            )

        else:
            model = None
            acquisition_maximizer = None
            acquisition_function = None
            random_design = None
            config_selector = ConfigSelector(
                scenario, retrain_after=1, retries=16 + 200
            )
            initial_design = SobolInitialDesign(
                scenario=scenario,
                n_configs=n_configs,
                n_configs_per_hyperparameter=10,
                max_ratio=0.25,
                additional_configs=initial_configs,
            )
            initial_design.select_configurations = types.MethodType(
                select_configurations, initial_design
            )

        facade = facade_cls(
            scenario=scenario,
            target_function="dummy",  # NOTE: https://github.com/automl/SMAC3/issues/946
            overwrite=True,
            logging_level=logging_level,
            multi_objective_algorithm=facade_cls.get_multi_objective_algorithm(
                scenario=scenario,
            ),
            acquisition_maximizer=acquisition_maximizer,
            acquisition_function=acquisition_function,
            model=model,
            random_design=random_design,
            initial_design=initial_design,
            config_selector=config_selector,
        )
        return cls(facade=facade, fidelities=fidelities, bucket=bucket, metrics=metrics)

    @override
    def ask(self) -> Trial[SMACTrialInfo]:
        """Ask the optimizer for a new config.

        Returns:
            The trial info for the new config.
        """
        smac_trial_info = self.facade.ask()
        config = smac_trial_info.config
        budget = smac_trial_info.budget
        instance = smac_trial_info.instance
        seed = smac_trial_info.seed

        if self.fidelities and budget:
            if len(self.fidelities) == 1:
                k, _ = next(iter(self.fidelities.items()))
                trial_fids = {k: budget}
            else:
                trial_fids = {"budget": budget}
        else:
            trial_fids = None

        config_id = self.facade.runhistory.config_ids[config]
        unique_name = f"{config_id=}_{seed=}_{budget=}_{instance=}"
        trial: Trial[SMACTrialInfo] = Trial(
            name=unique_name,
            config=dict(config),
            info=smac_trial_info,
            seed=seed,
            fidelities=trial_fids,
            bucket=self.bucket,
            metrics=self.metrics,
        )
        logger.debug(f"Asked for trial {trial.name}")
        return trial

    @override
    def tell(self, report: Trial.Report[SMACTrialInfo]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        assert report.trial.info is not None

        cost: float | list[float]
        match self.metrics:
            case [metric]:  # Single obj
                val: Metric.Value = first_true(
                    report.metric_values,
                    pred=lambda m: m.metric == metric,
                    default=metric.worst,
                )
                cost = self.cost(val)
            case metrics:
                # NOTE: We need to make sure that there sorted in the order
                # that SMAC expects, with any missing metrics filled in
                _lookup = {v.metric.name: v for v in report.metric_values}
                cost = [
                    self.cost(_lookup.get(metric.name, metric.worst))
                    for metric in metrics
                ]

        logger.debug(f"Telling report for trial {report.trial.name}")

        # If we're successful, get the cost and times and report them
        params: dict[str, Any]
        match report.status:
            case Trial.Status.SUCCESS:
                params = {
                    "time": report.time.duration,
                    "starttime": report.time.start,
                    "endtime": report.time.end,
                    "cost": cost,
                    "status": StatusType.SUCCESS,
                }
            case Trial.Status.FAIL:
                params = {
                    "time": report.time.duration,
                    "starttime": report.time.start,
                    "endtime": report.time.end,
                    "cost": cost,
                    "status": StatusType.CRASHED,
                }
            case Trial.Status.CRASHED | Trial.Status.UNKNOWN:
                params = {
                    "cost": cost,
                    "status": StatusType.CRASHED,
                }

        match report.exception:
            case None:
                pass
            case MemoryLimitException():
                params["status"] = StatusType.MEMORYOUT
                params["additional_info"] = {
                    "exception": str(report.exception),
                    "traceback": report.traceback,
                }
            case TimeoutException():
                params["status"] = StatusType.TIMEOUT
                params["additional_info"] = {
                    "exception": str(report.exception),
                    "traceback": report.traceback,
                }
            case _:
                params["additional_info"] = {
                    "exception": str(report.exception),
                    "traceback": report.traceback,
                }

        self.facade.tell(report.trial.info, value=SMACTrialValue(**params), save=True)

    @override
    @classmethod
    def preferred_parser(cls) -> Literal["configspace"]:
        """The preferred parser for this optimizer."""
        return "configspace"

    @overload
    @classmethod
    def crash_cost(cls, metric: Metric) -> float: ...

    @overload
    @classmethod
    def crash_cost(cls, metric: Sequence[Metric]) -> list[float]: ...

    @classmethod
    def crash_cost(cls, metric: Metric | Sequence[Metric]) -> float | list[float]:
        """Get the crash cost for a metric for SMAC."""
        match metric:
            case Metric(bounds=(lower, upper)):  # Bounded metrics
                return abs(upper - lower)
            case Metric():  # Unbounded metric
                return np.inf
            case metrics:
                return [cls.crash_cost(m) for m in metrics]

    @classmethod
    def cost(cls, value: Metric.Value) -> float:
        """Get the cost for a metric value for SMAC."""
        match value.distance_to_optimal:
            case None:  # If we can't compute the distance, use the loss
                return value.loss
            case distance:  # If we can compute the distance, use that
                return distance
