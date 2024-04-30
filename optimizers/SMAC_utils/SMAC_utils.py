from __future__ import annotations
import smac
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
from smac.initial_design import AbstractInitialDesign

import copy
import logging
import numpy as np

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
)

from collections import OrderedDict
import types

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant

class FixedSetRandomInitialDesign(AbstractInitialDesign):
    """Initial design that evaluates random configurations."""

    def __init__(self, limit_to_configs: list , *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.limit_to_configs = limit_to_configs

    def _select_configurations(self) -> list[Configuration]:
        configs_indx = self._rng.randint(0, len(self.limit_to_configs), size=self._n_configs)
        configs = [ self.limit_to_configs[i] for i in configs_indx]
        for config in configs:
            config.origin = "Initial Design: FixedSet Random"
        return configs

def select_configurations(initial_class) -> list[Configuration]:
    """Selects the initial configurations. Internally, `_select_configurations` is called,
    which has to be implemented by the child class.

    Returns
    -------
    configs : list[Configuration]
        Configurations from the child class.
    """
    
    configs: list[Configuration] = []

    # Adding additional configs
    configs += initial_class._additional_configs

    if initial_class._n_configs == 0:
        logger.info("No initial configurations are used.")
    else:
        configs += initial_class._select_configurations()

    for config in configs:
        if config.origin is None:
            config.origin = "Initial design"

    # Removing duplicates
    # (Reference: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)
    configs = list(OrderedDict.fromkeys(configs))
    logger.info(
        f"Using {len(configs) - len(initial_class._additional_configs)} initial design configurations "
        f"and {len(initial_class._additional_configs)} additional configurations."
    )

    #print("initial design")

    return configs



from typing import Callable, Iterator

from ConfigSpace import Configuration, ConfigurationSpace

from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.random_design.modulus_design import ModulusRandomDesign

class ChallengerList(Iterator):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    configspace : ConfigurationSpace
    challenger_callback : Callable
        Callback function which returns a list of challengers (without interleaved random configurations, must a be a
        python closure.
    random_design : AbstractRandomDesign | None, defaults to ModulusRandomDesign(modulus=2.0)
        Which random design should be used.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        challenger_callback: Callable,
        configurations: list,
        random_design: AbstractRandomDesign | None = ModulusRandomDesign(modulus=2.0),
        rng = None,
        previous_configs: list = None,
    ):
        self._challengers_callback = challenger_callback
        self._challengers: list[Configuration] | None = None
        self._configspace = configspace
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self._random_design = random_design
        self._configurations = configurations
        self._rng = rng
        self._previous_configs = previous_configs

    def __next__(self) -> Configuration:
        # If we already returned the required number of challengers
        if self._challengers is not None and self._index == len(self._challengers):
            raise StopIteration
        # If we do not want to have random configs, we just yield the next challenger
        elif self._random_design is None:
            if self._challengers is None:
                self._challengers = self._challengers_callback()

            config = self._challengers[self._index]
            self._index += 1

            return config
        # If we want to interleave challengers with random configs, sample one
        else:
            if self._random_design.check(self._iteration):
                configurations = [config for config in self._configurations if config not in self._previous_configs ]
                config =  configurations[self._rng.randint(0, len(configurations))]
                #config = self._configspace.sample_configuration()
                config.origin = "FixedSet Random Search"
            else:
                if self._challengers is None:
                    self._challengers = self._challengers_callback()

                config = self._challengers[self._index]
                self._index += 1
            self._iteration += 1

            return config

    def __len__(self) -> int:
        if self._challengers is None:
            self._challengers = self._challengers_callback()

        return len(self._challengers) - self._index

from smac.acquisition.maximizer.abstract_acqusition_maximizer import AbstractAcquisitionMaximizer
from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction

from smac.runhistory.runhistory import RunHistory

class FixedSet(AbstractAcquisitionMaximizer):
    def __init__(
        self,
        configurations: list[Configuration],
        acquisition_function: AbstractAcquisitionFunction,
        configspace: ConfigurationSpace,
        challengers: int = 5000,
        seed: int = 0,
    ):
        """Maximize the acquisition function over a finite list of configurations.
        Parameters
        ----------
        configurations : list[~smac._configspace.Configuration]
            Candidate configurations
        acquisition_function : ~smac.acquisition.AbstractAcquisitionFunction

        configspace : ~smac._configspace.ConfigurationSpace

        rng : np.random.RandomState or int, optional
        """
        super().__init__(
            acquisition_function=acquisition_function, configspace=configspace, challengers=challengers, seed=seed
        )
        self.configurations = configurations

    @override
    def maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int | None = None,
        random_design: AbstractRandomDesign | None = None,
    ) -> Iterator[Configuration]:
        """Maximize acquisition function using `_maximize`, implemented by a subclass.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previous evaluated configurations.
        n_points: int, defaults to None
            Number of points to be sampled. If `n_points` is not specified,
            `self._challengers` is used.
        random_design: AbstractRandomDesign, defaults to None
            Part of the returned ChallengerList such that we can interleave random configurations
            by a scheme defined by the random design. The method `random_design.next_iteration()`
            is called at the end of this function.

        Returns
        -------
        challengers : Iterator[Configuration]
            An iterable consisting of configurations.
        """

        print("maximize")
        if n_points is None:
            n_points = self._challengers

        def next_configs_by_acquisition_value() -> list[Configuration]:
            assert n_points is not None
            # since maximize returns a tuple of acquisition value and configuration,
            # and we only need the configuration, we return the second element of the tuple
            # for each element in the list
            return [t[1] for t in self._maximize(previous_configs, n_points)]

        challengers = ChallengerList(
            self._configspace,
            next_configs_by_acquisition_value,
            self.configurations,
            random_design,
            self._rng,
            previous_configs
        )
        print("challengers", challengers)

        if random_design is not None:
            random_design.next_iteration()

        return challengers
    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        configurations = [copy.deepcopy(config) for config in self.configurations if config not in previous_configs ]
        for config in configurations:
            config.origin = "Fixed Set"
        res = self._sort_by_acquisition_value(configurations)
        return res



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
        """Initialize the optimizer.

        Args:
            facade: The SMAC facade to use.
            bucket: The bucket given to trials generated by this optimizer.
            metrics: The metrics to optimize.
            fidelities: The fidelities to use, if any.
        """
        # We need to very that the scenario is correct incase user pass in
        # their own facade construction
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
        continue_from_last_run: bool = False,
        logging_level: int | Path | Literal[False] | None = False,
        acquisition_maximizer:AbstractAcquisitionMaximizer = None,
        acquisition_function:AbstractAcquisitionFunction = None,
        random_design = None,
        initial_configs=None,
        n_configs = None,
        n_trials = 100,

    ) -> Self:
        """Create a new SMAC optimizer using either the HPO facade or
        a mutli-fidelity facade.

        Args:
            space: The config space to optimize.
            metrics: The metrics to optimize.
            bucket: The bucket given to trials generated by this optimizer.
            deterministic: Whether the function your optimizing is deterministic, given
                a seed and config.
            seed: The seed to use for the optimizer.
            fidelities: The fidelities to use, if any.
            continue_from_last_run: Whether to continue from a previous run.
            logging_level: The logging level to use.
                This argument is passed forward to SMAC, use False to disable
                SMAC's handling of logging.
        """
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

        facade_cls: type[AbstractFacade]
        if fidelities:
            if len(fidelities) == 1:
                v = next(iter(fidelities.values()))
                min_budget, max_budget = v
            else:
                min_budget, max_budget = 1.0, 100.0

            scenario = Scenario(
                objectives=metric_names,
                configspace=space,
                output_directory=bucket.path / "smac3_output",
                seed=seed,
                min_budget=min_budget,
                max_budget=max_budget,
                crash_cost=cls.crash_cost(metrics),
            )
            facade_cls = MultiFidelityFacade
        else:
            scenario = Scenario(
                configspace=space,
                seed=seed,
                output_directory=bucket.path / "smac3_output",
                deterministic=deterministic,
                objectives=metric_names,
                n_trials = n_trials,
                crash_cost=cls.crash_cost(metrics),
            )
            facade_cls = HyperparameterOptimizationFacade
        if(acquisition_function!=None):
            model = RandomForest(
                        log_y=True,
                        n_trees=10,
                        bootstrapping=True,
                        ratio_features=1.0,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_depth= 2**20,
                        configspace=scenario.configspace,
                        instance_features=scenario.instance_features,
                        seed=scenario.seed
                    )
            acquisition_function._model = model
            acquisition_function.model = model
        else:
            model = None
        initial_design =  SobolInitialDesign(
            scenario=scenario,
            n_configs=n_configs,
            n_configs_per_hyperparameter=10,
            max_ratio= 0.25,
            additional_configs=initial_configs,)
        initial_design.select_configurations = types.MethodType(select_configurations, initial_design) 

        facade = facade_cls(
            scenario=scenario,
            target_function="dummy",  # NOTE: https://github.com/automl/SMAC3/issues/946
            overwrite=not continue_from_last_run,
            logging_level=logging_level,
            multi_objective_algorithm=facade_cls.get_multi_objective_algorithm(
                scenario=scenario,),
            acquisition_maximizer = acquisition_maximizer,
            acquisition_function = acquisition_function ,
            model = model,
            random_design= random_design,
            initial_design =initial_design
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
    def crash_cost(cls, metric: Metric) -> float:
        ...

    @overload
    @classmethod
    def crash_cost(cls, metric: Sequence[Metric]) -> list[float]:
        ...

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
