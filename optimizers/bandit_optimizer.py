from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import override


from utils.config_space_analysis import make_subspaces_by_conditions
import ConfigSpace
import copy

from amltk.optimization import Optimizer, Trial
from amltk.types import Space
if TYPE_CHECKING:
    from amltk.types import Config, Seed


@dataclass
class BanditTrialInfo:
    """The information about a Bandit Optimizer trial.

    Args:
        name: The name of the trial.
        trial_number: The number of the trial.
        config: The configuration sampled from the space.
    """

    name: str
    trial_number: int
    config: Config


class BanditOptimizer(Optimizer[BanditTrialInfo]):
    def __init__(
        self,
        space: Space = None,
        subsapce_optimizer: Any = None,
        subsapce_optimizer_parameters: dict | None = None,
        policy: Any = None,
        policy_parameters: dict | None = None,
        seed: Seed | None = None,
        initial_configs: list | None = None,
        limit_to_configs: list | None = None,
    ):
        self.space = space
        self.subsapce_optimizer = subsapce_optimizer
        self.subsapce_optimizer_parameters = subsapce_optimizer_parameters
        self.metrics = [subsapce_optimizer_parameters["metrics"]]
        self.bucket = subsapce_optimizer_parameters["bucket"]
        self.seed = seed
        self.t = 0
        self.subspace_optimizers = []
        self.subsapces, self.arms_names, self.parent_names = (
            make_subspaces_by_conditions(space)
        )
        self.number_of_arms = len(self.arms_names)

        self.initial_configs = initial_configs
        self.limit_to_configs = limit_to_configs

        for i, sub_space in enumerate(self.subsapces):
            if "n_configs" in self.subsapce_optimizer_parameters:
                if self.subsapce_optimizer_parameters["n_configs"] is None:
                    self.subsapce_optimizer_parameters["n_configs"] = (
                        self.subsapce_optimizer_parameters["n_trials"]
                        // (self.number_of_arms * 4)
                    )

            if self.initial_configs is not None:
                config = dict(self.initial_configs[i])
                del config[self.parent_names[i]]
                initial_configs = [
                    ConfigSpace.configuration_space.Configuration(
                        sub_space, values=config
                    )
                ]
                if "n_configs" in self.subsapce_optimizer_parameters:
                    self.subsapce_optimizer_parameters["n_configs"] -= 1

            if self.limit_to_configs is not None:
                limit_to_configs = []
                for config in self.limit_to_configs:
                    if self.parent_names[i] in config:
                        if config[self.parent_names[i]] == self.arms_names[i]:
                            new_config = copy.deepcopy(dict(config))
                            del new_config[self.parent_names[i]]
                            new_config = ConfigSpace.configuration_space.Configuration(
                                sub_space, values=new_config
                            )
                            limit_to_configs.append(new_config)
            self.subspace_optimizers.append(
                self.subsapce_optimizer(
                    space=sub_space,
                    **self.subsapce_optimizer_parameters,
                    initial_configs=initial_configs,
                    limit_to_configs=limit_to_configs,
                )
            )

        self.policy = policy(self.number_of_arms, **policy_parameters)
        self.selected_arm = self.policy.selected_arm
        self.last_trial_name = None

    @override
    def ask(self) -> Trial[BanditTrialInfo]:
        self.selected_arm = self.policy.play()
        trial = self.subspace_optimizers[self.selected_arm].ask()
        trial = copy.deepcopy(trial)
        self.last_trial_name = trial.name
        trial.name = (
            "exp"
            + str(self.t)
            + "-"
            + self.arms_names[self.selected_arm]
            + "-"
            + trial.name
        )
        trial.config[self.parent_names[self.selected_arm]] = self.arms_names[
            self.selected_arm
        ]
        return trial

    @override
    def tell(self, report: Trial.Report[BanditTrialInfo]) -> None:
        report = copy.deepcopy(report)
        arm_name = report.trial.name.split("-")[1]
        self.selected_arm = self.arms_names.index(arm_name)
        self.policy.update_cost(report.summary[self.metrics[0].name])
        # copy
        report.trial.name = report.trial.name.split("-")[-1]
        del report.trial.config[self.parent_names[self.selected_arm]]
        self.subspace_optimizers[self.selected_arm].tell(report)
        self.t += 1
