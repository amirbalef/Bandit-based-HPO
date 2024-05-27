"""A simple random search optimizer.

This optimizer will sample from the space provided and return the results
without doing anything with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec, override

from amltk.randomness import as_rng
from amltk.types import Space

from datetime import datetime

from collections.abc import Sequence

from amltk.store.paths.path_bucket import PathBucket

from amltk.optimization import Metric, Optimizer, Trial
import numpy as np


if TYPE_CHECKING:
    from amltk.types import Config, Seed

P = ParamSpec("P")
Q = ParamSpec("Q")
Result = TypeVar("Result")

MAX_INT = 2**32


@dataclass
class RSTrialInfo:
    """The information about a random search trial.

    Args:
        name: The name of the trial.
        trial_number: The number of the trial.
        config: The configuration sampled from the space.
    """

    name: str
    trial_number: int
    config: Config


class RandomSearch(Optimizer[RSTrialInfo]):
    """A random search optimizer."""

    def __init__(
        self,
        metrics: Metric | Sequence[Metric],
        space: Space,  # type: ignore
        bucket: PathBucket | None = None,
        seed: Seed | None = None,
        duplicates: bool = False,
        max_sample_attempts: int = 50,
        initial_configs: list | None = None,
        limit_to_configs: list | None = None,
    ):
        """Initialize the optimizer.

        Args:
            space: The space to sample from.
            seed: The seed to use for the sampler.
            duplicates: Whether to allow duplicate configurations.
            max_sample_attempts: The maximum number of attempts to sample a
                unique configuration. If this number is exceeded, an
                `ExhaustedError` will be raised. This parameter has no
                effect when `duplicates=True`.
        """
        self.space = space
        self.trial_count = 0
        metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        self.metrics = metrics
        self.seed = as_rng(seed) if seed is not None else None
        self.max_sample_attempts = max_sample_attempts
        self.bucket = (
            bucket
            if bucket is not None
            else PathBucket(f"{self.__class__.__name__}-{datetime.now().isoformat()}")
        )

        # We store any configs we've seen to prevent duplicates
        self._configs_seen: list[Config] | None = [] if not duplicates else None
        self.duplicates = duplicates
        self.initial_configs = [] if initial_configs is None else initial_configs
        self.limit_to_configs = limit_to_configs

    @override
    def ask(self) -> Trial[RSTrialInfo]:
        """Sample from the space.

        Raises:
            ExhaustedError: If the sampler is exhausted of unique configs.
                Only possible to raise if `duplicates=False` (default).
        """
        name = f"random-{self.trial_count}"
        if len(self.initial_configs) > 0:
            config = self.initial_configs.pop(0)
        else:
            if self.limit_to_configs is not None:
                config = self.limit_to_configs[
                    np.random.randint(0, len(self.limit_to_configs))
                ]
            else:
                config = self.space.sample_configuration()
                try_number = 1
                while config in self._configs_seen and self.duplicates is False:
                    config = self.space.sample_configuration()
                    if try_number >= self.max_sample_attempts:
                        break
                    try_number += 1

        if self._configs_seen is not None:
            self._configs_seen.append(config)

        info = RSTrialInfo(name, self.trial_count, config)
        trial = Trial(
            name=name,
            metrics=self.metrics,
            config=dict(config),
            info=info,
            seed=self.seed.integers(MAX_INT) if self.seed is not None else None,
            bucket=self.bucket,
        )
        self.trial_count = self.trial_count + 1
        return trial

    @override
    def tell(self, report: Trial.Report[RSTrialInfo]) -> None:
        """Do nothing with the report.

        ???+ note
            We do nothing with the report as it's random search
            and does not use the report to do anything useful.
        """
