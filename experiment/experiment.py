from __future__ import annotations

from asyncio import Future
from typing import Any

from amltk.optimization import History, Trial
from amltk.pipeline import  Sequential
from amltk.scheduling import Scheduler
from amltk.store import PathBucket


class Experiment():
    def __init__(self,pipeline, optimizer, iterations, output_path, bucket, n_worker_scheduler): 
        self.remaining_tasks = iterations
        self.output_path = output_path
        self.bucket = bucket
        self.trial_history = History()
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.scheduler = Scheduler.with_processes(n_worker_scheduler)
        self.task =  self.scheduler.task(self.target_function)

    def run(self):
        @self.scheduler.on_start
        def launch_initial_tasks() -> None:
            if(self.remaining_tasks>0):
                trial = self.optimizer.ask()
                self.task.submit(trial, bucket=self.bucket, _pipeline=self.pipeline)
                self.remaining_tasks = self.remaining_tasks - 1
            
        @self.task.on_result
        def tell_optimizer(future: Future, report: Trial.Report) -> None:
                self.optimizer.tell(report)

        @self.task.on_result
        def add_to_history(future: Future, report: Trial.Report) -> None:
            self.trial_history.add(report)

        @self.task.on_result
        def launch_another_task(*_: Any) -> None:
            if(self.remaining_tasks>0):
                trial = self.optimizer.ask()
                self.task.submit(trial, bucket=self.bucket, _pipeline=self.pipeline)
                self.remaining_tasks = self.remaining_tasks - 1

        @self.scheduler.on_timeout
        def stop_scheduler() -> None:
            self.scheduler.stop()

        @self.task.on_exception 
        def stop_scheduler_on_exception(*_: Any) -> None:
            self.scheduler.stop()

        @self.task.on_cancelled
        def stop_scheduler_on_cancelled(_: Any) -> None:
            self.scheduler.stop()
        
        self.scheduler.run()
        history_df = self.trial_history.df()
        history_df.to_pickle(self.output_path.joinpath("result.pkl"))

    @staticmethod     
    def target_function(
        trial: Trial,
        bucket: PathBucket,
        _pipeline: Sequential,
    ) -> Trial.Report:
        # Configure the pipeline with the trial config before building it.
        configured_pipeline = _pipeline.configure(trial.config)
        sklearn_pipeline = configured_pipeline.build("sklearn") 
        # Fit the pipeline, indicating when you want to start the trial timing and error
        with trial.begin():
            sklearn_pipeline.fit(None, None)

        if trial.exception:
            trial.store(
                {
                    "exception.txt": f"{trial.exception}\n traceback: {trial.traceback}",
                    "config.json": dict(trial.config),
                }
            )
            return trial.fail()
        
        # Make our predictions with the model
        result =  sklearn_pipeline.predict(None)

        if isinstance(result, tuple):
            model_error, infos = result
            trial.summary.update(
                {
                    "model_error": model_error,
                    "model_infos": infos,
                },
            )
        else:
            model_error = result
            trial.summary.update(
                {
                    "model_error": model_error,
                },
            )
        # Save all of this to the file system
        #trial.store(
        #    {
        #        "config.json": dict(trial.config),
        #        "scores.json": trial.summary,
        #    },
        #)
        # Finally report the success
        return trial.success(model_error=model_error)