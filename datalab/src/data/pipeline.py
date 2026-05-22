"""Reusable transformation pipeline primitives for DataLab."""

from dataclasses import dataclass
from typing import Any, Callable, List

import pandas as pd


@dataclass
class PipelineStep:
    """Describe one named transformation step in a pipeline.

    Parameters
    - name (str): Human-readable step name used in logs and errors.
    - func (Callable[..., pandas.DataFrame]): Transformation callable
      that receives a DataFrame and returns a new DataFrame.
    - params (dict[str, Any]): Keyword arguments forwarded to ``func``.
    """

    name: str
    func: Callable[..., pd.DataFrame]
    params: dict[str, Any]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the step against ``df`` and return the transformed frame.

        Parameters
        - df (pandas.DataFrame): Input DataFrame passed to the step.

        Returns
        - pandas.DataFrame: The transformed DataFrame returned by the
          step function.

        Raises
        - TypeError: Propagated if the configured step callable cannot be
          invoked with the provided DataFrame and parameters.
        """
        return self.func(df, **self.params)


class Pipeline:
    def __init__(self):
        """Create an empty pipeline with no registered steps.

        The pipeline stores transformations in execution order so data
        can be replayed reproducibly from the original dataset.
        """
        self.steps: List[PipelineStep] = []

    def add_step(self, name: str, func, **params):
        """Append a new transformation step to the pipeline.

        Parameters
        - name (str): Human-readable step name.
        - func (Callable[..., pandas.DataFrame]): Transformation callable
          that receives a DataFrame and returns a DataFrame.
        - **params: Keyword arguments passed to ``func`` when the step is
          executed.

        Returns
        - Pipeline: The pipeline instance, allowing fluent chaining.
        """
        step = PipelineStep(name=name, func=func, params=params)
        self.steps.append(step)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the configured steps against ``df`` in order.

        Parameters
        - df (pandas.DataFrame): Input DataFrame to process.

        Returns
        - pandas.DataFrame: The final transformed DataFrame after every
          registered step has been applied.

        Raises
        - RuntimeError: Raised when an individual step fails. The original
          exception is chained so callers can inspect the root cause.
        """
        for step in self.steps:
            try:
                print(f"Running step {step} with {step.params}")
                df = step(df)
            except Exception as e:
                raise RuntimeError(f"Error in step {step.name}: {e}") from e
        return df

    def clear(self):
        """Remove every registered step from the pipeline.

        Returns
        - None: The pipeline is mutated in place.
        """
        self.steps.clear()
        print(f"Steps cleared")

    def remove_step(self, index: int) -> None:
        """Remove the step at ``index`` from the pipeline.

        Parameters
        - index (int): Zero-based position of the step to remove.

        Returns
        - None: The pipeline is mutated in place.

        Raises
        - IndexError: Raised when ``index`` does not point to an existing
          step.
        """
        self.steps.pop(index)

    def to_dict(self):
        """Serialize the pipeline configuration to dictionaries.

        Returns
        - list[dict[str, Any]]: A JSON-friendly description of each step
          containing the step name and its stored parameters.
        """
        return [
            {
                "name": steps.name,
                "params": steps.params,
            }
            for steps in self.steps
        ]
