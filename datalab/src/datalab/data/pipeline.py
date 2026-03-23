import pandas as pd
from dataclasses import dataclass
from typing import Callable, Any, List


@dataclass
class PipelineStep:
    name: str
    func: Callable[..., pd.DataFrame]
    params: dict[str, Any]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.func(df, **self.params)


class Pipeline:
    def __init__(self):
        self.steps: List[PipelineStep] = []

    def add_step(self, name: str, func, **params):
        step = PipelineStep(name=name, func=func, params=params)
        self.steps.append(step)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            try:
                print(f"Running step {step} with {step.params}")
                df = step(df)
            except Exception as e:
                raise RuntimeError(f"Error in step {step.name}: {e}") from e
        return df

    def clear(self):
        self.steps.clear()
        print(f"Steps cleared")

    def remove_step(self, index: int) -> None:
        self.steps.pop(index)

    def to_dict(self):
        return [
            {
                "name": steps.name,
                "params": steps.params
            }
            for steps in self.steps
        ]
