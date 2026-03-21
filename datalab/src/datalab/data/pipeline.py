import data_cleaner as dc
import data_loader as dl
import pandas as pd


class Pipeline:
    def __init__(self, name: str):
        self.name = name
