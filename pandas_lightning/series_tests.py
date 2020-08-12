import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.errors import EmptyDataError
from scipy.stats import shapiro


@pd.api.extensions.register_series_accessor("tests")
class scaler:
    def __init__(self, pandas_obj):
        # self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def is_normal(self, alpha=0.05):
        # Shapiro-Wilk Test
        stat, p = shapiro(self._obj)
        print(f"Statistics={stat:.3f}, p={p:.3f}")

        return p > alpha
