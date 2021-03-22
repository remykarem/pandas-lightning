import pandas as pd
from scipy.stats import shapiro


@pd.api.extensions.register_series_accessor("tests")
class tests:
    def __init__(self, pandas_obj):
        # self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def is_normal(self, alpha=0.05):
        # Shapiro-Wilk Test
        stat, p = shapiro(self._obj)
        print(f"Statistics={stat:.3f}, p={p:.3f}")

        return p > alpha
