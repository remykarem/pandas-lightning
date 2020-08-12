import pandas as pd
import numpy as np
# from pandas.api.types import is_datetime64_dtype
from pandas.errors import EmptyDataError


@pd.api.extensions.register_series_accessor("dtpp")
class vectorised:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __add__(self, rhs):
        if isinstance(rhs, str):
            rhs_arr = rhs.split()
            if len(rhs_arr) != 2:
                raise ValueError

            quantity, unit = rhs_arr
            quantity = int(quantity)
            return self._obj + pd.to_timedelta(quantity, unit=unit)
        else:
            raise ValueError

    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __sub__(self, rhs):
        tz = pd.to_datetime(self._obj).dt.tz
        diff = self._obj - pd.to_datetime(rhs).tz_localize(tz)
        return diff

    def __rsub__(self, lhs):
        tz = pd.to_datetime(self._obj).dt.tz
        diff = pd.to_datetime(lhs).tz_localize(tz) - self._obj
        return diff
