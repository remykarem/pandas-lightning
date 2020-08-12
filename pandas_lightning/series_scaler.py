import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.errors import EmptyDataError


@pd.api.extensions.register_series_accessor("scaler")
class scaler:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")
        elif not is_numeric_dtype(_obj):
            raise ValueError("Series must be numeric")

    def standardize(self, ddof=1):
        """Standardize features by removing the mean and
        scaling to unit variance. Similar to scikit-learn's
        StandardScaler.

        Parameters
        ----------
        ddof : int, optional
            Degrees of freedom, by default 1

        Examples
        --------

        >>> import pandas as pd
        >>> import pandas_lightning
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.standardize()
        0   -1.264911
        1   -0.632456
        2    0.000000
        3    0.632456
        4    1.264911
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return (self._obj - self._obj.mean()) / self._obj.std(ddof=ddof)

    def minmax(self, feature_range=(0, 1)):
        """Transform series by scaling to a given range

        Parameters
        ----------
        feature_range : tuple, optional
            Desired range of transformed data, by default (0, 1)

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_lightning
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.minmax()
        0    0.00
        1    0.25
        2    0.50
        3    0.75
        4    1.00
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series.
        """
        feature_min, feature_max = feature_range

        max_val = max(self._obj)
        min_val = min(self._obj)

        std = (self._obj - min_val) / (max_val - min_val)
        scaled = std * (feature_max - feature_min) + feature_min

        return scaled

    def log1p(self):
        """Transform to log(1+x)

        Notes
        -----
        This transformation is numerically stable for small
        numbers compared to the log(x) transformation.

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_lightning
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.log1p()
        0    0.693147
        1    1.098612
        2    1.386294
        3    1.609438
        4    1.791759
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return np.log1p(self._obj)

    def expm1(self):
        """Transform to exp(x)-1

        Notes
        -----
        This transformation is numerically stable for small
        numbers compared to the exp(x) transformation.

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_lightning
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.expm1()
        0      1.718282
        1      6.389056
        2     19.085537
        3     53.598150
        4    147.413159
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return np.expm1(self._obj)


def standardize(series, ddof=1):
    return (series - series.mean()) / series.std(ddof=ddof)
